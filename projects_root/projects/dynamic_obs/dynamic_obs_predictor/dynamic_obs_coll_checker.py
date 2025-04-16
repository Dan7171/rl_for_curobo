from curobo.geom.sdf.world import CollisionQueryBuffer
from curobo.geom.sdf.world_mesh import WorldMeshCollision
from curobo.geom.types import WorldConfig
from curobo.types.math import Pose
from typing import List
import numpy as np
import copy
import torch
from curobo.geom.sdf.world import WorldCollisionConfig
from projects_root.utils.quaternion import integrate_quat
from projects_root.projects.dynamic_obs.dynamic_obs_predictor.utils import shift_tensor_left, mask_decreasing_values
from projects_root.projects.dynamic_obs.dynamic_obs_predictor.obstacle import Obstacle
from concurrent.futures import ProcessPoolExecutor, wait

SPHERE_DIM = 4 # The values which are required to define a sphere: x,y,z,radius. That's fixed in all curobo coll checkers

class DynamicObsCollPredictor:
    """
    
    This class is used to predict and compute costs for the collision of the dynamic obstacles in the world.
    You can think of this class as a collection of H collision checkers, one for each time step in the horizon + functionality to compute costs for the MPC (see cost_fn function).
    Each individual collision checker which it contains, is inspired by: https://curobo.org/get_started/2c_world_collision.html
    """
    

    # def __init__(self, tensor_args, world_cfg_template:WorldConfig , cache, step_dt_traj_mpc, H=30, n_rollouts=400, activation_distance= 0.05,robot_collision_sphere_num=65, cost_weight=100000, shift_cost_matrix_left=True, mask_decreasing_cost_entries=True):
    def __init__(self, tensor_args, obstacle_list:List[Obstacle] , cache, step_dt_traj_mpc, H=30, n_rollouts=400, activation_distance= 0.05,robot_collision_sphere_num=65, cost_weight=100000, shift_cost_matrix_left=True, mask_decreasing_cost_entries=True):
        """ Initialize H dynamic obstacle collision checker, for each time step in the horizon, 
        as well as setting the cost function parameters for the dynamic obstacle cost function.


        Args:
            tensor_args: pytorch tensor arguments.
            cache (dict): collision checker cache for the pre-defined dynamic primitives.
            step_dt_traj_mpc (float): Time passes between each step in the trajectory. This is what the mpc assumes time delta between steps in horizon is.
            H (int, optional): Defaults to 30. The horizon length. TODO: Should be taken from the mpc config.
            n_rollouts (int, optional): Defaults to 400. The number of rollouts. TODO: Should be taken from the mpc config.
            robot_collision_sphere_num: The number of collision spheres (the robot is approximated as spheres when calculating collision. NOTE: 65 is in franka,)
            activation_distance:  The distance in meters between a robot sphere and an obstacle over all obstacles, at which the collision cost starts to be "active" (positive). Denoted as "etha" or "activation distance" in the paper section 3.3.
            cost_weight: weight for the dynamic obstacle cost function (cost term weight). This is a hyper-parameter, unlike the weight_col_check which you should leave as 1. Default value is 100000, as by the original primitive collision cost weight of the mpc.
            shift_cost_matrix_left: if True, the cost matrix will be shifted left by one (to charge for the action that leads to the collision, and not for the state you are in collision).
            mask_decreasing_cost_entries: if True, the cost matrix will be modified so that only actions which take the robot closer to collision (i.e, the curobo cost function is increasing or distance to obstacle is decreasing) are considered and be charged at the dynamic cost function, and if an action is taking it away from a collision, it is not charged.
            mask_decreasing_cost_entries: if True, the cost matrix will be modified so that only the first violation of the safety margin is considered.
            """
        self._init_counter = 0 # number of steps until cuda graph initiation. Here Just for debugging. Can be removed. with no effect on the code. 
        self.weight_col_check = tensor_args.to_device([1]) # Leave 1. This is the weight for curobo collision checking. Explained in the paper. Since we are only interested at wether there is a collision or not (i.e. cost is positive in collision and 0 else), this is not important. See https://curobo.org/get_started/2c_world_collision.html        
        self.tensor_args = tensor_args 
        self.H = H 
        self.collision_sphere_num = robot_collision_sphere_num 
        self.n_rollouts = n_rollouts 
        self.step_dt_traj_mpc = step_dt_traj_mpc 
        self.activation_distance = tensor_args.to_device([activation_distance]) 
        self.cost_weight = cost_weight
        self.shift_cost_matrix_left = shift_cost_matrix_left
        self.mask_decreasing_cost_entries = mask_decreasing_cost_entries
        self._obstacles_predicted_paths = {} # dictionary of predicted paths for each obstacle

        # Make H copies of the collision supported world model template

        # First we make templates:
        cache_cp = copy.deepcopy(cache) # so we won't modify the original cache
        
        # Make a template of curobo world model (curobo representation of world) and inject the curobo representation of the obstacles into it (we will use it to make the template collision checker configuration).
        cu_world_model_template = WorldConfig()
        for obs in obstacle_list:
            obs.inject_curobo_obs(cu_world_model_template)
        
        # Convert the curobo world model template to a collision supported world model template.
        cu_coll_supported_world_model_template = WorldCollisionConfig(tensor_args, cu_world_model_template, cache_cp) # base template for collision checkers. We'll use this to make H copies.
        
        # Now we can make H copies:
        query_buffer_shape = torch.zeros((self.n_rollouts, 1, self.collision_sphere_num, SPHERE_DIM), device=self.tensor_args.device, dtype=self.tensor_args.dtype).shape # torch.Size([self.n_rollouts, 1, self.collision_sphere_num, SPHERE_DIM]) # n_rollouts x 1(current time step) x num of coll spheres x sphere_dim (x,y,z,radius) https://curobo.org/get_started/2c_world_collision.html
        self.H_collision_buffers = [] # not sure what they are for but they are needed.
        self.H_world_cchecks = [] # list of collision checkers (one checker for each time step in the horizon)
        for _ in range(self.H):
            world_collision_cfg_h = copy.deepcopy(cu_coll_supported_world_model_template) # NOTE: I use copies to avoid side effects. We duplicate the 
            col_checker_h = WorldMeshCollision(world_collision_cfg_h) # Initiate a single collision checker for each time step over the horizon.
            self.H_world_cchecks.append(col_checker_h)
            self.H_collision_buffers.append(CollisionQueryBuffer.initialize_from_shape(query_buffer_shape, self.tensor_args, col_checker_h.collision_types)) # I dont know yet what they are for 
        

    def predict_obstacle_path(self, cur_obstacle_pos:np.ndarray, cur_obstacle_lin_vel:np.ndarray, cur_obstacle_lin_acceleration:np.ndarray, cur_obstacle_orientation:np.ndarray, cur_obstacle_angular_vel:np.ndarray, cur_obstacle_angular_acceleration:np.ndarray):
        """
        Predict the path of the obstacle over the next H steps.
        """
        # path = np.ndarray(shape=(self.H, 3), dtype=np.float32)
        path = np.ndarray(shape=(self.H, 7), dtype=np.float32)
        for h in range(self.H):
            x_lin_vel = cur_obstacle_lin_vel * self.step_dt_traj_mpc * h
            x_lin_acc = cur_obstacle_lin_acceleration * self.step_dt_traj_mpc * h
            step_h_predicted_pos = cur_obstacle_pos + x_lin_vel + 0.5 * x_lin_acc ** 2 
            
            step_h_predicted_orientation = integrate_quat(cur_obstacle_orientation, cur_obstacle_angular_vel, self.step_dt_traj_mpc * h)
            # TODO: the angular acceleration is not taken into account. Need to implement this sometime.
            step_h_predicted_pose = np.concatenate([step_h_predicted_pos, step_h_predicted_orientation])
                
            path[h] = step_h_predicted_pose

        return path
    
    def _update_cchecker(self, cchecker, new_poses, obs_names):
        """
        Update the collision checker with the predicted positions of the obstacles.
        """
        # new_quaterion = np.array([1,0,0,0]) # TODO: This is temporary. We need to get the correct orientation of the obstacle.
        for obs_idx in range(len(obs_names)): # NOTE: could be parallelized
            pos_tensor = self.tensor_args.to_device(torch.from_numpy(new_poses[:,:3][obs_idx])) # x,y,z
            rot_tensor = self.tensor_args.to_device(torch.from_numpy(new_poses[:,3:][obs_idx])) # quaternion in scalar-first (w, x, y, z). https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.core/docs/index.html?highlight=get_world_pose#core-omni-isaac-core
            new_pose = Pose(pos_tensor, rot_tensor) 
            # NOTE: I added update_obstacle_pose_in_world_model before update_obstacle_pose only after I realized that there is a chance that update_obstacle_pose is not working, or at least not updating the CPU. I dont know if they are both neede, but now the pose in the cpu is updated too.
            # TODO: After I manage to give rise to awarness to obstacles in the cost and see change in behaviour, I should remove one of the next two calls, if redundant. For now they are here only to be on the safe side.
            cchecker.update_obstacle_pose_in_world_model(pose=new_pose, name=obs_names[obs_idx]) 
            cchecker.update_obstacle_pose(name=obs_names[obs_idx], w_obj_pose=new_pose, update_cpu_reference=True) 
    
    def _get_predicted_pose_at_step_h(self, h:int, obstacle_name:str):
        """
        Get the pose of the obstacle in the collision checker.
        """
        return self.H_world_cchecks[h].world_model.objects[obstacle_name].pose
    
    def update_predictive_collision_checkers(self, obstacles:List[Obstacle]):
        """
        First, For each object, predict its path (future positions) over the next H time steps.
        Then, for each time step, update the collision checker with the predicted positions of the objects.
        https://curobo.org/get_started/2c_world_collision.html#attach-object-note
        """
        def _process_single_step(args):
            self, h, obs_pose_preds, obs_names = args
            step_h_cchecker = self.H_world_cchecks[h]
            step_h_pose_predictions_all_rollouts = obs_pose_preds[:, h, :]
            self._update_cchecker(step_h_cchecker, step_h_pose_predictions_all_rollouts, obs_names)

        # 1) Get the names of the objects in the collision checker at time step 0. Order of the objects is important but remains the same for all collision checkers.
        obs_at_0 = self.H_world_cchecks[0].world_model.objects
        obs_names = [obj.name for obj in obs_at_0] 
        
        
        # 2) Predict the path of the obstacle over the next H steps. Result is a tensor of shape (len(obstacles), H, 3): 
        # Initialize a tensor to store the predicted positions of the obstacles over the next H time steps.
        # obs_pos_preds = np.ndarray(shape=(len(obstacles), self.H, 3), dtype=np.float32)
        obs_pose_preds = np.ndarray(shape=(len(obstacles), self.H, 7), dtype=np.float32)
        for ob_idx, obstacle in enumerate(obstacles):
            
            # Get the current position velocity and acceleration of the obstacle
            cur_position = obstacle.simulation_representation.get_world_pose()[0]
            cur_lin_vel = obstacle.simulation_representation.get_linear_velocity()
            # cur_lin_acceleration = obstacle.simulation_representation.get_rigid_body_linear_acceleration()
            cur_lin_acceleration = np.zeros(3) # TODO: THIS IS TEMPORARY, NEED TO IMPLEMENT THIS CORRECTLY
            
            cur_orientation = obstacle.simulation_representation.get_world_pose()[1] # quaternion is scalar-first (w, x, y, z). https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.core/docs/index.html?highlight=get_world_pose#core-omni-isaac-core
            cur_angular_vel = obstacle.simulation_representation.get_angular_velocity() # Get the angular velocity of the root articulation prim
            cur_angular_acceleration = np.zeros(3) # TODO: THIS IS TEMPORARY, NEED TO IMPLEMENT THIS CORRECTLY

            # Predict the path of the obstacle over the next H time steps. Result is a tensor of shape (H, 3)
            predicted_obstacle_path = self.predict_obstacle_path(cur_position, cur_lin_vel, cur_lin_acceleration, cur_orientation, cur_angular_vel, cur_angular_acceleration) 
            
            # Store the predicted positions of the obstacle in the tensor
            obs_pose_preds[ob_idx] = predicted_obstacle_path
            self._obstacles_predicted_paths[obstacle.name] = predicted_obstacle_path # store the predicted path for the obstacle in the dictionary. This is used for debugging.
        
        # 3) Update each collision checker with the predicted positions of the obstacles in its time step.
        # from multiprocessing import Pool
        for h in range(self.H): # NOTE: We can parallelize this loop.
            step_h_cchecker = self.H_world_cchecks[h] # h'th collision checker
            step_h_pose_predictions_all_rollouts = obs_pose_preds[:, h, :] # predicted positions of all obstacles at time step h
            self._update_cchecker(step_h_cchecker, step_h_pose_predictions_all_rollouts, obs_names) # update the collision checker with the predicted positions of the obstacles
        
        # with ProcessPoolExecutor() as executor:
        #     futures = [
        #     executor.submit(_process_single_step, h, obs_pose_preds, obs_names) for h in range(self.H)
        # ]
        # wait(futures)  # Explicitly wait for all futures to complete

        
        # with Pool() as pool:
        #     args = [(self, h, obs_pose_preds, obs_names) for h in range(self.H)]
        #     pool.map(_process_single_step, args)

            
        print("debug updated collision checkers")
    def cost_fn(self, robot_spheres, env_query_idx=None, method='curobo_prim_coll_cost_fn', binary=False):
        """
        Compute the collision cost for the robot spheres. Called by the MPC cost function (in ArmBase).
        Args:
            robot_spheres: tensor of shape [n_rollouts, horizon, n_spheres, 4]. Collision spheres of the robot. This is the standard structure of the input to the cost function.
            
            env_query_idx: optional index for querying specific environments. If None, the collision cost will be computed for the one and only environment in the collision checker.
            
            method: the way to compute the collision cost. Currently only curobo_prim_coll_cost_fn is implemented. By curobo_prim_coll_cost_fn, we mean the collision cost function implemented in curobo, which broadlly discussed at https://curobo.org/get_started/2c_world_collision.html#:~:text=process%20is%20illustrated.-,Collision%20Metric,-%C2%B6 (and in the technical report https://curobo.org/reports/curobo_report.pdf at section 3.3).
            NOTE: at this point, the method using signed distance is not used, since I failed to make it work.

            binary: if True, cost[r,h] will be self.cost_weight if any of the robot spheres are too close to an obstacle (i.e, their "safety zone" is violated). Otherwise it will be 0.
            If False, cost[r,h] will be the sum of the collision costs over all spheres. For now it seems that when its False, the overall performance is better, so I kept it as that.
        
        Returns:
            tensor of shape [n_rollouts, horizon] containing collision costs (we denot the returned tensor by the matrix named "dynamic_coll_cost_matrix" 
            where dynamic_coll_cost_matrix[r,h] is the  predictrive collision cost for the robot for rollout r and time step h, where r is the rollout index and h is the time step index)
        """
        
        if not torch.cuda.is_current_stream_capturing():
            self._init_counter += 1
            print(f"Initialization iteration {self._init_counter} (ignore this printing if not use_cuda_graph is False. Intializtion refers to the cuda graph capture. TODO: If use_cuda_graph is False, this should not be printed.)")
            if torch.cuda.is_current_stream_capturing():
                print("During graph capture")

        # Initialize cost tensor
        dynamic_coll_cost_matrix = torch.zeros(self.n_rollouts, self.H, device=self.tensor_args.device)
        # Make input contiguous and ensure proper shape
        robot_spheres = robot_spheres.contiguous() # Returns a contiguous in memory tensor containing the same data as self tensor. If self tensor is already in the specified memory format, this function returns the self tensor. https://pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html
        for h in range(self.H):
            buffer_step_h = self.H_collision_buffers[h]
            
            # Extract and reshape spheres for current timestep
            robot_spheres_step_h = robot_spheres[:, h].contiguous() # From [n_rollouts, H, n_spheres, SPHERE_DIM] to [n_rollouts, n_spheres, SPHERE_DIM]. We now focus on rollouts only from the time step "t+h" over the horizon (horizon starts at t, meaning h=0).
            robot_spheres_step_h = robot_spheres_step_h.reshape(self.n_rollouts, 1, self.collision_sphere_num, SPHERE_DIM) # From [n_rollouts, n_spheres, SPHERE_DIM] to [n_rollouts, 1, n_spheres, SPHERE_DIM] 
            
            if method == 'curobo_prim_coll_cost_fn': # NOTE: currently the only method implemented. There is an option to use the sdfs but I failed to make it work, I suspect there is a bug there
                spheres_curobo_coll_costs = self.H_world_cchecks[h].get_sphere_distance( # NOTE: although the method is called get_sphere_distance, it actually returns the "curobo collision cost" (see section 3.3 in paper).
                    query_sphere=robot_spheres_step_h, 
                    collision_query_buffer=buffer_step_h,
                    activation_distance=self.activation_distance,
                    weight=self.weight_col_check,
                    # compute_esdf=True # https://curobo.org/_api/curobo.geom.sdf.world.html#curobo.geom.sdf.world.WorldPrimitiveCollision.enable_obb:~:text=compute_esdf%20%E2%80%93%20Compute%20Euclidean%20signed%20distance%20instead%20of%20collision%20cost.%20When%20True%2C%20the%20returned%20tensor%20will%20be%20the%20signed%20distance%20with%20positive%20values%20inside%20an%20obstacle%20and%20negative%20values%20outside%20obstacles.
                ) 
                # NOTE: 1 "spheres_curobo_coll_costs" is a tensor holding a "curobo collision cost" (see section 3.3 in paper).
                # To make things simple, let's call it "curobo collision cost".
                # If that cost is positive, even if by a small margin, it means the minimal actual distance between the sphere and any of the obstacles (over all obstacles) is less than the the "activation distance" etha.
                # So one should set the activation distance to be the safety margin, i.e. the distance between some robot sphere and the nearest obstacle, in which if an obstacle is present (meaning that its somwhere in an imaginary sphere with radius etha around the robot sphere (approximating some rigid part of the robot)), you want to consider collision.
                # Another thing to know: if you want to know if there is an actuall collision (real contact, not just a safety margin violation), you can know that by checking if the "curobo collision cost" is greater or equal to  0.5 * self.activation_distance. See the paper section 3.3.
                spheres_curobo_coll_costs = spheres_curobo_coll_costs.squeeze(1) # from [n_rollouts, 1, n_spheres] to [n_rollouts, n_spheres]. the second dimension is squeezed out because it's 1 (representing only the  h'th step over the horizon).
                # NOTE: 2 "spheres_curobo_coll_costs" is a 3d tensor of shape:
                # [n_rollouts # number of rollouts, 1 # horizon length is 1 because we check collision for each time step separately, n_spheres # number of spheres in the robot (65 for franka)  
                # We don't need the horizon dimension, so we squeezed it out.
                if binary:
                    safety_margin_violation_rollouts = torch.any(spheres_curobo_coll_costs > 0, dim=1) # sets True for rollout if any of the robot spheres are too close to an obstacle (i.e, their "safety zone" is violated). Its a vector in length of n_rollouts, for each rollout, checks if for that rollout (at the h'th step) any of the robot spheres got too close to any of the obstacles. It does that by checking if there is any positive of "curobo collision cost" for that specific rollout (in the specific step h). 
                    safety_margin_violation_rollouts = safety_margin_violation_rollouts.float() # The .float() converts bool to float (True (safety margin violation) turns 1, False (no violation) turns 0).                
                else:
                    safety_margin_violation_rollouts = torch.sum(spheres_curobo_coll_costs, dim=1) # sum of collisioncosts over all spheres.
                
                dynamic_coll_cost_matrix[:, h] = safety_margin_violation_rollouts 


                # #### debug ####
                # if h % 7 == 0:
                #     print(f"step {h}: col_checker obs estimated pose: {self.H_world_cchecks[h].world_model.objects[0].pose}")
                # ############### 
        
        # Now that we have the full cost matrix, which is of shape [n_rollouts, horizon].
        # Some post-processing if needed:

        # For each rollout, if a cost entry is less than the previous entry, set it to 0. The idea is to avoid charging for actions which take the robot out of collision. For those actions, we set a cost of 0.
        if self.mask_decreasing_cost_entries:
            dynamic_coll_cost_matrix = mask_decreasing_values(dynamic_coll_cost_matrix)
        
        # Shift the cost matrix left by one (to charge for the action that leads to the collision, and not for the state you are in collision):
        if self.shift_cost_matrix_left:
            dynamic_coll_cost_matrix = shift_tensor_left(dynamic_coll_cost_matrix)
        
        # Scale the cost matrix by the cost weight:
        dynamic_coll_cost_matrix *= self.cost_weight
        
        return dynamic_coll_cost_matrix 
    
    def get_predicted_path(self, obstacle_name:str):
        """
        Get the predicted path of the obstacle over the next H time steps.
        """
        return self._obstacles_predicted_paths[obstacle_name]
        
         
       
