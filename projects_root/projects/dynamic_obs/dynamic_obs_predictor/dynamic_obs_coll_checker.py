from curobo.geom.sdf.world import CollisionQueryBuffer
from curobo.geom.sdf.world_mesh import WorldMeshCollision
from curobo.geom.types import Obstacle
from curobo.types.math import Pose
from typing import List
import numpy as np
import copy
import torch
from curobo.geom.sdf.world import WorldCollisionConfig
from projects_root.utils.quaternion import integrate_quat
SPHERE_DIM = 4 # The values which are required to define a sphere: x,y,z,radius. That's fixed in all curobo coll checkers

class DynamicObsCollPredictor:
    """
    This class is used to check the collision of the dynamic obstacles in the world.
    It also contains the functionality to compute costs.
    Inspired by: https://curobo.org/get_started/2c_world_collision.html
    
    """
    

    def __init__(self, tensor_args, dynamic_obs_world_cfg, cache, step_dt_traj_mpc, H=30, n_rollouts=400, activation_distance= 0.05,robot_collision_sphere_num=65, cost_weight=100000):
        """_summary_

        Args:
            tensor_args (_type_): _description_
            world_cfg (_type_): _description_
            cache (_type_): _description_
            step_dt_traj_mpc (_type_): time passes between each step in the trajectory.
            H (int, optional): _description_. Defaults to 30.
            robot_collision_sphere_num = The number of collision spheres (the robot is approximated as spheres when calculating collision. NOTE: 65 is in franka,)
            activation_distance = The distance in meters between a robot sphere and an obstacle over all obstacles, at which the collision cost starts to be "active" (positive). Denoted as "etha" in the paper.
            """
        self._init_counter = 0 # number of steps until cuda graph initiation. Here Just for debugging  
        world_collision_config_base = WorldCollisionConfig(tensor_args, copy.deepcopy(dynamic_obs_world_cfg), copy.deepcopy(cache)) # base template for collision checkers. We'll use this to make H copies.
        self.tensor_args = tensor_args 
        self.H = H # Horizon length
        self.collision_sphere_num = robot_collision_sphere_num # The number of collision spheres (the robot is approximated as spheres when calculating collision. NOTE: 65 is in franka,)
        self.n_rollouts = n_rollouts # The number of rollouts
        query_buffer_shape = torch.zeros((self.n_rollouts, 1, self.collision_sphere_num, SPHERE_DIM), device=self.tensor_args.device, dtype=self.tensor_args.dtype).shape # torch.Size([self.n_rollouts, 1, self.collision_sphere_num, SPHERE_DIM]) # n_rollouts x 1(current time step) x num of coll spheres x sphere_dim (x,y,z,radius) https://curobo.org/get_started/2c_world_collision.html
        self.step_dt_traj_mpc = step_dt_traj_mpc # time passes between each step in the trajectory.
        self.activation_distance = tensor_args.to_device([activation_distance]) # activation distance for collision checking (below this distance, consider collision)
        self.weight_col_check = tensor_args.to_device([1]) # Leave 1. This is the weight for curobo collision checking. Explained in the paper. Since we are only interested at wether there is a collision or not (i.e. cost is positive in collision and 0 else), this is not important. See https://curobo.org/get_started/2c_world_collision.html
        self.cost_weight = cost_weight # weight for the dynamic obstacle cost function (cost term weight). This is a hyper-parameter, unlike the weight_col_check which you should leave as 1.
        
        # Initialize the collision buffers and collision checkers
        self.H_collision_buffers = []
        self.H_world_cchecks = [] # list of collision checkers for each time step
        for _ in range(self.H):
            world_config = copy.deepcopy(world_collision_config_base) # NOTE: I use copies to avoid side effects.
            col_checker = WorldMeshCollision(world_config) # Initiate a single collision checker for each time step over the horizon.
            self.H_world_cchecks.append(col_checker)
            self.H_collision_buffers.append(CollisionQueryBuffer.initialize_from_shape(query_buffer_shape, self.tensor_args, col_checker.collision_types)) # I dont know yet what they are for 
            # enable for collision checking all objects in the collision checker. Just a verification step beacuse I am not sure if all objects are automatically enabled when initiating the collision checker.
            # TODO: May be redundant. Remove if not needed.
            # for obj in col_checker.world_model.objects:
            #     col_checker.enable_obb(name=obj.name) 
    

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
    
    def update_predictive_collision_checkers(self, obstacles:List[Obstacle]):
        """
        First, For each object, predict its path (future positions) over the next H time steps.
        Then, for each time step, update the collision checker with the predicted positions of the objects.
        https://curobo.org/get_started/2c_world_collision.html#attach-object-note
        """

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

        # 3) Update each collision checker with the predicted positions of the obstacles in its time step.
        for h in range(self.H): # NOTE: We can parallelize this loop.
            step_h_cchecker = self.H_world_cchecks[h] # h'th collision checker
            step_h_pose_predictions_all_rollouts = obs_pose_preds[:, h, :] # predicted positions of all obstacles at time step h
            self._update_cchecker(step_h_cchecker, step_h_pose_predictions_all_rollouts, obs_names) # update the collision checker with the predicted positions of the obstacles

    def cost_fn(self, robot_spheres, env_query_idx=None, method='curobo_prim_coll_cost_fn'):
        """
        Compute the collision cost for the robot spheres. Called by the MPC cost function (in ArmBase).
        Args:
            robot_spheres: tensor of shape [n_rollouts, horizon, n_spheres, 4]
            env_query_idx: optional index for querying specific environments
            method: collision checking method ('distance', 'collision', or 'swept_distance')
            NOTE: 
                1. cuRobo's signed distance queries return a positive value when the sphere is inside an obstacle or within activation distance. If outside this range, the distance value will be zero. https://curobo.org/get_started/2c_world_collision.html#:~:text=cuRobo's%20signed%20distance%20queries%20return%20a%20positive%20value%20when%20the%20sphere%20is%20inside%20an%20obstacle%20or%20within%20activation%20distance.%20If%20outside%20this%20range%2C%20the%20distance%20value%20will%20be%20zero.
        Returns:
            tensor of shape [n_rollouts, horizon] containing collision costs
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
                    
                spheres_curobo_coll_costs = self.H_world_cchecks[h].get_sphere_distance(
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
                
                spheres_curobo_coll_costs = spheres_curobo_coll_costs.squeeze(1) # from [n_rollouts, 1, n_spheres] to [n_rollouts, n_spheres]. the second dimension is squeezed out because it's 1 (representing only the  h'th step over the horizon).
                # NOTE: 2 "spheres_curobo_coll_costs" is a 3d tensor of shape:
                # [n_rollouts # number of rollouts, 1 # horizon length is 1 because we check collision for each time step separately, n_spheres # number of spheres in the robot (65 for franka)  
                # We don't need the horizon dimension, so we squeezed it out.
                
                safety_margin_violation_rollouts = torch.any(spheres_curobo_coll_costs > 0, dim=1) # sets True for rollout if any of the robot spheres are too close to an obstacle (i.e, their "safety zone" is violated). Its a vector in length of n_rollouts, for each rollout, checks if for that rollout (at the h'th step) any of the robot spheres got too close to any of the obstacles. It does that by checking if there is any positive of "curobo collision cost" for that specific rollout (in the specific step h). 
                safety_margin_violation_rollouts = safety_margin_violation_rollouts.float() # The .float() converts bool to float (True (safety margin violation) turns 1, False (no violation) turns 0).                
                dynamic_coll_cost_matrix[:, h] = safety_margin_violation_rollouts 


                # #### debug ####
                # if h % 7 == 0:
                #     print(f"step {h}: col_checker obs estimated pose: {self.H_world_cchecks[h].world_model.objects[0].pose}")
                # ############### 
        
        mask_to_keep_first_violaion_only = False # If True, the cost matrix will be modified so that only the first violation of the safety margin is considered.
        if mask_to_keep_first_violaion_only:
            dynamic_coll_cost_matrix = torch.where(
                (dynamic_coll_cost_matrix == 1) & (torch.cumsum(dynamic_coll_cost_matrix, dim=1) == 1),
                torch.ones_like(dynamic_coll_cost_matrix),
                torch.zeros_like(dynamic_coll_cost_matrix)
            )
            
        dynamic_coll_cost_matrix *= self.cost_weight
        # cost_matrix = torch.rand_like(cost_matrix) * self.cost_weight # DEBUG
        return dynamic_coll_cost_matrix 
        
       
