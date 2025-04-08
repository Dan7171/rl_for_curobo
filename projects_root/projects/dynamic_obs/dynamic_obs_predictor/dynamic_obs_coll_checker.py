from curobo.geom.sdf.world import CollisionQueryBuffer
from curobo.geom.sdf.world_mesh import WorldMeshCollision
from curobo.geom.types import Obstacle
from curobo.types.math import Pose
from typing import List
import numpy as np
import copy
import torch
from curobo.geom.sdf.world import WorldCollisionConfig

SPHERE_DIM = 4 # The values which are required to define a sphere: x,y,z,radius. That's fixed in all curobo coll checkers

class DynamicObsCollPredictor:
    """
    This class is used to check the collision of the dynamic obstacles in the world.
    It also contains the functionality to compute costs.
    Inspired by: https://curobo.org/get_started/2c_world_collision.html
    
    """
    

    def __init__(self, tensor_args, dynamic_obs_world_cfg, cache, step_dt_traj_mpc, H=30, n_rollouts=400, act_distance=1.0,robot_collision_sphere_num=65, cost_weight=5000):
        """_summary_

        Args:
            tensor_args (_type_): _description_
            world_cfg (_type_): _description_
            cache (_type_): _description_
            step_dt_traj_mpc (_type_): time passes between each step in the trajectory.
            H (int, optional): _description_. Defaults to 30.
            robot_collision_sphere_num = The number of collision spheres (the robot is approximated as spheres when calculating collision. NOTE: 65 is in franka,)
            
            """
        self._init_counter = 0 # number of steps until cuda graph initiation. Here Just for debugging  
        world_collision_config_base = WorldCollisionConfig(tensor_args, copy.deepcopy(dynamic_obs_world_cfg), copy.deepcopy(cache)) # base template for collision checkers. We'll use this to make H copies.
        self.tensor_args = tensor_args 
        self.H = H # Horizon length
        self.collision_sphere_num = robot_collision_sphere_num # The number of collision spheres (the robot is approximated as spheres when calculating collision. NOTE: 65 is in franka,)
        self.n_rollouts = n_rollouts # The number of rollouts
        query_buffer_shape = torch.zeros((self.n_rollouts, 1, self.collision_sphere_num, SPHERE_DIM), device=self.tensor_args.device, dtype=self.tensor_args.dtype).shape # torch.Size([self.n_rollouts, 1, self.collision_sphere_num, SPHERE_DIM]) # n_rollouts x 1(current time step) x num of coll spheres x sphere_dim (x,y,z,radius) https://curobo.org/get_started/2c_world_collision.html
        self.step_dt_traj_mpc = step_dt_traj_mpc # time passes between each step in the trajectory.
        self.act_distance = tensor_args.to_device([act_distance]) # activation distance for collision checking (below this distance, consider collision)
        self.weight_col_check = tensor_args.to_device([1]) # weight for curobo collision checking. Not sure why we need this but in all examples its 1. See https://curobo.org/get_started/2c_world_collision.html
        self.cost_weight = cost_weight # weight for the cost function (cost term weight)
        # Initialize the collision buffers and collision checkers
        self.H_collision_buffers = []
        self.H_world_cchecks = [] # list of collision checkers for each time step
        for _ in range(self.H):
            world_config = copy.deepcopy(world_collision_config_base) # NOTE: I use copies to avoid side effects.
            col_checker = WorldMeshCollision(world_config) # Initiate a single collision checker for each time step over the horizon.
            self.H_world_cchecks.append(col_checker)
            self.H_collision_buffers.append(CollisionQueryBuffer.initialize_from_shape(query_buffer_shape, self.tensor_args, col_checker.collision_types)) # I dont know yet what they are for
        
    # def update_obstacle_pose_in_cchecker(self, obstacle_name, obstacle_pose, obstacle_vel, obstacle_acceleration):
    #     obstacle_path = self.predict_obstacle_path(obstacle_pose, obstacle_vel, obstacle_acceleration)
    #     for h in range(self.H):
    #         self.H_world_cchecks[h].update_obstacle_pose(obstacle_name, obstacle_path[h])
    
    def predict_obstacle_path(self, cur_obstacle_pos:np.ndarray, cur_obstacle_lin_vel:np.ndarray, cur_obstacle_lin_acceleration:np.ndarray):
        """
        Predict the path of the obstacle over the next H steps.
        """
        path = np.ndarray(shape=(self.H, 3), dtype=np.float32)
        for h in range(self.H):
            x_vel = cur_obstacle_lin_vel * self.step_dt_traj_mpc * h
            x_acc = cur_obstacle_lin_acceleration * self.step_dt_traj_mpc * h
            path[h] = cur_obstacle_pos + x_vel + 0.5 * x_acc ** 2 # predicted pos at time step h
        
        return path
    
    def _update_ccheck(self, ccheker, new_positions, obs_names):
        """
        Update the collision checker with the predicted positions of the obstacles.
        """
        new_quaterion = np.array([0,0,0,1]) # TODO: This is temporary. We need to get the correct orientation of the obstacle.
        for obs_idx in range(len(obs_names)): # NOTE: could be parallelized
            pos_tensor = self.tensor_args.to_device(torch.from_numpy(new_positions[obs_idx]))
            rot_tensor = self.tensor_args.to_device(torch.from_numpy(new_quaterion))
            new_pose = Pose(pos_tensor, rot_tensor) 
            ccheker.update_obstacle_pose_in_world_model(pose=new_pose, name=obs_names[obs_idx])
    
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
        obs_pos_preds = np.ndarray(shape=(len(obstacles), self.H, 3), dtype=np.float32)
        for ob_idx, obstacle in enumerate(obstacles):
            
            # Get the current position velocity and acceleration of the obstacle
            cur_position = obstacle.simulation_representation.get_world_pose()[0]
            cur_lin_vel = obstacle.simulation_representation.get_linear_velocity()
            # cur_lin_acceleration = obstacle.simulation_representation.get_rigid_body_linear_acceleration()
            cur_lin_acceleration = np.zeros(3) # TODO: THIS IS TEMPORARY, NEED TO IMPLEMENT THIS CORRECTLY
            
            # Predict the path of the obstacle over the next H time steps. Result is a tensor of shape (H, 3)
            predicted_obstacle_path = self.predict_obstacle_path(cur_position, cur_lin_vel, cur_lin_acceleration) 
            
            # Store the predicted positions of the obstacle in the tensor
            obs_pos_preds[ob_idx] = predicted_obstacle_path

        # 3) Update each collision checker with the predicted positions of the obstacles in its time step.
        for h in range(self.H): # NOTE: We can parallelize this loop.
            ccheck = self.H_world_cchecks[h] # h'th collision checker
            predictions = obs_pos_preds[:, h, :] # predicted positions of all obstacles at time step h
            self._update_ccheck(ccheck, predictions, obs_names) # update the collision checker with the predicted positions of the obstacles

    def cost_fn(self, robot_spheres, env_query_idx=None, method='distance',collision_threshold=0.2):
        """
        Compute the collision cost for the robot spheres.
        Args:
            robot_spheres: tensor of shape [n_rollouts, horizon, n_spheres, 4]
            env_query_idx: optional index for querying specific environments
            method: collision checking method ('distance', 'collision', or 'swept_distance')
            NOTE: 
                1. cuRobo’s signed distance queries return a positive value when the sphere is inside an obstacle or within activation distance. If outside this range, the distance value will be zero. https://curobo.org/get_started/2c_world_collision.html#:~:text=cuRobo%E2%80%99s%20signed%20distance%20queries%20return%20a%20positive%20value%20when%20the%20sphere%20is%20inside%20an%20obstacle%20or%20within%20activation%20distance.%20If%20outside%20this%20range%2C%20the%20distance%20value%20will%20be%20zero.
        Returns:
            tensor of shape [n_rollouts, horizon] containing collision costs
        """
        
        if not torch.cuda.is_current_stream_capturing():
            self._init_counter += 1
            print(f"Initialization iteration {self._init_counter}")
            if torch.cuda.is_current_stream_capturing():
                print("During graph capture")

        # Initialize cost tensor
        ans = torch.zeros(self.n_rollouts, self.H, device=self.tensor_args.device)
        # Make input contiguous and ensure proper shape
        robot_spheres = robot_spheres.contiguous()
        for h in range(self.H):
            buffer_step_h = self.H_collision_buffers[h]
            
            # Extract and reshape spheres for current timestep
            robot_spheres_step_h = robot_spheres[:, h].contiguous().reshape(self.n_rollouts, 1, self.collision_sphere_num, SPHERE_DIM)
            
            if method == 'distance':
                    
                act_rad_max_pen_depth = self.H_world_cchecks[h].get_sphere_distance(
                    robot_spheres_step_h, 
                    buffer_step_h,
                    self.act_distance,
                    self.weight_col_check
                )
                # NOTE: "act_rad_max_pen_depth" is a 3d tensor of shape:
                # [n_rollouts # number of rollouts, 1 # horizon length is 1 because we check collision for each time step separately, n_spheres # number of spheres in the robot (65 for franka)  
                # We don't need the horizon dimension, so we squeeze it out.
                act_rad_max_pen_depth = act_rad_max_pen_depth.squeeze(1) # from [n_rollouts, 1, n_spheres] to [n_rollouts, n_spheres]
                
                
                # NOTE: 1) By definition: act_rad_max_pen_depth[i, j]:= the the maximum penetration depth of an obstacle (over all obstacles), inside of the jth sphere's activation radius of the ith rollout }.
                # So self.act_distance - act_rad_max_pen_depth[i, j] is the minimum distance between the jth sphere and the nearest obstacle!

                # NOTE: 2) By definition: act_rad_max_pen_depth[i, j] is either positive or 0. Never negative. 
                # - act_rad_max_pen_depth[i, j] is 0 <=> At rollout i, No object is violationg the jth sphere's activation radius. (all objects are outside the activation radius of the jth sphere (at this rollout at this step))
                # - act_rad_max_pen_depth[i, j] is positive <=> At rollout i, there is an object that is violating the jth sphere's activation radius.  
                # - In particular:
                #  - act_rad_max_pen_depth[i, j] is max the penetration depth of an object, into the jth sphere's *activation radius* of the ith rollout (max over all objects).
                #  - => Since the activation radius represents the distance at which the collision checking is active, the
                #  - => Additionaly, the exact penetration depth *into the jth sphere's itself (not the activation radius)* is act_rad_max_pen_depth[i, j] - self.act_distance. And of course that if its non negative, there is a collision.
                # For more info check https://curobo.org/get_started/2c_world_collision.html

                robot_spheres_max_pen_depth = act_rad_max_pen_depth - self.act_distance
                robot_spheres_min_dist_to_obstacles = - robot_spheres_max_pen_depth # for each sphere, the minimum distance to the nearest obstacle.
                # rollouts_with_col = torch.any(robot_spheres_max_pen_depth >= 0, dim=1) # vector in length of n_rollouts, for each rollout, checks if for that rollout at the current time step any collision spheres are in collision
                rollouts_with_col = torch.any(robot_spheres_min_dist_to_obstacles <= collision_threshold, dim=1) # vector in length of n_rollouts, for each rollout, checks if for that rollout at the current time step any collision spheres are in collision
                ans[:, h] = rollouts_with_col.float() # convert bool to float (collision is 1, no collision is 0.  now ans[i,j] is 1 <=> there is some collision for the ith rollout at the jth time step.
                
        ans *= self.cost_weight
        # ans = torch.rand_like(ans) * self.cost_weight # DEBUG
        return ans 
        
        # # Create ans_causing to detect transitions from no-collision to collision
        # ans_causing = torch.zeros_like(ans)
        # # For all timesteps except the first one (since we need to look at previous timestep)
        # ans_causing[:, :-1] = (ans[:, 1:] == 1) & (ans[:, :-1] == 0)
        # return ans_causing
            
            # elif method == 'collision':
            #     out = self.H_world_cchecks[h].get_sphere_collision(
            #         robot_spheres_step_h,
            #         buffer_step_h,
            #         self.act_distance,
            #         self.weight
            #     )
            # elif method == 'swept_distance':
            #     out = self.H_world_cchecks[h].get_swept_sphere_distance(
            #         robot_spheres_step_h,
            #         buffer_step_h,
            #         self.act_distance,
            #         self.weight
                # )
            
            # Sum the collision costs across spheres to get one cost per rollout
            # ans[:, h] = out.sum(dim=1).reshape(self.n_rollouts)

