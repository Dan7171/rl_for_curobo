from curobo.geom.sdf.world import CollisionQueryBuffer
from curobo.geom.sdf.world_mesh import WorldMeshCollision
from curobo.geom.types import Cuboid, Sphere, WorldConfig
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
import time
SPHERE_DIM = 4 # The values which are required to define a sphere: x,y,z,radius. That's fixed in all curobo coll checkers

def _parallel_update_cchecker_single_step(h, all_H_ccheckers, full_H_obs_pose_preds, obs_names):
    step_h_cchecker = all_H_ccheckers[h]
    step_h_pose_predictions_all_rollouts = full_H_obs_pose_preds[:, h, :]
    step_h_cchecker._update_cchecker(step_h_pose_predictions_all_rollouts, obs_names)

def compute_signed_distance(p_from, p_to, rad_from, rad_to):
    """
    Compute the signed distance between two spheres.
    """
    return torch.norm(p_to - p_from, dim=3) - (rad_from + rad_to)

class DynamicObsCollPredictor:
    """
    
    This class is used to predict and compute costs for the collision of the dynamic obstacles in the world.
    You can think of this class as a collection of H collision checkers, one for each time step in the horizon + functionality to compute costs for the MPC (see cost_fn function).
    Each individual collision checker which it contains, is inspired by: https://curobo.org/get_started/2c_world_collision.html
    """
    

    # def __init__(self, tensor_args, world_cfg_template:WorldConfig , cache, step_dt_traj_mpc, H=30, n_rollouts=400, activation_distance= 0.05,n_spheres_own=65, cost_weight=100000, shift_cost_matrix_left=True, mask_decreasing_cost_entries=True):
    def __init__(self, tensor_args, step_dt_traj_mpc, H=30, n_rollouts=400, activation_distance= 0.05, n_spheres_own=65, cost_weight=100000, shift_cost_matrix_left=True, mask_decreasing_cost_entries=True):
        """ Initialize H dynamic obstacle collision checker, for each time step in the horizon, 
        as well as setting the cost function parameters for the dynamic obstacle cost function.


        Args:
            tensor_args: pytorch tensor arguments.
            cache (dict): collision checker cache for the pre-defined dynamic primitives.
            step_dt_traj_mpc (float): Time passes between each step in the trajectory. This is what the mpc assumes time delta between steps in horizon is.
            H (int, optional): Defaults to 30. The horizon length. TODO: Should be taken from the mpc config.
            n_checkers(int, optional): Defaults to H (marked by passing -1). The number of collision checkers to use. If n_checkers is not H, then the collision checkers will be used in a sliding window fashion.
            n_rollouts (int, optional): Defaults to 400. The number of rollouts. TODO: Should be taken from the mpc config.
            n_spheres_own: The number of collision spheres (the robot is approximated as spheres when calculating collision. NOTE: 65 is in franka,)
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
        self.n_spheres_own = n_spheres_own 
        self.n_rollouts = n_rollouts 
        self.step_dt_traj_mpc = step_dt_traj_mpc 
        self.activation_distance = tensor_args.to_device([activation_distance]) 
        self.cost_weight = cost_weight
        self.shift_cost_matrix_left = shift_cost_matrix_left
        self.mask_decreasing_cost_entries = mask_decreasing_cost_entries
        self._obstacles_predicted_paths = {} # dictionary of predicted paths for each obstacle
        # self.n_obs = n_sphere_obs
        
        self.rad_obs = torch.zeros(0, device=self.tensor_args.device) 
        self.p_obs = torch.zeros((self.H, 0, 3), device=self.tensor_args.device) # pos and radius of the obstacles (spheres) over horizon
        

    def add_obs(self, p_obs:torch.Tensor, rad_obs:torch.Tensor):
        """
        Initialize the obstacles.
        """
        self.p_obs = torch.cat([self.p_obs, p_obs], dim=1)
        self.rad_obs = torch.cat([self.rad_obs, rad_obs], dim=0)
        
    
    def update_p_obs(self, p_obs:torch.Tensor):
        """
        Update the poses of the obstacles.
        Args:
            p_obs: tensor of shape [H, n_obs, 3]. The poses of the obstacles.
        """
        self.p_obs = p_obs
    
    def cost_fn(self, prad_own:torch.Tensor, env_query_idx=None,  binary=False, safety_margin=0.1):
        """
        Compute the collision cost for the robot spheres. Called by the MPC cost function (in ArmBase).
        Args:
            prad_own: tensor of shape [n_rollouts, horizon, n_spheres, 4]. Collision spheres of the robot. This is the standard structure of the input to the cost function.
            
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
        
        
        dynamic_coll_cost_matrix = torch.zeros(self.n_rollouts, self.H, device=self.tensor_args.device)
        # Initialize cost tensor

        # Make input contiguous and ensure proper shape
        prad_own = prad_own.contiguous() # Returns a contiguous in memory tensor containing the same data as self tensor. If self tensor is already in the specified memory format, this function returns the self tensor. https://pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html
        p_own, rad_own = prad_own[:,:,:,:3], prad_own[0,0,:,3]
        
        # filter out invalid own collision spheres 
        is_rad_own_valid = rad_own > 0
        rad_own_valid = rad_own[is_rad_own_valid]
        p_own_valid = p_own[:,:,is_rad_own_valid,:] 
        
        if len(self.rad_obs) > 0: # If there are any obstacles
            # filter out invalid obstacles 
            p_obs = self.p_obs
            is_rad_obs_valid = self.rad_obs > 0
            rad_obs_valid = self.rad_obs[is_rad_obs_valid]
            p_obs_valid = p_obs[:,is_rad_obs_valid,:]
            # compute distance between surfaces of valid own spheres and valid obstacles (positive means no overlap, negative means overlap)
            sd_ownobs = torch.zeros((self.n_rollouts, self.H, len(rad_own_valid), len(rad_obs_valid)), device=self.tensor_args.device) # n_rollouts x H x n_spheres_own x n_obs 
            # sd_ownobs = torch.norm(p_own_valid.reshape([1,*p_own_valid.shape]) - p_obs_valid , dim=3) - (rad_own_valid + rad_obs_valid.T) # each entry is the distance from robot sphere surface to obstacle sphere surface.
            sd_ownobs = torch.cdist(p_own_valid, p_obs_valid) - rad_own_valid.reshape(*rad_own_valid.shape, 1) + rad_obs_valid.reshape(1,*rad_obs_valid.shape)
            # sd_ownobs = torch.clamp(sd_ownobs, max=0.1) # Set positive values to 0 .
            col = (sd_ownobs < safety_margin).float() # sets True for rollout if any of the robot spheres are too close to any of the obstacles (i.e, their "safety zone" is violated). Its a vector in length of n_rollouts, for each rollout, checks if for that rollout any of the robot spheres got too close to any of the obstacles. 
            col = col.sum(dim=3).sum(dim=2) # for each (rollout,step) sum all collisions (over all pairs of own spheres and obstacles). Get a matrix of shape [n_rollouts, H] where each entry is the number of violations of the safety margin for the robot spheres.
            dynamic_coll_cost_matrix = col
        # dynamic_coll_cost_matrix = torch.sum(col, dim=3) # sum of collisioncosts over all spheres.
        # sd_ownobs = compute_signed_distance(p_own, ,c)

        # dynamic_coll_cost_matrix = torch.zeros(self.n_rollouts, self.H, device=self.tensor_args.device)

        # prad_ownh = prad_own.contiguous() # From [n_rollouts, H, n_spheres, SPHERE_DIM] to [n_rollouts, n_spheres, SPHERE_DIM]. We now focus on rollouts only from the time step "t+h" over the horizon (horizon starts at t, meaning h=0).
        # prad_ownh = prad_ownh.reshape(self.n_rollouts, 1, self.n_spheres_own, SPHERE_DIM) # From [n_rollouts, n_spheres, SPHERE_DIM] to [n_rollouts, 1, n_spheres, SPHERE_DIM] 
        # prad_ownh = prad_ownh.squeeze(1) # From [n_rollouts, 1, n_spheres, SPHERE_DIM] to [n_rollouts, n_spheres, SPHERE_DIM]. the second dimension is squeezed out because it's 1 (representing only the  h'th step over the horizon).
        # p_ownh, rad_ownh = prad_ownh[:,:,:3], prad_ownh[:,:,3]
        # d_collh = torch.zeros((self.n_rollouts, self.n_spheres_own), device=self.tensor_args.device)


        # for h in range(self.H):
    
        #     # checker_to_use_at_step_h = max(self.cur_checker_idx + h, self.n_valid_checkers - 1) # self.routing_table[h]
        #     # buffer_step_h = self.collision_buffers[checker_to_use_at_step_h]

        #     # Extract and reshape spheres for current timestep
        #     prad_ownh = prad_own[:, h].contiguous() # From [n_rollouts, H, n_spheres, SPHERE_DIM] to [n_rollouts, n_spheres, SPHERE_DIM]. We now focus on rollouts only from the time step "t+h" over the horizon (horizon starts at t, meaning h=0).
        #     prad_ownh = prad_ownh.reshape(self.n_rollouts, 1, self.n_spheres_own, SPHERE_DIM) # From [n_rollouts, n_spheres, SPHERE_DIM] to [n_rollouts, 1, n_spheres, SPHERE_DIM] 
        #     prad_ownh = prad_ownh.squeeze(1) # From [n_rollouts, 1, n_spheres, SPHERE_DIM] to [n_rollouts, n_spheres, SPHERE_DIM]. the second dimension is squeezed out because it's 1 (representing only the  h'th step over the horizon).
        #     p_ownh, rad_ownh = prad_ownh[:,:,:3], prad_ownh[:,:,3]
        #     d_collh = torch.zeros((self.n_rollouts, self.n_spheres_own), device=self.tensor_args.device)

        #     # cost_matrix_step_h = self.cchecks[checker_to_use_at_step_h].get_sphere_distance( # NOTE: although the method is called get_sphere_distance, it actually returns the "curobo collision cost" (see section 3.3 in paper).
        #     # cost_matrix_step_h = cost_matrix_step_h.squeeze(1) # from [n_rollouts, 1, n_spheres] to [n_rollouts, n_spheres]. the second dimension is squeezed out because it's 1 (representing only the  h'th step over the horizon).
        #     # NOTE: 2 "spheres_curobo_coll_costs" is a 3d tensor of shape:
        #     # [n_rollouts # number of rollouts, 1 # horizon length is 1 because we check collision for each time step separately, n_spheres # number of spheres in the robot (65 for franka)  
        #     # We don't need the horizon dimension, so we squeezed it out.
        
            
        #     if binary:
        #         safety_margin_violation_rollouts = torch.any(cost_matrix_step_h > 0, dim=1) # sets True for rollout if any of the robot spheres are too close to an obstacle (i.e, their "safety zone" is violated). Its a vector in length of n_rollouts, for each rollout, checks if for that rollout (at the h'th step) any of the robot spheres got too close to any of the obstacles. It does that by checking if there is any positive of "curobo collision cost" for that specific rollout (in the specific step h). 
        #         safety_margin_violation_rollouts = safety_margin_violation_rollouts.float() # The .float() converts bool to float (True (safety margin violation) turns 1, False (no violation) turns 0).                
        #     else:
        #         safety_margin_violation_rollouts = torch.sum(spheres_curobo_coll_costs, dim=1) # sum of collisioncosts over all spheres.
            

            
        # dynamic_coll_cost_matrix[:, h] = safety_margin_violation_rollouts 


                # #### debug ####
                # if h % 7 == 0:
                #     print(f"step {h}: col_checker obs estimated pose: {self.cchecks[h].world_model.objects[0].pose}")
                # ############### 
        
            # Now that we have the full cost matrix, which is of shape [n_rollouts, horizon].
            # Some post-processing if needed:

        # For each rollout, if a cost entry is less than the previous entry, set it to 0. The idea is to avoid charging for actions which take the robot out of collision. For those actions, we set a cost of 0.
        # if self.mask_decreasing_cost_entries:
        #     dynamic_coll_cost_matrix = mask_decreasing_values(dynamic_coll_cost_matrix)
        
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
        
    # def update_cuboids_in_checker(self, cuboid_list:List[Cuboid], checker_idx:int):
    #     """
    #     Update the cuboids in the collision checker at a specific index.
    #     """
    #     for cuboid in cuboid_list:
    #         self.cchecks[checker_idx].add_obb(cuboid)
    #         if cuboid.name not in self._obstacles_predicted_paths:
    #             self._obstacles_predicted_paths[cuboid.name] = np.zeros((self.n_checkers, 7))
    #         self._obstacles_predicted_paths[cuboid.name][checker_idx] = np.array(cuboid.pose)

    # def add_cuboids_to_checker(self, cuboid_list:List[Cuboid], checker_idx:int):
    #     """
    #     Add an obstacle to the collision checker at a specific index.
    #     """
    #     for cuboid in cuboid_list:
    #         self.cchecks[checker_idx].add_obb(cuboid)
    #         if cuboid.name not in self._obstacles_predicted_paths:
    #             self._obstacles_predicted_paths[cuboid.name] = np.zeros((self.n_checkers, 7))
    #         self._obstacles_predicted_paths[cuboid.name][checker_idx] = np.array(cuboid.pose)

    # def disable_obstacles_in_checker(self, obstacle_names:List[str], checker_idx:int):
    #     """
    #     Disable the obstacles in the collision checker at a specific index.
    #     """
    #     for obstacle_name in obstacle_names:
    #         self.cchecks[checker_idx].enable_obstacle(obstacle_name, False)

    # def enable_obstacles_in_checker(self, obstacle_names:List[str], checker_idx:int):
    #     """
    #     Enable the obstacles in the collision checker at a specific index.
    #     """
    #     for obstacle_name in obstacle_names:
    #         self.cchecks[checker_idx].enable_obstacle(obstacle_name)



    

    




