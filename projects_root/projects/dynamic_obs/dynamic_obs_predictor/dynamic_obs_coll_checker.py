import threading
from curobo.geom.sdf.world import CollisionQueryBuffer
from curobo.geom.sdf.world_mesh import WorldMeshCollision
from curobo.geom.types import Cuboid, Sphere, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from typing import List
import numpy as np
import copy
import torch
from curobo.geom.sdf.world import WorldCollisionConfig
from projects_root.projects.dynamic_obs.dynamic_obs_predictor.utils import shift_tensor_left, mask_decreasing_values
from concurrent.futures import ProcessPoolExecutor, wait 
import time

def get_size_bites(tensor:torch.Tensor):
    return tensor.element_size() * tensor.nelement()

def get_size_kb(tensor:torch.Tensor):
    return get_size_bites(tensor) / 1024

def get_size_mb(tensor:torch.Tensor):
    return get_size_kb(tensor) / 1024

def get_size_gb(tensor:torch.Tensor):
    return get_size_mb(tensor) / 1024

class DynamicObsCollPredictor:
    """
    
    This class is used to predict and compute costs for the collision of the dynamic obstacles in the world.
    You can think of this class as a collection of H collision checkers, one for each time step in the horizon + functionality to compute costs for the MPC (see cost_fn function).
    Each individual collision checker which it contains, is inspired by: https://curobo.org/get_started/2c_world_collision.html
    """
    

    def __init__(self, tensor_args, step_dt_traj_opt=None, H=30, n_rollouts=400, n_own_spheres=61, n_obs=61, cost_weight=100.0, obs_groups_nspheres=[], manually_express_p_own_in_world_frame=False, p_R=torch.zeros(3)):
        """ Initialize H dynamic obstacle collision checker, for each time step in the horizon, 
        as well as setting the cost function parameters for the dynamic obstacle cost function.


        Args:
            tensor_args: pytorch tensor arguments.
            cache (dict): collision checker cache for the pre-defined dynamic primitives.
            step_dt_traj_opt (float): Time passes between each step in the trajectory. This is what the mpc/curobo solver assumes time delta between steps in horizon is when parforming path planning.
            H (int, optional): Defaults to 30. The horizon length- number of states in the trajectory during trajectory optimization.
            n_checkers(int, optional): Defaults to H (marked by passing -1). The number of collision checkers to use. If n_checkers is not H, then the collision checkers will be used in a sliding window fashion.
            n_rollouts (int, optional): Defaults to 400. The number of rollouts. TODO: Should be taken from the mpc config.
            cost_weight: weight for the dynamic obstacle cost function (cost term weight). This is a hyper-parameter, unlike the weight_col_check which you should leave as 1. Default value is 100000, as by the original primitive collision cost weight of the mpc.
            obs_groups_nspheres: list of ints, each int is the number of obstacles in the group. # This is useful in cases where the obstacles are grouped together in the world (example: a group could be another robot, or an obstacle made out of multiple spheres).
            manually_express_p_own_in_world_frame: if True, the robot spheres positions are expressed in the world frame, otherwise they are expressed in the robot base frame.
            """
        self._init_counter = 0 # number of steps until cuda graph initiation. Here Just for debugging. Can be removed. with no effect on the code. 
        self.tensor_args = tensor_args 
        self.H = H # number of states in the trajectory during trajectory optimization. https://curobo.org/_api/curobo.wrap.reacher.trajopt.html#curobo.wrap.reacher.trajopt.TrajOptSolver.action_horizon
        self.n_rollouts = n_rollouts 
        self.step_dt_traj_opt = step_dt_traj_opt 
        self.cost_weight = cost_weight
        self.n_own_spheres = n_own_spheres # number of valid spheres of the robot (ignoring 4 spheres which are not valid due to negative radius)
        self.n_obs = n_obs # number of valid obstacles (ignoring 4 spheres which are not valid due to negative radius)
        self.obs_groups_nspheres = obs_groups_nspheres # list of ints, each int is the number of obstacles in the group. # This is useful in cases where the obstacles are grouped together in the world (example: a group could be another robot, or an obstacle made out of multiple spheres).
        if len(obs_groups_nspheres) > 0:
            assert sum(obs_groups_nspheres) == self.n_obs, "Error: The sum of the number of obstacles in the groups must be equal to the total number of obstacles"
            self.obs_groups_start_idx_list = ([0] + list(np.cumsum(obs_groups_nspheres)))[:-1] # list of ints, each int is the start index of the group in the world.
        
        # Buffers for obstacles (spheres): position and radius
        self.rad_obs_buf = torch.zeros(self.n_obs, device=self.tensor_args.device) # [n_obs] obstacles radii buffer
        self.rad_obs_buf_unsqueezed = self.rad_obs_buf.reshape(1, *self.rad_obs_buf.shape) # [1 x n_obs]  Added 1 dimension for intermediate calculations
        self.p_obs_buf = torch.zeros((self.H, self.n_obs, 3), device=self.tensor_args.device) # [H x n_obs x 3] pos and radius of the obstacles (spheres) over horizon
        self.p_obs_buf_unsqueezed = self.p_obs_buf.reshape(1,H,1,self.n_obs,3) # [1 x H x 1 x n_obs x 3] Added 1 dimension for intermediate calculations      
        
        # Buffers for own spheres: position and radius
        self.p_own_buf = torch.empty(n_rollouts, H, self.n_own_spheres, 3, device=self.tensor_args.device) # [n_rollouts x H x n_own x 3] Own spheres positions buffer
        
        self.p_own_buf_unsqueezed = self.p_own_buf.reshape(self.n_rollouts,self.H,self.n_own_spheres,1,3) # added 1 dimension for intermediate calculations [n_rollouts x H x n_own x 1 x 3]
        self.rad_own_buf = torch.zeros(self.n_own_spheres, device=self.tensor_args.device) # [n_own] Own spheres radii buffer
        self.rad_own_buf_unsqueezed = self.rad_own_buf.reshape(*self.rad_own_buf.shape, 1) # [n_own x 1] added 1 dimension for intermediate calculations

        # Buffers for intermediate calculations
        self.pairwise_ownobs_radsum_buf = self.rad_own_buf_unsqueezed + self.rad_obs_buf_unsqueezed # [n_own x n_obs] matrix for own radius i + obstacle radius j for all possible own and obstacle sphere pairs (Note: this is broadcasted because its a sum of n_own x 1 + 1 x n_obs radii)
        self.pairwise_ownobs_radsum_buf_unsqueezed = self.pairwise_ownobs_radsum_buf.reshape(1,1, *self.pairwise_ownobs_radsum_buf.shape, 1) # [1 x 1 x n_own x n_obs x 1] Added 1 dimension for intermediate calculations
    
        self.ownobs_diff_vector_buff = self.p_own_buf_unsqueezed - self.p_obs_buf_unsqueezed # [n_rollouts x H x n_own x n_obs x 3] A tensor for all pairs of the difference vectors between own and obstacle spheres. Each entry i,j will be storing the vector (p_own_buf[...][i] - p_obs_buf[...][j] where ... is the rollout, horizon and sphere indices)
        self.pairwise_ownobs_surface_dist_buf = torch.zeros(self.ownobs_diff_vector_buff.shape[:-1] + (1,), device=self.tensor_args.device) # [n_rollouts x H x n_own x n_obs x 1] A tensor for all pairs of the (non negative) distance between own and obstacle spheres. Each entry i,j will be storing the signed distance (p_own_buf[...][i] - p_obs_buf[...][j]) - (rad_own_buf[i] + rad_obs_buf[j]) where ... is the rollout, horizon and sphere indices)
        self.cost_mat_buf = torch.zeros(n_rollouts, H, device=self.tensor_args.device) # [n_rollouts x H] A tensor for the collision cost for each rollout and time step in the horizon. This is the output of the cost function.

        # flags
        self.init_rad_buffs = torch.tensor([0], device=self.tensor_args.device) # [1] If 1, the rad_obs_buffs are initialized (obstacles which should be set only once).
        self.manually_express_p_own_in_world_frame = manually_express_p_own_in_world_frame # if True, the robot spheres positions are expressed in the world frame, otherwise they are expressed in the robot base frame.
        if self.manually_express_p_own_in_world_frame:
            self.p_R = p_R.to(self.tensor_args.device) # xyz of own base in world frame
            self.p_R_broadcasted_buf = torch.zeros(self.p_own_buf.shape, device=self.tensor_args.device) + self.p_R # [n_rollouts x H x n_own x 3] A tensor for the robot spheres positions in the world frame.
    
    def get_obs_group_idx_range(self, obs_group_idx:int):
        """
        Get the start (inclusive) and end (exclusive) indices of the obstacles in the group.
        For example, if the group is the first group, then the start index is 2 and its length is 2, returns (2,4).
        """
        return self.obs_groups_start_idx_list[obs_group_idx], (self.obs_groups_start_idx_list[obs_group_idx+1] if (obs_group_idx + 1) < len(self.obs_groups_start_idx_list) else self.n_obs) # self.obs_groups_nspheres[obs_group_idx]
    
    def get_n_obs(self):
        return self.n_obs
    
        
    def set_obs_rads(self, rad_obs:torch.Tensor):
        """
        rad_obs: tensor of shape [n_obs]. The radii of the obstacles.
        """
        
        # assert self.rad_obs_buf.sum() > 0, "Error: Must set the obstacles (radii) before activating the collision checker"
        assert rad_obs.ndim == 1, "Error: The obstacle radii must be a 1D tensor"
        assert rad_obs.shape[0] == self.n_obs, "Error: The number of obstacles must be equal to the number of obstacle radii"

        self.rad_obs_buf.copy_(rad_obs)
        self.rad_obs_buf_unsqueezed.copy_(self.rad_obs_buf.view_as(self.rad_obs_buf_unsqueezed)) 

    def set_own_rads(self, rad_own:torch.Tensor):
        """
        Set the radii of the own spheres.
        """
        self.rad_own_buf.copy_(rad_own) # [n_own] init own spheres radii
        self.rad_own_buf_unsqueezed.copy_(self.rad_own_buf.view_as(self.rad_own_buf_unsqueezed)) 
        torch.add(self.rad_own_buf_unsqueezed, self.rad_obs_buf_unsqueezed, out=self.pairwise_ownobs_radsum_buf) # broadcasted addition of rad_own and rad_obs
        self.pairwise_ownobs_radsum_buf_unsqueezed[0,0,:,:,0].copy_(self.pairwise_ownobs_radsum_buf) 
        self.init_rad_buffs[0] = 1 # so that the following code will not re-initialize the buffers again (its just for efficiency).
    
    def update(self, p_obs:torch.Tensor):
        """
        Update the poses of the obstacles.
        Args:
            p_obs: tensor of shape [H, n_obs, 3]. The poses of the obstacles.
        """
        # self.p_obs = p_obs
        self.p_obs_buf.copy_(p_obs) # copy p_obs to self.p_obs in place.
    
    
    def update_obs_groups(self, p_obs:list[torch.Tensor],obs_groups_to_update:list[int]):
        """
        Update the poses of the obstacles in the groups.
        """
        for obs_group_id in obs_groups_to_update:
            range_start, range_end = self.get_obs_group_idx_range(obs_group_id)
            self.p_obs_buf[:, range_start:range_end, :].copy_(p_obs[obs_group_id][:, range_start:range_end,:]) # copy p_obs to self.p_obs in place.


    def cost_fn(self, prad_own:torch.Tensor, safety_margin=0.1):
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

       
        
        # if not bool(self.init_rad_buffs[0]):
        #     self.rad_own_buf.copy_(prad_own[0,0,:self.n_own_spheres,3]) # [n_own] init own spheres radii
        #     self.rad_own_buf_unsqueezed.copy_(self.rad_own_buf.view_as(self.rad_own_buf_unsqueezed)) 
        #     # self.rad_obs_buf_unsqueezed.copy_(self.rad_obs_buf.view_as(self.rad_obs_buf_unsqueezed)) 
        #     torch.add(self.rad_own_buf_unsqueezed, self.rad_obs_buf_unsqueezed, out=self.pairwise_ownobs_radsum_buf) # broadcasted addition of rad_own and rad_obs
        #     self.pairwise_ownobs_radsum_buf_unsqueezed[0,0,:,:,0].copy_(self.pairwise_ownobs_radsum_buf) 
        #     self.init_rad_buffs[0] = 1 # so that the following code will not re-initialize the buffers again (its just for efficiency).
        
        # Every time
        
        self.p_own_buf.copy_(prad_own[:,:,:self.n_own_spheres,:3]) # read robot spheres positions to the "prad_own" buffer.
        if self.manually_express_p_own_in_world_frame: # shift the robot spheres positions to the world frame, if not already in world frame (if manually_express_p_own_in_world_frame is True).
            torch.add(self.p_own_buf, self.p_R_broadcasted_buf, out=self.p_own_buf)
        self.p_own_buf_unsqueezed.copy_(self.p_own_buf.reshape(self.n_rollouts,self.H,self.n_own_spheres,1,3)) # Copy the reshaped version as well.
        self.p_obs_buf_unsqueezed.copy_(self.p_obs_buf.reshape(1,self.H,1,self.n_obs,3)) # Copy the reshaped version 
        torch.sub(self.p_own_buf_unsqueezed, self.p_obs_buf_unsqueezed, out=self.ownobs_diff_vector_buff) # Compute the difference vector between own and obstacle spheres and put it in the buffer. [n_rollouts x H x n_own x n_obs x 3]
        torch.norm(self.ownobs_diff_vector_buff, dim=-1, keepdim=True, out=self.pairwise_ownobs_surface_dist_buf) # Compute the distance between each own and obstacle sphere centers and put it in the buffer. n_rollouts x H x n_own x n_obs x 1
        self.pairwise_ownobs_surface_dist_buf.sub_(self.pairwise_ownobs_radsum_buf_unsqueezed) # subtract the sum of the radii from the distance (so we now get the distance between the spheres surfaces and not the centers). [n_rollouts x H x n_own x n_obs x 1]
        self.pairwise_ownobs_surface_dist_buf.lt_(safety_margin) # [n_rollouts x H x n_own x n_obs x 1] 1 where the distance between surfaces is less than the safety margin (meaning: very close to collision) and 0 otherwise.
        torch.sum(self.pairwise_ownobs_surface_dist_buf, dim=[2,3,4], out=self.cost_mat_buf) # [n_rollouts x H] (Counting the number of times the safety margin is violated, for each step in each rollout (sum over all spheres of the robot and all obstacles).)
        
        # if (self.cost_mat_buf.sum() / (self.n_rollouts * self.n_obs * self.n_own_spheres)) > 0:
        #     print(f'Debug: violations per rollout and col sphere obstacle (on average) =  {(self.cost_mat_buf.sum() / (self.n_rollouts * self.n_obs * self.H))} ')

        self.cost_mat_buf.mul_(self.cost_weight) # muliply the cost by the cost weight.
        
        return self.cost_mat_buf # dynamic_coll_cost_matrix 

class DynamicObsCollPredictorBatch:

    DTYPE = torch.float16 

    def __init__(self,
                tensor_args:TensorDeviceType,
                n_envs:int, 
                # n_robots_envwise:list[int],
                H:int, 
                b:int, # n rollouts per robot  
                n_robot_spheres:int,
                cost_weight:float=100.0, 
                # p_robotBaseEnvwise_F:list[list[torch.Tensor]]=None, # robot base position (xyz) in frame F. TODO: check if F is world or env frame.
                n_robots_envwise:list[int]=[1],
                safety_margin:float=0.1
                ):
        
        self.tensor_args = tensor_args
        self.n_envs = n_envs
        self.H = H
        self.b = b
        self.cost_weight = cost_weight
        # self.p_robotBaseEnvwise_F = p_robotBaseEnvwise_F
        self.n_robot_spheres = n_robot_spheres
        self.safety_margin = safety_margin
        self.n_robots_envwise = n_robots_envwise
        # self.n_robots_envwise = [len(robots_bases) for robots_bases in self.p_robotBaseEnvwise_F] # num of robots in each env
        self.n_robots_total = sum(self.n_robots_envwise) # total number of robots over all environments
        self.n_envs = len(self.n_robots_envwise) # len(self.p_robotBaseEnvwise_F)
        self.B = self.b * self.n_robots_total # total number of rollouts
        
        self._p_collider_buf = torch.empty(self.b, self.H, self.n_robot_spheres,1, 3, device=self.tensor_args.device, dtype=self.DTYPE) # [n_rollouts x H x n_own x 3] Own spheres positions buffer
        self._p_collide_with_buf = torch.empty(self.b, self.H, 1, self.n_robot_spheres, 3, device=self.tensor_args.device, dtype=self.DTYPE) # [n_rollouts x H x n_own x 3] Own spheres positions buffer
        self._rad_sum_buf = torch.empty(1, 1, self.n_robot_spheres, device=self.tensor_args.device, dtype=self.DTYPE) # [n_rollouts x H x n_own x n_obs] Own spheres positions buffer
        
        # _idx_map_tree_to_flat[env][collider] = start idx of the collider in the flat buffer (at the input/output row num B).
        self._idx_map_tree_to_flat = []
        tmp_cntr = 0
        for env in range(self.n_envs):
            self._idx_map_tree_to_flat.append([])
            for collider in range(self.n_robots_envwise[env]):
                self._idx_map_tree_to_flat[env].append(tmp_cntr)
                tmp_cntr += self.b

        # self._dim0_map = []
        # for env in range(self.n_envs):
        #     for collider in range(self.n_robots_envwise[env]):
        #         for collide_with in range(self.n_robots_envwise[env]):
        #             if collider == collide_with:
        #                 continue
        #             self._dim0_map.append((env, collider, collide_with))

        # ∑_i=0:n_envs-1 [n_robotsᵢ * (n_robotsᵢ - 1)]. 
        # All possible ordered pairs of robots in all environments.
        # self._n_same_env_pairs_ordered = len(self._dim0_map)  
        
        # self._tmp_buffer =  torch.empty(
        #         len(self._dim0_map),
        #         self.b, 
        #         self.H, 
        #         self.n_robot_spheres,
        #         self.n_robot_spheres,
        #         device=self.tensor_args.device,
        #         dtype=self.DTYPE
        #     )
        self._sphere_centers_diff_buf =  torch.empty(
                self.b, 
                self.H, 
                self.n_robot_spheres,
                self.n_robot_spheres,
                3,
                device=self.tensor_args.device,
                dtype=self.DTYPE
            )
        self._out_buf = torch.empty(
            self.B,
            self.H,
            device=self.tensor_args.device,
            dtype=self.DTYPE
        )


    def cost_fn(self, prad:torch.Tensor, env_query_idx:torch.Tensor, parallel_envs = False):
        """
        prad: 
            torch.Tensor[B, H, n_own_spheres (normally 65), 4],
                where B = b * num of all robots over all environments
                H = horizon length
                n_own_spheres = number of own spheres
                n_obs_max_per_robot = max number of obstacles per robot
            env_query_idx:
                torch.Tensor[B, 1]
                the environment index of each rollout in prad_batch.
        """
        
        # for env in range(self.n_envs):
        #     n_envi = self.n_robots_envwise[env]
        #     for collider in range(n_envi):
        #         for collide_with in range(n_envi):
        #             if collider == collide_with:
        #                 continue
        #             self._tmp_buffer[env,,r_obss = prad[env_query_idx[:,0] == e,:,:,:,:]
        if not parallel_envs:
            for env in range(self.n_envs):
                env_prad = prad[env_query_idx[:,0] == env,:,:,:]
                for collider in range(self.n_robots_envwise[env]):
                    collider_range_start, collider_range_end = self.b*collider, self.b*(collider+1)
                    p_collider = env_prad[collider_range_start:collider_range_end,:,:,:3]
                    rad_collider = env_prad[collider_range_start:collider_range_end,:,:,3]
                    for collide_with in range(self.n_robots_envwise[env]):
                        if collider == collide_with:
                            continue
                        collide_with_range_start, collide_with_range_end = self.b*collide_with, self.b*(collide_with+1)
                        p_collide_with = env_prad[collide_with_range_start:collide_with_range_end,:,:,:3]
                        rad_collide_with = env_prad[collide_with_range_start:collide_with_range_end,:,:,3]
                        self._p_collider_buf.copy_(p_collider.reshape(self.b,self.H,self.n_robot_spheres,1,3)) # Copy the reshaped version as well.
                        self._p_collide_with_buf.copy_(p_collide_with.reshape(self.b,self.H, 1,self.n_robot_spheres,3)) # Copy the reshaped version as well.
                        torch.sub(self._p_collider_buf,self._p_collide_with_buf, out=self._sphere_centers_diff_buf) # Compute the difference vector between own and obstacle spheres and put it in the buffer. [n_rollouts x H x n_own x n_obs x 3]
                        torch.norm(self._sphere_centers_diff_buf, dim=-1, keepdim=True,out=self._sphere_centers_diff_buf[:,:,:,:,0]) # Compute the distance between the sphere centers
                        torch.add(rad_collider, rad_collide_with, out=self._rad_sum_buf)
                        torch.sub(self._sphere_centers_diff_buf[:,:,:,:,0], self._rad_sum_buf, out=self._sphere_centers_diff_buf[:,:,:,:,0]) # Compute the distance between the sphere centers minus the sum of the radii.
                        self._sphere_centers_diff_buf[:,:,:,:,0].lt_(self.safety_margin) # 1 where the distance between surfaces is less than the safety margin (meaning: very close to collision) and 0 otherwise.
                        # self._tmp_buffer[env,:,:,collider,:] = env_prad[:,:,:,collider,:]
                        out_start_idx = self._idx_map_tree_to_flat[env][collider]
                        torch.sum(self._sphere_centers_diff_buf[:,:,:,:,0], dim=[2,3,4], out=self._out_buf[out_start_idx:out_start_idx+self.b,:]) # [n_rollouts x H] (Counting the number of times the safety margin is violated, for each step in each rollout (sum over all spheres of the robot and all obstacles).)
        else:
            pass    

        return self._out_buf

# class DynamicObsCollPredictorBatch2: 
#     def __init__(
#             self,
#             tensor_args, 
#             n_predictors_envwise:list[int]=[1,], # total number of robots {summing over all robots with prediction capabilities in all environments}
#             H=30, 
#             n_rollouts_per_predictor=400, 
#             n_own_spheres=61, 
#             n_obs_max_per_predictor=100,
#             cost_weight=100.0, 
#             p_R_envwise=list[list[torch.Tensor]]
#         ):
#         """
#         Initialize a batch of dynamic obstacle collision predictors.
#         Args:
#             tensor_args: pytorch tensor arguments.
#             n_predictors_envwise: list of ints, each int is the number of predictors (robots with predictive prediction capabilities) in the environment.
#                 n_predictors_envwise[i] is the number of predictors in the i'th environment.
#             H: int, the horizon length.
#             n_rollouts_per_predictor: int, the number of rollouts per predictor.
#             n_own_spheres: int, the number of self spheres for each predictor (robot).
#             n_obs_max_per_predictor: int, each int is the maximum number of obstacles for each predictor (robot). This will be the allocated buffer size for computations
#             cost_weight: float, the cost weight.
#             p_R_envwise: list of list of torch.Tensors,
#                 where p_R_envwise[i][j] is the i'th env j'thj robot base position (xyz) TODO: determine if to express in world or env frame.
#         """
#         self.n_envs = len(n_predictors_envwise)
#         self.n_predictors = sum(n_predictors_envwise)
#         self.n_rollouts_per_predictor = n_rollouts_per_predictor
#         self.n_rollouts_total = self.n_rollouts_per_predictor * self.n_predictors # num of rows in the output cost matrix
#         self.tensor_args = tensor_args
#         self.H = H
#         self.n_own_spheres = n_own_spheres
#         self.n_obs_max_per_predictor = n_obs_max_per_predictor
#         self.cost_weight = cost_weight
#         self.p_R_envwise = p_R_envwise

#         # mapping env idx i, predictor idx j (at the i'th environment) to start rollout idx in the aggregated
#         # rollout tensor. 
#         # (The last idx of this ij robot will be self.rollout_range_map[i][j] + self.n_rollouts_per_predictor - 1)
#         start_idx_tmp = 0
#         self.rollout_range_map = [] 
#         for i in range(self.n_envs):
#             for _ in range(n_predictors_envwise[i]):
#                 self.rollout_range_map.append(start_idx_tmp) 
#                 start_idx_tmp += self.n_rollouts_per_predictor
        

        
    
#         # Buffers for obstacles (spheres): position and radius
#         self.buffers_envwise:list[dict[str, torch.Tensor]] = []
#         for i in range(self.n_envs):
#             env_total_obs = self.n_obs_max_per_predictor * n_predictors_envwise[i]
#             env_rad_obs_buf = torch.zeros(env_total_obs, device=self.tensor_args.device) # [n_envs x n_obs] obstacles radii buffer
#             env_rad_obs_buf_unsqueezed = env_rad_obs_buf.reshape(1, *env_rad_obs_buf.shape) # [1 x n_obs]  Added 1 dimension for intermediate calculations
#             env_p_obs_buf = torch.zeros((self.H, env_total_obs, 3), device=self.tensor_args.device) # [H x n_obs x 3] pos and radius of the obstacles (spheres) over horizon
#             env_p_obs_buf_unsqueezed = env_p_obs_buf.reshape(1,H,1, env_total_obs, 3) # [1 x H x 1 x n_obs x 3] Added 1 dimension for intermediate calculations      
#             env_pairwise_ownobs_radsum_buf = env_rad_obs_buf_unsqueezed + env_rad_obs_buf_unsqueezed # [n_own x n_obs] matrix for own radius i + obstacle radius j for all possible own and obstacle sphere pairs (Note: this is broadcasted because its a sum of n_own x 1 + 1 x n_obs radii)
#             self.buffers_envwise.append(
#                 {
#                     "rad_obs_buf": env_rad_obs_buf,
#                     "rad_obs_buf_unsqueezed": env_rad_obs_buf_unsqueezed,
#                     "p_obs_buf": env_p_obs_buf,
#                     "p_obs_buf_unsqueezed": env_p_obs_buf_unsqueezed,
#                 }
#             )

#         # Buffers for own spheres: position and radius
#         self.p_own_buf = torch.empty(self.n_rollouts_total, H, self.n_own_spheres, 3, device=self.tensor_args.device) # [n_rollouts x H x n_own x 3] Own spheres positions buffer
        
#         self.p_own_buf_unsqueezed = self.p_own_buf.reshape(self.n_rollouts,self.H,self.n_own_spheres,1,3) # added 1 dimension for intermediate calculations [n_rollouts x H x n_own x 1 x 3]
#         self.rad_own_buf = torch.zeros(self.n_own_spheres, device=self.tensor_args.device) # [n_own] Own spheres radii buffer
#         self.rad_own_buf_unsqueezed = self.rad_own_buf.reshape(*self.rad_own_buf.shape, 1) # [n_own x 1] added 1 dimension for intermediate calculations

#         # Buffers for intermediate calculations
#         self.pairwise_ownobs_radsum_buf = self.rad_own_buf_unsqueezed + self.rad_obs_buf_unsqueezed # [n_own x n_obs] matrix for own radius i + obstacle radius j for all possible own and obstacle sphere pairs (Note: this is broadcasted because its a sum of n_own x 1 + 1 x n_obs radii)
#         self.pairwise_ownobs_radsum_buf_unsqueezed = self.pairwise_ownobs_radsum_buf.reshape(1,1, *self.pairwise_ownobs_radsum_buf.shape, 1) # [1 x 1 x n_own x n_obs x 1] Added 1 dimension for intermediate calculations
    
#         self.ownobs_diff_vector_buff = self.p_own_buf_unsqueezed - self.p_obs_buf_unsqueezed # [n_rollouts x H x n_own x n_obs x 3] A tensor for all pairs of the difference vectors between own and obstacle spheres. Each entry i,j will be storing the vector (p_own_buf[...][i] - p_obs_buf[...][j] where ... is the rollout, horizon and sphere indices)
#         self.pairwise_ownobs_surface_dist_buf = torch.zeros(self.ownobs_diff_vector_buff.shape[:-1] + (1,), device=self.tensor_args.device) # [n_rollouts x H x n_own x n_obs x 1] A tensor for all pairs of the (non negative) distance between own and obstacle spheres. Each entry i,j will be storing the signed distance (p_own_buf[...][i] - p_obs_buf[...][j]) - (rad_own_buf[i] + rad_obs_buf[j]) where ... is the rollout, horizon and sphere indices)
#         self.cost_mat_buf = torch.zeros(n_rollouts, H, device=self.tensor_args.device) # [n_rollouts x H] A tensor for the collision cost for each rollout and time step in the horizon. This is the output of the cost function.

#         # flags
#         self.init_rad_buffs = torch.tensor([0], device=self.tensor_args.device) # [1] If 1, the rad_obs_buffs are initialized (obstacles which should be set only once).
#         self.manually_express_p_own_in_world_frame = manually_express_p_own_in_world_frame # if True, the robot spheres positions are expressed in the world frame, otherwise they are expressed in the robot base frame.
#         if self.manually_express_p_own_in_world_frame:
#             self.p_R = p_R.to(self.tensor_args.device) # xyz of own base in world frame
#             self.p_R_broadcasted_buf = torch.zeros(self.p_own_buf.shape, device=self.tensor_args.device) + self.p_R # [n_rollouts x H x n_own x 3] A tensor for the robot spheres positions in the world frame.
    
#     def cost_fn(self, prad_own:torch.Tensor, env_query_idx:torch.Tensor):
#         """
#         Compute the collision cost for the robot spheres.
#         """
#         """
#         prad_own: 
#             torch.Tensor[n_rollouts_total, H, n_own_spheres (normally 65), 4],
#                 where n_rollouts_total = n_rollouts_per_robot * n_robots_total (over all environments) 
#             the robot sphere poses over the horizon, aggregated over the first dimension, matching the environments at the same row index in env_query_idx.
        
#         env_query_idx: 
#             torch.Tensor[n_rollouts_total, 1] 
    
#             More specifically:
#                 [{sum ij: n rollouts} # env i robot j, 1] # basically the number of rows is n rollouts per robot x total robot num (over all environments)
#                 Example: 3 envs [0,1,2], env 0 has 2 robots [0,1], env 1 has 1 robot [0], env 2 has 3 robots [0,1,2].
#                 So in total there are 6 robots, and env_query_idx is a tensor of shape [n_rollouts_total x 1] = [6 x num rollouts per robot,1].
#                 assume n_rollouts_per_robot is 2, then env_query_idx is a tensor of shape [6 x 2,1] = [12,1].
#                 and it will be 
#                     torch.Tesor([
#                     [0], # env 0 robot 0 rollout 0
#                     [0], # env 0 robot 0 rollout 1
#                     [0], # env 0 robot 1 rollout 0
#                     [0], # env 0 robot 1 rollout 1
#                     [1], # env 1 robot 0 rollout 0
#                     [1], # env 1 robot 0 rollout 1
#                     [2], # env 2 robot 0 rollout 0
#                     [2], # env 2 robot 0 rollout 1
#                     [2], # env 2 robot 1 rollout 0
#                     [2], # env 2 robot 1 rollout 1
#                     [2], # env 2 robot 2 rollout 0
#                     [2], # env 2 robot 2 rollout 1
#                     ])
#                     # 12 rows, each contains the env index of the rollout.
#                     # first 4 rows are env 0, next 2 are env 1, next 6 are env 2.
        
#         """
#         pass
    
#     def _get_idx_range(self, env_idx:int, predictor_idx:int):
#         """
#         Get the start and end indices of the rollouts for a given environment and predictor.
#         """

#         start_inclusive = self.rollout_range_map[env_idx][predictor_idx]  
#         end_inclusive = start_inclusive + self.n_rollouts_per_predictor
#         return start_inclusive, end_inclusive

if __name__ == "__main__":
    # x = [0] + list(np.cumsum([1,3,4]))[:-1]
    # print(x)
    
    
    n_envs = 3
    H = 30
    b = 400
    n_robot_spheres = 61
    cost_weight = 100.0
    # p_R_envwise = [
    #     [torch.zeros(3), torch.zeros(3), torch.zeros(3)],
    #     [torch.zeros(3), torch.zeros(3)],
    #     [torch.zeros(3), torch.zeros(3), torch.zeros(3)],
    # ]
    tensor_args = TensorDeviceType()
    n_robots_envwise = [3,2,3]
    predictor = DynamicObsCollPredictorBatch(tensor_args, n_envs, H, b, n_robot_spheres, cost_weight, n_robots_envwise)
    print(predictor._idx_map_tree_to_flat)
    print(predictor._out_buf.shape)
    print(predictor._p_collider_buf.shape)
    print(predictor._p_collide_with_buf.shape)
    print(predictor._sphere_centers_diff_buf.shape)
    print(get_size_mb(predictor._sphere_centers_diff_buf))
    
    
    env_query_idx = torch.zeros(predictor.b*n_robots_envwise[0], 1)
    for i in range(1, len(n_robots_envwise)):
        env_query_idx = torch.cat([env_query_idx, torch.ones(predictor.b*n_robots_envwise[i], 1) * i], dim=0)

                              
    print(env_query_idx.shape)
    print(env_query_idx)
                    #   torch.cat([torch.zeros(predictor.b*len(p_R_envwise[0]), 1), 
                    #             torch.ones(predictor.b*len(p_R_envwise[1]), 1),
                    #             torch.ones(predictor.b*len(p_R_envwise[2]), 1) + 1], dim=0))
    # print(predictor._sphere_centers_diff_buf.shape)
    # print(predictor._rad_sum_buf.shape)
    # print(predictor._sphere_centers_diff_buf[:,:,:,:,0].shape)
    # print(predictor._sphere_centers_diff_buf[:,:,:,:,0])
    