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
    
    # def activate(self, p_obs:torch.Tensor, rad_obs:torch.Tensor):
    #     """
    #     Activate the collision checker.
    #     p_obs: tensor of shape [H, n_obs, 3]. The poses of the obstacles.
    #     rad_obs: tensor of shape [n_obs]. The radii of the obstacles.
    #     """
    #     # assert self.rad_obs_buf.sum() > 0, "Error: Must set the obstacles (radii) before activating the collision checker"
    #     assert p_obs.ndim == 3, "Error: The obstacle poses must be a 3D tensor"
    #     assert p_obs.shape[0] == self.H, "Error: The number of time steps in the obstacle poses must be equal to the horizon length"
    #     assert p_obs.shape[1] == self.n_obs, "Error: The number of obstacles must be equal to the number of obstacle poses"
    #     assert p_obs.shape[2] == 3, "Error: The obstacle poses must be in 3D"
        
    #     assert rad_obs.ndim == 1, "Error: The obstacle radii must be a 1D tensor"
    #     assert rad_obs.shape[0] == self.n_obs, "Error: The number of obstacles must be equal to the number of obstacle radii"

    #     self.p_obs_buf.copy_(p_obs)
    #     self.rad_obs_buf.copy_(rad_obs)
    #     self.rad_obs_buf_unsqueezed.copy_(self.rad_obs_buf.view_as(self.rad_obs_buf_unsqueezed)) 

        
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

     
# class BatchDynamicObsCollChecker:
#     def __init__(self, tensor_args, col_pred_list:List[DynamicObsCollPredictor]):
#         """
#         Initialize a batch of dynamic obstacle collision checkers.
#         Args:
#             tensor_args: pytorch tensor arguments.
#         """
#         self.tensor_args = tensor_args
#         self.n_colliders = len(col_pred_list) 
        
#         self.streams = [torch.cuda.Stream() for _ in range(self.n_colliders)]
#         self.graphs:List[torch.cuda.CUDAGraph] = [torch.cuda.CUDAGraph() for _ in range(self.n_colliders)]
#         self.collision_checkers:List[DynamicObsCollPredictor] = [DynamicObsCollPredictor(self.tensor_args,None,self.H_list[i],self.n_rollouts_list[i],self.n_own_spheres_list[i],self.n_obs_list[i]) for i in range(self.n_colliders)]  #


#     def capture(self):
        
#         for i in range(self.n_colliders):
#             prad_obs_static = self.prad_obs_static_list[i]
#             self.collision_checkers[i].update(prad_obs_static[:3])

#             prad_own_static = self.prad_own_static_list[i]  # use dummy data with same shape as runtime data
#             stream = self.streams[i]
#             graph = self.graphs[i]
#             checker = self.collision_checkers[i]

#             with torch.cuda.stream(stream):
#                 torch.cuda.synchronize()  # ensure the stream is idle
#                 with torch.cuda.graph(graph):
#                     checker.cost_fn(prad_own_static)

    

#     def run_graph(self, i, prad_own):
#         stream = self.streams[i]
#         checker = self.collision_checkers[i]
#         # Copy the actual input to the pre-allocated buffer
#         checker.prad_own_buf.copy_(prad_own)  # `prad_own_input` used inside `cost_fn`
#         with torch.cuda.stream(stream):
#             self.graphs[i].replay()



# # In your timestep loop:
# if __name__ == "__main__":
#     h_list = [10, 10]
#     n_rollouts_list = [20, 20]
#     n_own_spheres_list = [12,12]
#     n_obs_list = [12,23]
#     batch_dynamic_obs_coll_checker = BatchDynamicObsCollChecker(TensorDeviceType(), H_list=h_list, n_rollouts_list=n_rollouts_list, n_own_spheres_list=n_own_spheres_list, n_obs_list=n_obs_list)
#     batch_dynamic_obs_coll_checker.capture()
#     threads = []

#     while True:
#         for i in range(batch_dynamic_obs_coll_checker.n_colliders):
#             new_prad_own = torch.randn(h_list[i], n_own_spheres_list[i], 4, device=batch_dynamic_obs_coll_checker.tensor_args.device)
#             t = threading.Thread(target=batch_dynamic_obs_coll_checker.run_graph, args=(i, new_prad_own))
#             t.start()
#             threads.append(t)

#         for t in threads:
#             t.join()

#         # Optionally synchronize
#         torch.cuda.synchronize()




        
# class BatchDynamicObsCollChecker:
#     def __init__(self, tensor_args, H_list:list[int], n_rollouts_list:list[int], n_own_spheres_list:list[int], n_obs_list:list[int]):
#         """
#         Initialize a batch of dynamic obstacle collision checkers.
#         Args:
#             tensor_args: pytorch tensor arguments.
#             n_colliders: number of robots to check collision for
#             H: horizon length
#             n_rollouts: number of rollouts ()
#             n_own_spheres: number of own spheres
#             n_obs: number of obstacles
#         """
#         self.tensor_args = tensor_args
#         self.n_colliders = len(H_list) 
#         self.H_list = H_list # list of horizons, one for each robot
#         self.n_rollouts_list = n_rollouts_list # list of number of rollouts, one for each robot
#         self.n_own_spheres_list = n_own_spheres_list # list of number of own spheres, one for each robot
#         self.n_obs_list = n_obs_list # list of number of obstacles, one for each robot

#         self.prad_own_static_list:List[torch.Tensor] = [torch.zeros(
#             self.n_rollouts_list[i], # number of rollouts
#             self.H_list[i], # horizon length
#             self.n_own_spheres_list[i], # number of own spheres
#             4, # 4: position and radius
#             device=self.tensor_args.device) for i in range(self.n_colliders)]
        
#         self.prad_obs_static_list:List[torch.Tensor] = [torch.zeros(
#             self.n_rollouts_list[i], # number of rollouts
#             self.H_list[i], # horizon length
#             self.n_obs_list[i], # number of obstacles
#             4, # 4: position and radius
#             device=self.tensor_args.device) for i in range(self.n_colliders)]
        
            
#         self.streams = [torch.cuda.Stream() for _ in range(self.n_colliders)]
#         self.graphs:List[torch.cuda.CUDAGraph] = [torch.cuda.CUDAGraph() for _ in range(self.n_colliders)]
#         self.collision_checkers:List[DynamicObsCollPredictor] = [DynamicObsCollPredictor(self.tensor_args,None,self.H_list[i],self.n_rollouts_list[i],self.n_own_spheres_list[i],self.n_obs_list[i]) for i in range(self.n_colliders)]  #


#         for i in range(self.n_colliders):
#             self.collision_checkers[i].activate(self.prad_obs_static_list[i][0,...,:3], self.prad_obs_static_list[i][0,0,...,3].flatten())
        
    
#     def capture(self):
        
#         for i in range(self.n_colliders):
#             prad_obs_static = self.prad_obs_static_list[i]
#             self.collision_checkers[i].update(prad_obs_static[:3])

#             prad_own_static = self.prad_own_static_list[i]  # use dummy data with same shape as runtime data
#             stream = self.streams[i]
#             graph = self.graphs[i]
#             checker = self.collision_checkers[i]

#             with torch.cuda.stream(stream):
#                 torch.cuda.synchronize()  # ensure the stream is idle
#                 with torch.cuda.graph(graph):
#                     checker.cost_fn(prad_own_static)

    

#     def run_graph(self, i, prad_own):
#         stream = self.streams[i]
#         checker = self.collision_checkers[i]
#         # Copy the actual input to the pre-allocated buffer
#         checker.prad_own_buf.copy_(prad_own)  # `prad_own_input` used inside `cost_fn`
#         with torch.cuda.stream(stream):
#             self.graphs[i].replay()



# # In your timestep loop:
# if __name__ == "__main__":
#     h_list = [10, 10]
#     n_rollouts_list = [20, 20]
#     n_own_spheres_list = [12,12]
#     n_obs_list = [12,23]
#     batch_dynamic_obs_coll_checker = BatchDynamicObsCollChecker(TensorDeviceType(), H_list=h_list, n_rollouts_list=n_rollouts_list, n_own_spheres_list=n_own_spheres_list, n_obs_list=n_obs_list)
#     batch_dynamic_obs_coll_checker.capture()
#     threads = []

#     while True:
#         for i in range(batch_dynamic_obs_coll_checker.n_colliders):
#             new_prad_own = torch.randn(h_list[i], n_own_spheres_list[i], 4, device=batch_dynamic_obs_coll_checker.tensor_args.device)
#             t = threading.Thread(target=batch_dynamic_obs_coll_checker.run_graph, args=(i, new_prad_own))
#             t.start()
#             threads.append(t)

#         for t in threads:
#             t.join()

#         # Optionally synchronize
#         torch.cuda.synchronize()





if __name__ == "__main__":
    x = [0] + list(np.cumsum([1,3,4]))[:-1]
    print(x)