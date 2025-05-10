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
from concurrent.futures import ProcessPoolExecutor, wait
import time



class DynamicObsCollPredictor:
    """
    
    This class is used to predict and compute costs for the collision of the dynamic obstacles in the world.
    You can think of this class as a collection of H collision checkers, one for each time step in the horizon + functionality to compute costs for the MPC (see cost_fn function).
    Each individual collision checker which it contains, is inspired by: https://curobo.org/get_started/2c_world_collision.html
    """
    

    def __init__(self, tensor_args, step_dt_traj_opt, H=30, n_rollouts=400, n_own_spheres=61, n_obs=61, cost_weight=100):
        """ Initialize H dynamic obstacle collision checker, for each time step in the horizon, 
        as well as setting the cost function parameters for the dynamic obstacle cost function.


        Args:
            tensor_args: pytorch tensor arguments.
            cache (dict): collision checker cache for the pre-defined dynamic primitives.
            step_dt_traj_opt (float): Time passes between each step in the trajectory. This is what the mpc/curobo solver assumes time delta between steps in horizon is when parforming path planning.
            H (int, optional): Defaults to 30. The horizon length. TODO: Should be taken from the mpc config.
            n_checkers(int, optional): Defaults to H (marked by passing -1). The number of collision checkers to use. If n_checkers is not H, then the collision checkers will be used in a sliding window fashion.
            n_rollouts (int, optional): Defaults to 400. The number of rollouts. TODO: Should be taken from the mpc config.
            cost_weight: weight for the dynamic obstacle cost function (cost term weight). This is a hyper-parameter, unlike the weight_col_check which you should leave as 1. Default value is 100000, as by the original primitive collision cost weight of the mpc.
            # REMOVED FOR NOW:mask_decreasing_cost_entries: if True, the cost matrix will be modified so that only the first violation of the safety margin is considered.
            """
        self._init_counter = 0 # number of steps until cuda graph initiation. Here Just for debugging. Can be removed. with no effect on the code. 
        self.tensor_args = tensor_args 
        self.H = H 
        self.n_rollouts = n_rollouts 
        self.step_dt_traj_opt = step_dt_traj_opt 
        self.cost_weight = cost_weight

        self.n_own_spheres = n_own_spheres # number of valid spheres of the robot (ignoring 4 spheres which are not valid due to negative radius)
        self.n_obs = n_obs # number of valid obstacles (ignoring 4 spheres which are not valid due to negative radius)
        
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
        # self.init_obs = torch.tensor([0]) # [1] If 1, the obstacles are initialized.
        self.init_rad_buffs = torch.tensor([0]) # [1] If 1, the rad_obs_buffs are initialized (obstacles which should be set only once).
        # self.is_active = torch.tensor([0]) # [1] If 1, the collision checker is active.
    
    def activate(self, p_obs:torch.Tensor, rad_obs:torch.Tensor):
        """
        Activate the collision checker.
        """
        # assert self.rad_obs_buf.sum() > 0, "Error: Must set the obstacles (radii) before activating the collision checker"
        self.p_obs_buf.copy_(p_obs)
        self.rad_obs_buf.copy_(rad_obs)
        self.rad_obs_buf_unsqueezed.copy_(self.rad_obs_buf.view_as(self.rad_obs_buf_unsqueezed)) 
        # self.is_active[0] = 1 

    # def reset_obs(self, p_obs:torch.Tensor, rad_obs:torch.Tensor):
    #     """
    #     Initialize the obstacles.
    #     """
    #     # self.p_obs_buf = torch.cat([self.p_obs_buf, p_obs], dim=1)
    #     # self.rad_obs_buf = torch.cat([self.rad_obs_buf, rad_obs], dim=0)
        
    #     self.p_obs_buf.copy_(p_obs)
    #     self.rad_obs_buf.copy_(rad_obs)
    #     self.init_obs[0] = 1
        
    
    def update(self, p_obs:torch.Tensor):
        """
        Update the poses of the obstacles.
        Args:
            p_obs: tensor of shape [H, n_obs, 3]. The poses of the obstacles.
        """
        # self.p_obs = p_obs
        self.p_obs_buf.copy_(p_obs) # copy p_obs to self.p_obs in place.
    
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


        if not bool(self.init_rad_buffs[0]):
            self.rad_own_buf.copy_(prad_own[0,0,:self.n_own_spheres,3]) # [n_own] init own spheres radii
            self.rad_own_buf_unsqueezed.copy_(self.rad_own_buf.view_as(self.rad_own_buf_unsqueezed)) 
            # self.rad_obs_buf_unsqueezed.copy_(self.rad_obs_buf.view_as(self.rad_obs_buf_unsqueezed)) 
            torch.add(self.rad_own_buf_unsqueezed, self.rad_obs_buf_unsqueezed, out=self.pairwise_ownobs_radsum_buf) # broadcasted addition of rad_own and rad_obs
            self.pairwise_ownobs_radsum_buf_unsqueezed[0,0,:,:,0].copy_(self.pairwise_ownobs_radsum_buf) 
            self.init_rad_buffs[0] = 1 # so that the following code will not re-initialize the buffers again (its just for efficiency).
        
        # Every time
        self.p_own_buf.copy_(prad_own[:,:,:self.n_own_spheres,:3]) # read robot spheres positions to the "prad_own" buffer.
        self.p_own_buf_unsqueezed.copy_(self.p_own_buf.reshape(self.n_rollouts,self.H,self.n_own_spheres,1,3)) # Copy the reshaped version as well.
        self.p_obs_buf_unsqueezed.copy_(self.p_obs_buf.reshape(1,self.H,1,self.n_obs,3)) # Copy the reshaped version 
        torch.sub(self.p_own_buf_unsqueezed, self.p_obs_buf_unsqueezed, out=self.ownobs_diff_vector_buff) # Compute the difference vector between own and obstacle spheres and put it in the buffer. [n_rollouts x H x n_own x n_obs x 3]
        torch.norm(self.ownobs_diff_vector_buff, dim=-1, keepdim=True, out=self.pairwise_ownobs_surface_dist_buf) # Compute the distance between each own and obstacle sphere centers and put it in the buffer. n_rollouts x H x n_own x n_obs x 1
        self.pairwise_ownobs_surface_dist_buf.sub_(self.pairwise_ownobs_radsum_buf_unsqueezed) # subtract the sum of the radii from the distance (so we now get the distance between the spheres surfaces and not the centers). [n_rollouts x H x n_own x n_obs x 1]
        self.pairwise_ownobs_surface_dist_buf.lt_(safety_margin) # [n_rollouts x H x n_own x n_obs x 1] 1 where the distance between surfaces is less than the safety margin (meaning: very close to collision) and 0 otherwise.
        torch.sum(self.pairwise_ownobs_surface_dist_buf, dim=[2,3,4], out=self.cost_mat_buf) # [n_rollouts x H] (Counting the number of times the safety margin is violated, for each step in each rollout (sum over all spheres of the robot and all obstacles).)
        self.cost_mat_buf.mul_(self.cost_weight) # muliply the cost by the cost weight.
        return self.cost_mat_buf # dynamic_coll_cost_matrix 

     

        




    

    



