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
from projects_root.projects.dynamic_obs.dynamic_obs_predictor.runtime_topics import get_topics
from projects_root.utils.plot_spheres import SphereVisualizer
from projects_root.utils.quaternion import integrate_quat
from projects_root.projects.dynamic_obs.dynamic_obs_predictor.utils import shift_tensor_left, mask_decreasing_values
from concurrent.futures import ProcessPoolExecutor, wait
from projects_root.utils.transforms import transform_poses_batched, transform_robot_positions_to_world, create_optimized_collision_checker_buffers, transform_positions_with_precomputed_matrix
from curobo.geom.transform import batch_transform_points

import time



class DynamicObsCollPredictor:
    """
    
    This class is used to predict and compute costs for the collision of the dynamic obstacles in the world.
    You can think of this class as a collection of H collision checkers, one for each time step in the horizon + functionality to compute costs for the MPC (see cost_fn function).
    Each individual collision checker which it contains, is inspired by: https://curobo.org/get_started/2c_world_collision.html
    """
    

    def __init__(self, 
                 tensor_args, 
                 H=30, 
                 n_rollouts=400, 
                 n_own_spheres=65, # total number of spheres of the robot (including ones that we don't want to check for collision)
                 n_obs=65, # total number of spheres of the obstacles in the world (including ones that we don't want to check for collision)
                 cost_weight=100.0, 
                 X = [0,0,0,1,0,0,0],
                 sparse_steps:dict={'use': False, 'ratio': 0.5},
                 sparse_spheres:dict={'exclude_self': [], 'exclude_others': []}, # list of ints, each int is the index of the sphere to exclude from the collision check.
                 ):
        """ Initialize H dynamic obstacle collision checker, for each time step in the horizon, 
        as well as setting the cost function parameters for the dynamic obstacle cost function.


        Args:
            tensor_args: pytorch tensor arguments.
            cache (dict): collision checker cache for the pre-defined dynamic primitives.
            H (int, optional): Defaults to 30. The horizon length- number of states in the trajectory during trajectory optimization.
            n_checkers(int, optional): Defaults to H (marked by passing -1). The number of collision checkers to use. If n_checkers is not H, then the collision checkers will be used in a sliding window fashion.
            n_rollouts (int, optional): Defaults to 400. The number of rollouts. TODO: Should be taken from the mpc config.
            cost_weight: weight for the dynamic obstacle cost function (cost term weight). This is a hyper-parameter, unlike the weight_col_check which you should leave as 1. Default value is 100000, as by the original primitive collision cost weight of the mpc.
            """
        

        self.X = X # the pose (x y z qw qx qy qz) of the robot in the world frame
        self.tensor_args = tensor_args 
        self.H = H
        
        self.n_rollouts = n_rollouts
        self.n_own_spheres = n_own_spheres

        # control sparsity over horizon (for efficiency)
        self.sparse_steps = sparse_steps
        if self.sparse_steps['use']:
            assert self.sparse_steps['ratio'] > 0 and self.sparse_steps['ratio'] <= 1, "Error: The ratio must be between 0 and 1"
            self.n_sampling_steps = int(self.H * self.sparse_steps['ratio']) # number of timesteps to sample for collision
            self.sampling_timesteps = np.linspace(0, self.H-1, self.n_sampling_steps, dtype=int) # evenly spaced timesteps in the horizon
        else:
            self.n_sampling_steps = self.H
            self.sampling_timesteps = np.arange(self.H)
    

        self.cost_weight = cost_weight

        # control sparsity over spheres - filter out spheres which are not needed/less relevant (for efficiency)
        self.sparse_spheres = sparse_spheres # sparse spheres config
        
        self.n_own_spheres = self.n_own_spheres # n_own_spheres # number of valid spheres of the robot (ignoring 4 spheres which are not valid due to negative radius)
        
        # Filter out invalid sphere indices that are outside the range of available spheres
        valid_exclude_self = [idx for idx in self.sparse_spheres['exclude_self'] if 0 <= idx < self.n_own_spheres]
        self.valid_own_spheres = np.array(list(set(list(range(self.n_own_spheres))) - set(valid_exclude_self)))
        
        self.n_obs = n_obs # number of valid obstacles (ignoring 4 spheres which are not valid due to negative radius)
        
        # Filter out invalid sphere indices that are outside the range of available obstacle spheres
        valid_exclude_others = [idx for idx in self.sparse_spheres['exclude_others'] if 0 <= idx < self.n_obs]
        self.valid_obs_spheres = np.array(list(set(list(range(self.n_obs))) - set(valid_exclude_others)))
        
        # Pre-compute index tensors for efficient indexing
        self.sampling_timesteps_tensor = torch.tensor(self.sampling_timesteps, device=self.tensor_args.device, dtype=torch.long)
        self.valid_own_spheres_tensor = torch.tensor(self.valid_own_spheres, device=self.tensor_args.device, dtype=torch.long)
        self.valid_obs_spheres_tensor = torch.tensor(self.valid_obs_spheres, device=self.tensor_args.device, dtype=torch.long)
        
        # Compute effective dimensions after filtering
        self.n_valid_own = len(self.valid_own_spheres)
        self.n_valid_obs = len(self.valid_obs_spheres)
        
        # Buffers for obstacles (spheres): position and radius
        self.rad_obs_buf = torch.zeros(self.n_valid_obs, device=self.tensor_args.device) # [n_valid_obs] obstacles radii buffer
        
        # Pre-allocate obstacle position buffer in the final shape to avoid reshaping
        self.p_obs_buf = torch.zeros((self.n_sampling_steps, self.n_valid_obs, 3), device=self.tensor_args.device)
        # Pre-allocate in broadcast-ready shape to eliminate reshaping in cost_fn
        self.p_obs_buf_broadcast = torch.zeros((1, self.n_sampling_steps, 1, self.n_valid_obs, 3), device=self.tensor_args.device)

        # Buffers for own spheres: position and radius  
        self.p_own_buf = torch.empty(n_rollouts, self.n_sampling_steps, self.n_valid_own, 3, device=self.tensor_args.device)
        # self.XFiltered_own_R = torch.zeros(n_rollouts, self.n_sampling_steps, self.n_valid_own, 7, device=self.tensor_args.device)
        # self.XFiltered_own_R[:,:,:,3] = 1 # this is making the quat 1,0,0,0, which is the identity quaternion

        self.p_own_buf_unfiltered = torch.zeros(n_rollouts, self.H, self.n_own_spheres, 3, device=self.tensor_args.device)
        self.X_own_R = torch.zeros(n_rollouts, self.H, self.n_own_spheres, 7, device=self.tensor_args.device)
        self.X_own_R[:,:,:,3] = 1 # this is making the quat 1,0,0,0, which is the identity quaternion

        # Pre-allocate in broadcast-ready shape to eliminate reshaping in cost_fn
        self.p_own_buf_broadcast = torch.empty(self.n_rollouts, self.n_sampling_steps, self.n_valid_own, 1, 3, device=self.tensor_args.device)
        
        self.rad_own_buf = torch.zeros(self.n_valid_own, device=self.tensor_args.device) # [n_valid_own] Own spheres radii buffer

        # Pre-compute radius sum matrix in final broadcast shape to avoid repeated operations
        # Shape: [1, 1, n_valid_own, n_valid_obs, 1] - ready for broadcasting
        self.pairwise_radsum_broadcast = torch.zeros(1, 1, self.n_valid_own, self.n_valid_obs, 1, device=self.tensor_args.device)
    
        # Main computation buffers - pre-allocated in final shapes
        self.ownobs_diff_vector_buff = torch.zeros(self.n_rollouts, self.n_sampling_steps, self.n_valid_own, self.n_valid_obs, 3, device=self.tensor_args.device)
        self.pairwise_surface_dist_buf = torch.zeros(self.n_rollouts, self.n_sampling_steps, self.n_valid_own, self.n_valid_obs, 1, device=self.tensor_args.device)
        
        # Output buffers
        self.cost_mat_buf = torch.zeros(n_rollouts, H, device=self.tensor_args.device) # [n_rollouts x H] 
        self.tmp_cost_mat_buf_sparse = torch.zeros(n_rollouts, self.n_sampling_steps, device=self.tensor_args.device)
        
        # Pre-compute projection parameters for sparse-to-full horizon mapping
        if self.sparse_steps['use']:
            self.base_repeat = self.H // self.n_sampling_steps
            self.remainder = self.H % self.n_sampling_steps
            if self.remainder != 0:
                # Pre-compute repeat counts to avoid recreation each time
                self.repeat_counts = torch.full((self.n_sampling_steps,), self.base_repeat, dtype=torch.long, device=self.tensor_args.device)
                self.repeat_counts[:self.remainder] += 1
        
        # flags
        self.init_rad_buffs = torch.tensor([0], device=self.tensor_args.device) # [1] If 1, the rad_obs_buffs are initialized (obstacles which should be set only once).
    
        self.rotation_matrix = None
        self.world_translation = None
        self.transform_matrix_dirty = True
        
    def _project_sparse_to_full_horizon(self, sparse_matrix: torch.Tensor, full_matrix: torch.Tensor):
        """
        Project a sparse matrix from (n_rollouts, n_sampling_steps) to (n_rollouts, H).
        Optimized version with pre-computed parameters.
        """
        if self.sparse_steps['use']:
            if self.remainder == 0:
                # Simple case: all columns repeated equally
                full_matrix.copy_(sparse_matrix.repeat_interleave(self.base_repeat, dim=1))
            else:
                # Complex case: use pre-computed repeat counts
                full_matrix.copy_(sparse_matrix.repeat_interleave(self.repeat_counts, dim=1))
        else:
            # If not using sparse steps, just copy directly
            full_matrix.copy_(sparse_matrix)
        
    def set_obs_rads(self, rad_obs:torch.Tensor):
        """
        rad_obs: tensor of shape [n_obs]. The radii of the obstacles.
        """
        assert rad_obs.ndim == 1, "Error: The obstacle radii must be a 1D tensor"
        
        # Direct indexing and copy
        self.rad_obs_buf.copy_(rad_obs[self.valid_obs_spheres])

    def set_own_rads(self, rad_own:torch.Tensor):
        """
        Set the radii of the own spheres and pre-compute radius sum matrix.
        """
        # Copy own radii
        self.rad_own_buf.copy_(rad_own[self.valid_own_spheres])
        
        # Pre-compute pairwise radius sums in broadcast-ready shape
        # Broadcasting: [n_valid_own, 1] + [1, n_valid_obs] -> [n_valid_own, n_valid_obs]
        rad_sum = self.rad_own_buf.unsqueeze(1) + self.rad_obs_buf.unsqueeze(0)
        # Store in final broadcast shape [1, 1, n_valid_own, n_valid_obs, 1]
        self.pairwise_radsum_broadcast[0, 0, :, :, 0] = rad_sum
        
        self.init_rad_buffs[0] = 1
    
    def update(self, p_obs:torch.Tensor):
        """
        Update the positions of the obstacles (sphere centers).
        note: assuming that the poses are expressed in the world frame [0,0,0,1,0,0,0].
        Args:
            p_obs: tensor of shape [H, n_obs, 3]. The poses of the obstacles.
        """
        # Optimized indexing using pre-computed tensors
        temp_obs = torch.index_select(p_obs, 0, self.sampling_timesteps_tensor)  # Select timesteps
        temp_obs = torch.index_select(temp_obs, 1, self.valid_obs_spheres_tensor)  # Select spheres
        self.p_obs_buf.copy_(temp_obs)
        
        # Update broadcast-ready buffer
        self.p_obs_buf_broadcast[0, :, 0, :, :] = self.p_obs_buf

    def update_world_pose(self, world_pose):
        """Update world pose and pre-compute transformation matrix."""
        self.X = world_pose
        if not hasattr(self, 'X_tensor') or self.X_tensor is None:
            self.X_tensor = torch.tensor(self.X, device=self.tensor_args.device, dtype=torch.float32)
        
        # Pre-compute rotation matrix for ultra-fast transforms
        self.rotation_matrix, self.world_translation = create_optimized_collision_checker_buffers(
            self.n_rollouts, self.H, self.n_own_spheres, self.X_tensor, self.tensor_args.device
        )
        self.transform_matrix_dirty = False

    def cost_fn(self, prad_own_R: torch.Tensor, safety_margin=0.1):
        """Ultra-optimized collision cost computation."""
        
        # Update transformation matrix if needed
        if self.transform_matrix_dirty or self.rotation_matrix is None:
            self.update_world_pose(self.X)
        
        # Extract positions from robot frame poses
        robot_positions = prad_own_R[:, :, :, :3]  # (n_rollouts, H, n_spheres, 3)
        
        # ULTRA-FAST transformation using pre-computed matrix
        self.p_own_buf_unfiltered = transform_positions_with_precomputed_matrix(
            robot_positions, self.rotation_matrix, self.world_translation
        )
        
        # filter out the timesteps that are not needed
        p_spheres_tmp_time_filtered = torch.index_select(self.p_own_buf_unfiltered, 1, self.sampling_timesteps_tensor) # Select timesteps
        # filter out the spheres that are not needed
        self.p_own_buf = torch.index_select(p_spheres_tmp_time_filtered, 2, self.valid_own_spheres_tensor)
        
        # Update broadcast-ready buffer
        self.p_own_buf_broadcast[:, :, :, 0, :] = self.p_own_buf
        
        # Fused difference computation: own_pos - obs_pos
        # Broadcasting: [n_rollouts, n_sampling_steps, n_valid_own, 1, 3] - [1, n_sampling_steps, 1, n_valid_obs, 3]
        torch.sub(self.p_own_buf_broadcast, self.p_obs_buf_broadcast, out=self.ownobs_diff_vector_buff)
        
        # Compute L2 norm (distance between sphere centers)
        torch.norm(self.ownobs_diff_vector_buff, dim=-1, keepdim=True, out=self.pairwise_surface_dist_buf)
        
        # Subtract radius sums to get surface-to-surface distance
        self.pairwise_surface_dist_buf.sub_(self.pairwise_radsum_broadcast)
        
        # Check collision condition and count violations
        # Using lt_ for in-place comparison, then sum to count violations
        collision_mask = self.pairwise_surface_dist_buf.lt_(safety_margin)
        torch.sum(collision_mask, dim=[2, 3, 4], out=self.tmp_cost_mat_buf_sparse)
        
        # Interpolate the sparse costs over the horizon: (Project sparse results to full horizon, to get a valid cost matrix for the whole horizon)
        self._project_sparse_to_full_horizon(self.tmp_cost_mat_buf_sparse, self.cost_mat_buf)
        
        # Apply cost weight
        self.cost_mat_buf.mul_(self.cost_weight)
        
        return self.cost_mat_buf



if __name__ == "__main__":
    x = [0] + list(np.cumsum([1,3,4]))[:-1]
    print(x)