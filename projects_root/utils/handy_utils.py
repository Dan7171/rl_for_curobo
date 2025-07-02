import torch
from projects_root.utils.transforms import transform_poses_batched

def get_rollouts_in_world_frame(visual_rollouts, X_R):
        """
        Get visual rollouts transformed to world frame for visualization.
        
        Args:
            X_R: Base frame pose [x, y, z, qw, qx, qy, qz] of the robot (necesarry because the rollouts are in the robot's base frame).
        Returns:
            torch.Tensor: Visual rollouts with poses in world frame
        """
        p_visual_rollouts_robotframe = visual_rollouts # solver.get_visual_rollouts()
        q_visual_rollouts_robotframe = torch.empty(p_visual_rollouts_robotframe.shape[:-1] + torch.Size([4]), device=p_visual_rollouts_robotframe.device)
        q_visual_rollouts_robotframe[...,:] = torch.tensor([1,0,0,0], device=p_visual_rollouts_robotframe.device, dtype=p_visual_rollouts_robotframe.dtype) 
        visual_rollouts = torch.cat([p_visual_rollouts_robotframe, q_visual_rollouts_robotframe], dim=-1)                
        visual_rollouts = transform_poses_batched(visual_rollouts, X_R) # transform to world frame
        return visual_rollouts