import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Optional

import torch
from warp.context import Union

def transform_pose_between_frames(pose:Union[list[float],torch.Tensor], frame_expressed_in:Optional[Union[list[float],torch.Tensor]]=None, frame_express_at:Optional[Union[list[float],torch.Tensor]]=None):
    """
    Transforms pose expressed in some frame F1 (frame_expressed_in), from frame F1 to frame F2 (frame_express_at).
    Both frame_expressed_in and frame_express_at are expressed in the world frame!

    If frame_expressed_in or frame_express_at is not provided, it is assumed that it is the world frame.
    Meaning, that if frame_expressed_in is not provided, it is assumed that the pose is a pose in the world frame.
    And if frame_express_at is not provided, it is assumed that we want to transform the pose to the world frame.
    
    Parameters:
    - pose: [px, py, pz, qw, qx, qy, qz] expressed in F1
    - frame_expressed_in: [px, py, pz, qw, qx, qy, qz] pose of F1 in world (the frame the pose is expressed in)
    - frame_express_at: [px, py, pz, qw, qx, qy, qz] pose of F2 in world (the frame the pose is transformed to)
    
    Returns:
    - pose_in_f2: [px, py, pz, qw, qx, qy, qz] expressed in F2
    """
    
    def decompose(pose):
        
        pos = np.array(pose[:3])
        quat = np.array(pose[3:])
        rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # convert qw,qx,qy,qz -> x,y,z,w
        return pos, rot

    def compose(pos, rot):
        quat = rot.as_quat()  # x,y,z,w
        return np.concatenate([pos, [quat[3], quat[0], quat[1], quat[2]]])  # to qw,qx,qy,qz

    world_pose = [0.0, 0, 0, 1, 0, 0, 0]
    if frame_expressed_in is None:
        frame_expressed_in = world_pose
    if frame_express_at is None:
        frame_express_at = world_pose
    # Decompose all poses
    if isinstance(pose, torch.Tensor):
        pose = pose.cpu().numpy().tolist()
    if isinstance(frame_expressed_in, torch.Tensor):
        frame_expressed_in = frame_expressed_in.cpu().numpy().tolist()
    if isinstance(frame_express_at, torch.Tensor):
        frame_express_at = frame_express_at.cpu().numpy().tolist()

    p_f1, r_f1 = decompose(frame_expressed_in)
    p_f2, r_f2 = decompose(frame_express_at)
    p_rel, r_rel = decompose(pose)
    
    # Pose in world frame: T_world = T_f1 * T_rel
    p_world = r_f1.apply(p_rel) + p_f1
    r_world = r_f1 * r_rel

    # Inverse of F2 pose
    r_f2_inv = r_f2.inv()
    p_f2_inv = -r_f2_inv.apply(p_f2)

    # Transform to F2: T_f2 = T_f2_inv * T_world
    p_in_f2 = r_f2_inv.apply(p_world) + p_f2_inv
    r_in_f2 = r_f2_inv * r_world

    return compose(p_in_f2, r_in_f2)



def transform_poses_batched(poses: torch.Tensor, frame_expressed_in: Optional[list[float]] = None, frame_express_at: Optional[list[float]] = None) -> torch.Tensor:
    """
    Transforms batched poses expressed in some frame F1 (frame_expressed_in), from frame F1 to frame F2 (frame_express_at).
    Both frame_expressed_in and frame_express_at are expressed in the world frame!

    If frame_expressed_in or frame_express_at is not provided, it is assumed that it is the world frame.
    Meaning, that if frame_expressed_in is not provided, it is assumed that the poses are in the world frame.
    And if frame_express_at is not provided, it is assumed that we want to transform the poses to the world frame.
    
    Parameters:
    - poses: torch.Tensor of shape (..., 7) where last dimension is [px, py, pz, qw, qx, qy, qz] expressed in F1
    - frame_expressed_in: [px, py, pz, qw, qx, qy, qz] pose of F1 in world (the frame the poses are expressed in)
    - frame_express_at: [px, py, pz, qw, qx, qy, qz] pose of F2 in world (the frame the poses are transformed to)
    
    Returns:
    - poses_in_f2: torch.Tensor of same shape as input, expressed in F2
    """
    
    def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Multiply two quaternions (qw, qx, qy, qz format)"""
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return torch.stack([w, x, y, z], dim=-1)
    
    def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
        """Conjugate of quaternion (qw, qx, qy, qz format)"""
        return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)
    
    def quat_rotate_vector(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Rotate vector v by quaternion q"""
        # Convert vector to quaternion (0, vx, vy, vz)
        v_quat = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
        
        # q * v_quat * q_conjugate
        temp = quat_multiply(q, v_quat)
        result = quat_multiply(temp, quat_conjugate(q))
        
        return result[..., 1:]  # Return only the vector part
    
    # Validate input
    if poses.shape[-1] != 7:
        raise ValueError(f"Expected poses tensor with last dimension 7, got {poses.shape[-1]}")
    
    device = poses.device
    dtype = poses.dtype
    original_shape = poses.shape
    
    # Flatten to (N, 7) for easier processing
    poses_flat = poses.view(-1, 7)
    batch_size = poses_flat.shape[0]
    
    # Default world pose
    world_pose_list = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    
    # Convert frames to tensors
    if frame_expressed_in is None:
        frame_expressed_in_tensor = torch.tensor(world_pose_list, device=device, dtype=dtype)
    else:
        frame_expressed_in_tensor = torch.tensor(frame_expressed_in, device=device, dtype=dtype)
    
    if frame_express_at is None:
        frame_express_at_tensor = torch.tensor(world_pose_list, device=device, dtype=dtype)
    else:
        frame_express_at_tensor = torch.tensor(frame_express_at, device=device, dtype=dtype)
    
    # Extract positions and quaternions
    p_rel = poses_flat[:, :3]  # (N, 3)
    q_rel = poses_flat[:, 3:]  # (N, 4)
    
    p_f1 = frame_expressed_in_tensor[:3]  # (3,)
    q_f1 = frame_expressed_in_tensor[3:]  # (4,)
    
    p_f2 = frame_express_at_tensor[:3]  # (3,)
    q_f2 = frame_express_at_tensor[3:]  # (4,)
    
    # Expand frame quaternions to match batch size
    q_f1_batch = q_f1.unsqueeze(0).expand(batch_size, -1)  # (N, 4)
    q_f2_batch = q_f2.unsqueeze(0).expand(batch_size, -1)  # (N, 4)
    
    # Transform to world frame: T_world = T_f1 * T_rel
    p_world = quat_rotate_vector(q_f1_batch, p_rel) + p_f1.unsqueeze(0)
    q_world = quat_multiply(q_f1_batch, q_rel)
    
    # Transform to F2: T_f2 = T_f2_inv * T_world
    q_f2_inv = quat_conjugate(q_f2)
    q_f2_inv_batch = q_f2_inv.unsqueeze(0).expand(batch_size, -1)  # (N, 4)
    
    p_f2_inv = -quat_rotate_vector(q_f2_inv.unsqueeze(0), p_f2.unsqueeze(0))
    
    p_in_f2 = quat_rotate_vector(q_f2_inv_batch, p_world) + p_f2_inv
    q_in_f2 = quat_multiply(q_f2_inv_batch, q_world)
    
    # Combine position and quaternion
    result = torch.cat([p_in_f2, q_in_f2], dim=-1)
    
    # Reshape back to original shape
    return result.view(original_shape)