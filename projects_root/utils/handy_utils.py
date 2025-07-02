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

# --------------------------------------------------------------
# Utility: export cuRobo WorldConfig to a mesh file
# --------------------------------------------------------------

def save_curobo_world(out_path: str, world) -> None:
    """Save a cuRobo `WorldConfig` (or list thereof) as a single OBJ mesh.

    Args:
        out_path: Destination file path (e.g. "/tmp/curobo_world.obj").
        world:    Instance of `curobo.geom.types.WorldConfig` or list of them.
                   The helper calls `save_world_as_mesh` internally.
    """

    try:
        from curobo.geom.types import WorldConfig  # lazy import to avoid heavy deps

        if isinstance(world, list):
            # merge multiple worlds into one for convenience
            merged = WorldConfig()
            for w in world:
                if isinstance(w, WorldConfig):
                    for obj in w.objects:
                        merged.add_obstacle(obj)
            world_to_save = merged
        else:
            world_to_save = world

        if hasattr(world_to_save, "save_world_as_mesh"):
            world_to_save.save_world_as_mesh(out_path)
            print(f"[CuRobo] World mesh written to {out_path}")
        else:
            print("[CuRobo] Provided object is not a WorldConfig – nothing saved")
    except Exception as e:
        print(f"[CuRobo] Failed to save world mesh: {e}")