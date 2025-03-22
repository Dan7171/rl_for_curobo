import numpy as np
import torch
from omni.isaac.sensor import Camera
import omni.isaac.core.utils.numpy.rotations as rot_utils

class SimulatedCamera:
    def __init__(self, clipping_distance_m=1.0):
        self.clipping_distance = clipping_distance_m
        self.camera = Camera(
            prim_path="/World/camera",
            position=np.array([-0.05, 0.0, 0.45]),  # Match the camera_marker position
            orientation=rot_utils.euler_angles_to_quats(np.array([0, -90, 0]), degrees=True),
            frequency=20,
            resolution=(640, 480),  # Standard RealSense resolution
        )
        self.camera.initialize()
        
        # RealSense D435 intrinsics (from RealSense documentation)
        self.intrinsics = np.array([
            [612.4178466796875, 0.0, 309.72296142578125],
            [0.0, 612.362060546875, 245.35870361328125],
            [0.0, 0.0, 1.0]
        ])

    def get_data(self):
        # Get raw depth data from camera
        depth_data = self.camera.get_depth_data()
        
        # Clip depth data based on clipping distance
        depth_data = np.clip(depth_data, 0, self.clipping_distance)
        
        # Convert to torch tensor and reshape to match expected format
        depth_tensor = torch.from_numpy(depth_data).cuda().float()
        
        # Create data dictionary matching RealSense format
        data = {
            "depth": depth_tensor,
            "raw_depth": depth_data,
            "intrinsics": self.intrinsics,
        }
        
        return data

    def stop_device(self):
        pass  # No need to stop a simulated device 