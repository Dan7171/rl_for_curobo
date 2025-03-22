# Third Party
import numpy as np
import torch
from matplotlib import cm
# Replace RealSense import with our simulated camera
from simulated_camera import SimulatedCamera

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType

if __name__ == "__main__":
    radius = 0.05
    act_distance = 0.4
    voxel_size = 0.05
    render_voxel_size = 0.02
    clipping_distance = 0.7

    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage

    stage = my_world.stage
    my_world.scene.add_default_ground_plane()

    # Replace RealSense initialization with simulated camera
    realsense_data = SimulatedCamera(clipping_distance_m=clipping_distance)
    data = realsense_data.get_data() 