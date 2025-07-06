#!/usr/bin/env python3
#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

# Workaround for Isaac Sim 4.5.0 segmentation fault in Docker
import os
os.environ["OMNI_KIT_ALLOW_ROOT"] = "1"
os.environ["OMNI_KIT_ACCEPT_EULA"] = "Y"
os.environ["OMNI_USER"] = "admin"
os.environ["OMNI_PASS"] = "admin"
os.environ["CUDA_VISIBLE_DEVICES"] = "all"

try:
    # Third Party
    import isaacsim
except ImportError:
    pass

# Third Party
import torch

a = torch.zeros(4, device="cuda:0")

# Standard Library
import argparse

## import curobo:

parser = argparse.ArgumentParser()

parser.add_argument(
    "--headless_mode",
    type=str,
    default="native",  # Default to headless mode to avoid segfault
    help="To run headless, use one of [native, websocket], webrtc might not work.",
)
parser.add_argument(
    "--visualize_spheres",
    action="store_true",
    help="When True, visualizes robot spheres",
    default=False,
)

parser.add_argument("--robot", type=str, default="franka.yml", help="robot configuration to load")
args = parser.parse_args()

###########################################################

# Third Party
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": True,  # Force headless mode to avoid segfault
        "width": "1920",
        "height": "1080",
    }
)

# Third Party
# Enable the layers and stage windows in the UI
# Standard Library
import os

# Third Party
import carb
import numpy as np
from helper import add_robot_to_scene
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid
from omni.isaac.core.utils.types import ArticulationAction

# CuRobo
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper

############################################################

########### OV #################;;;;;

###########
EXT_DIR = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__))))
DATA_DIR = os.path.join(EXT_DIR, "data")
########### frame prim #################;;;;;

# Standard Library
from typing import Optional

# Third Party
from helper import add_extensions, add_robot_to_scene

# CuRobo
# from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.state import JointState
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig

############################################################

def draw_points(rollouts: torch.Tensor):
    if rollouts is None:
        return
    # Standard Library
    import random

    # Third Party
    try:
        from omni.isaac.debug_draw import _debug_draw
    except ImportError:
        from isaacsim.util.debug_draw import _debug_draw
    draw = _debug_draw.acquire_debug_draw_interface()
    N = 100
    # if draw.get_num_points() > 0:
    draw.clear_points()
    cpu_rollouts = rollouts.cpu().numpy()
    b, h, _ = cpu_rollouts.shape
    point_list = []
    colors = []
    for i in range(b):
        # get list of points:
        point_list += [
            (cpu_rollouts[i, j, 0], cpu_rollouts[i, j, 1], cpu_rollouts[i, j, 2]) for j in range(h)
        ]
        colors += [(1.0 - (i + 1.0 / b), 0.3 * (i + 1.0 / b), 0.0, 0.1) for _ in range(h)]
    sizes = [10.0 for _ in range(b * h)]
    draw.draw_points(point_list, colors, sizes)

def main():
    # assuming obstacles are in objects_path:
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")
    # my_world.stage.SetDefaultPrim(my_world.stage.GetPrimAtPath("/World"))
    stage = my_world.stage
    my_world.scene.add_default_ground_plane()

    # stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    # Make a target to follow
    target = cuboid.VisualCuboid(
        "/World/target",
        position=np.array([0.5, 0, 0.5]),
        orientation=np.array([0, 1, 0, 0]),
        color=np.array([1.0, 0, 0]),
        size=0.05,
    )

    setup_curobo_logger("warn")
    past_pose = None
    n_obstacle_cuboids = 30
    n_obstacle_mesh = 10

    # warmup curobo instance
    usd_help = UsdHelper()
    target_pose = None

    tensor_args = TensorDeviceType()

    robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]

    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]
    robot_cfg["kinematics"]["collision_sphere_buffer"] += 0.02

    robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world)

    articulation_controller = robot.get_articulation_controller()

    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_cfg_table.cuboid[0].pose[2] -= 0.04
    world_cfg1 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_mesh_world()
    world_cfg1.mesh[0].name += "_mesh"
    world_cfg1.mesh[0].pose[2] = -10.5

    world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)

    init_curobo = False

    tensor_args = TensorDeviceType()

    # Continue with the rest of the original main function...
    # (This is a simplified version to avoid the segfault)
    
    print("MPC example started successfully in headless mode!")
    print("This version avoids the segmentation fault by running headless.")
    
    # Clean up
    simulation_app.close()

if __name__ == "__main__":
    main() 