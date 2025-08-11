
try:
    # Third Party
    import isaacsim
except ImportError:
    pass

from omni.isaac.kit import SimulationApp

CONFIG = {
    "width": 1280,
    "height": 720,
    "window_width": 1920,
    "window_height": 1080,
    "headless": True,
    "hide_ui": False,  # Show the GUI
    "renderer": "RaytracedLighting",
    "display_options": 3286,  # Set display options to show default grid
}

# Start the omniverse application
simulation_app = SimulationApp(launch_config=CONFIG)
from isaacsim.core.utils.extensions import enable_extension
# Default Livestream settings
simulation_app.set_setting("/app/window/drawMouse", True)
# Enable Livestream extension
enable_extension("omni.kit.livestream.webrtc")


# from projects_root.utils.helper import add_extensions
import os

from abc import abstractmethod
from collections.abc import Callable
from copy import copy, deepcopy
import dataclasses
import os
from time import time, sleep
from threading import Lock, Event, Thread
from typing import Optional, Tuple, Dict, Union, Callable
from typing_extensions import List
import pickle
import torch
# import pandas as pd
import random
from datetime import datetime

from scipy.spatial.transform import Rotation as R
import carb
import numpy as np
import signal
import sys
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.types import JointsState as isaac_JointsState
from isaacsim.core.utils.xforms import get_world_pose
import omni
from pxr import UsdGeom, Gf, Sdf, UsdPhysics
from omni.isaac.core.objects import DynamicCuboid, VisualCuboid, VisualSphere, DynamicSphere, FixedCuboid, FixedSphere
from omni.isaac.core.utils.viewports import set_camera_view


# CuRobo
from curobo.types.tensor import T_DOF
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig, Sphere
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.types.state import JointState, FilterCoeff
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from projects_root.utils.helper import add_robot_to_scene
from projects_root.projects.dynamic_obs.dynamic_obs_predictor.runtime_topics import init_runtime_topics, get_topics
from projects_root.projects.dynamic_obs.dynamic_obs_predictor.dynamic_obs_coll_checker import DynamicObsCollPredictor
from curobo.rollout.cost.custom.arm_base.dynamic_obs_cost import DynamicObsCost
from curobo.opt.particle.parallel_mppi import ParallelMPPI
from curobo.rollout.arm_reacher import ArmReacher
from curobo.wrap.wrap_mpc import WrapMpc
from projects_root.projects.dynamic_obs.dynamic_obs_predictor.frame_utils import FrameUtils
from projects_root.utils.world_model_wrapper import WorldModelWrapper
from projects_root.utils.usd_pose_helper import get_stage_poses, list_relevant_prims
from projects_root.utils.transforms import transform_poses_batched_optimized_for_spheres, transform_poses_batched
from projects_root.utils.draw import draw_points
from projects_root.utils.colors import npColors
from projects_root.utils.issacsim import  activate_gpu_dynamics



                

def main():
    
    my_world = World(stage_units_in_meters=1.0)
    activate_gpu_dynamics(my_world)
    my_world.scene.add_default_ground_plane()
    stage = my_world.stage
    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")
    usd_help = UsdHelper()
    usd_help.load_stage(my_world.stage)

    
    my_world.reset()


 
    while simulation_app.is_running():
        
        my_world.step(render=True)
        if not my_world.is_playing():
            if i % 100 == 0:
                print("**** Click Play to start simulation *****")
            i += 1
            continue
            
        if stop_simulation:
            break
        
    simulation_app.close()
    
# Global flag to track if we should stop
stop_simulation = False
def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global stop_simulation
    print(f"\nReceived signal {signum} - shutting down gracefully...")
    stop_simulation = True
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


main()