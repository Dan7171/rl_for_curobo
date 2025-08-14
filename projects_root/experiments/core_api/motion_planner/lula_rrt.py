import numpy as np
import os

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp()

from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.api.objects.cuboid import VisualCuboid
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats, quats_to_rot_matrices
from isaacsim.core.utils.distance_metrics import rotational_distance_angle

from isaacsim.robot_motion.motion_generation import PathPlannerVisualizer
from isaacsim.robot_motion.motion_generation.lula import RRT
from isaacsim.robot_motion.motion_generation import interface_config_loader

class FrankaRrtExample():
    def __init__(self):
        self._rrt = None
        self._path_planner_visualizer = None
        self._plan = []

        self._articulation = None
        self._target = None
        self._target_position = None

        self._frame_counter = 0

    def load_example_assets(self):
        # Add the Franka and target to the stage
        # The position in which things are loaded is also the position in which they

        robot_prim_path = "/panda"
        path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"

        add_reference_to_stage(path_to_robot_usd, robot_prim_path)
        self._articulation = Articulation(robot_prim_path)

        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", "/World/target")
        self._target = XFormPrim("/World/target", scale=[.04,.04,.04])
        self._target.set_default_state(np.array([.45, .5, .7]),euler_angles_to_quats([3*np.pi/4, 0, np.pi]))

        self._obstacle = VisualCuboid("/World/Wall", position = np.array([.3,.6,.6]), size = 1.0, scale = np.array([.1,.4,.4]))

        # Return assets that were added to the stage so that they can be registered with the core.World
        return self._articulation, self._target

    def setup(self):
        # Lula config files for supported robots are stored in the motion_generation extension under
        # "/path_planner_configs" and "/motion_policy_configs"
        mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
        rmp_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")
        rrt_config_dir = os.path.join(mg_extension_path, "path_planner_configs")

        #Initialize an RRT object
        self._rrt = RRT(
            robot_description_path = rmp_config_dir + "/franka/rmpflow/robot_descriptor.yaml",
            urdf_path = rmp_config_dir + "/franka/lula_franka_gen.urdf",
            rrt_config_path = rrt_config_dir + "/franka/rrt/franka_planner_config.yaml",
            end_effector_frame_name = "right_gripper"
        )

        # RRT for supported robots can also be loaded with a simpler equivalent:
        # rrt_config = interface_config_loader.load_supported_path_planner_config("Franka", "RRT")
        # self._rrt = RRT(**rrt_confg)

        self._rrt.add_obstacle(self._obstacle)

        # Set the maximum number of iterations of RRT to prevent it from blocking Isaac Sim for
        # too long.
        self._rrt.set_max_iterations(5000)

        # Use the PathPlannerVisualizer wrapper to generate a trajectory of ArticulationActions
        self._path_planner_visualizer = PathPlannerVisualizer(self._articulation, self._rrt)

        self.reset()

    def update(self, step: float):
        current_target_translation, current_target_orientation = self._target.get_world_pose()
        current_target_rotation = quats_to_rot_matrices(current_target_orientation)

        translation_distance = np.linalg.norm(self._target_translation - current_target_translation)
        rotation_distance = rotational_distance_angle(current_target_rotation, self._target_rotation)
        target_moved = translation_distance > 0.01 or rotation_distance > 0.01

        if (self._frame_counter % 60 == 0 and target_moved):
            # Replan every 60 frames if the target has moved
            self._rrt.set_end_effector_target(current_target_translation, current_target_orientation)
            self._rrt.update_world()
            self._plan = self._path_planner_visualizer.compute_plan_as_articulation_actions(max_cspace_dist=.01)

            self._target_translation = current_target_translation
            self._target_rotation = current_target_rotation

        if self._plan:
            action = self._plan.pop(0)
            self._articulation.apply_action(action)

        self._frame_counter += 1

    def reset(self):
        self._target_translation = np.zeros(3)
        self._target_rotation = np.eye(3)
        self._frame_counter = 0
        self._plan = []