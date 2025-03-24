"""

https://docs.isaacsim.omniverse.nvidia.com/4.0.0/core_api_tutorials/tutorial_core_adding_multiple_robots.html#robot-handover
Here I implemented the part under "Robot Handover" section.
But read the whole tutorial at this link for more details.
Basically, 2 robots are added to the world, and the jetbot is tasked with pushing the cube to the franka.

"""

############## STANDARD INIIATION IN ALL STGANDALONE PY APPS - DO NOT MODIFY ##############
import time
import isaacsim
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})   # https://docs.omniverse.nvidia.com/isaacsim/latest/manual_standalone_python.html#simulationapp

####################### IMPORTS ###################################################3
from omni.isaac.core import World
from omni.isaac.franka.tasks import PickPlace
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.tasks import BaseTask
from omni.isaac.wheeled_robots.controllers.wheel_base_pose_controller import WheelBasePoseController
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.franka.controllers import PickPlaceController
import numpy as np

class RobotsPlaying(BaseTask):
    def __init__(
        self,
        name
    ):
        super().__init__(name=name, offset=None)
        self._jetbot_goal_position = np.array([1.3, 0.3, 0]) # self._jetbot_goal_position = np.array([130, 30, 0])
        # Add a subtask of pick and place instead
        # of writing the same task again
        # we just need to add a jetbot and change the positions of the assets and
        # the cube target position
        # Add task logic to signal to the robots which task is active
        self._task_event = 0
        self._pick_place_task = PickPlace(cube_initial_position=np.array([0.1, 0.3, 0.05]),
                                        target_position=np.array([0.7, -0.3, 0.0515 / 2.0]))
        
        
        return

    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        self._pick_place_task.set_up_scene(scene)
        assets_root_path = get_assets_root_path()
        jetbot_asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
        self._jetbot = scene.add(
            WheeledRobot(
                prim_path="/World/Fancy_Robot",
                name="fancy_robot",
                wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                create_robot=True,
                usd_path=jetbot_asset_path,
                position=np.array([0, 0.3, 0]))
        )
        pick_place_params = self._pick_place_task.get_params()
        self._franka = scene.get_object(pick_place_params["robot_name"]["value"])
        # Changes Franka's default position
        # so that it is set at this position after reset
        self._franka.set_world_pose(position=np.array([1.0, 0, 0]))
        self._franka.set_default_state(position=np.array([1.0, 0, 0]))
        return

    def get_observations(self):
        current_jetbot_position, current_jetbot_orientation = self._jetbot.get_world_pose()
        # Observations needed to drive the jetbot to push the cube
        observations= {
            "task_event": self._task_event,
            self._jetbot.name: {
                "position": current_jetbot_position,
                "orientation": current_jetbot_orientation,
                "goal_position": self._jetbot_goal_position
            }
        }
        # add the subtask's observations as well
        observations.update(self._pick_place_task.get_observations()) # update the dict (from jetson task) with the the pick place task's observations
        return observations

    def get_params(self):
        # To avoid hard coding names..etc.
        pick_place_params = self._pick_place_task.get_params()
        params_representation = pick_place_params
        params_representation["jetbot_name"] = {"value": self._jetbot.name, "modifiable": False}
        params_representation["franka_name"] = pick_place_params["robot_name"]
        return params_representation

    def pre_step(self, control_index, simulation_time):
        if self._task_event == 0:
            current_jetbot_position, _ = self._jetbot.get_world_pose()
            if np.mean(np.abs(current_jetbot_position[:2] - self._jetbot_goal_position[:2])) < 0.04:
                self._task_event += 1
                self._cube_arrive_step_index = control_index
        elif self._task_event == 1:
            # Jetbot has 200 time steps to back off from Franka
            if control_index - self._cube_arrive_step_index == 200:
                self._task_event += 1
        return
    def post_reset(self):
        self._franka.gripper.set_joint_positions(self._franka.gripper.joint_opened_positions)
        self._task_event = 0




####################### Setup world ##########################

world = World()
world.scene.add_default_ground_plane()
###################### Add task to the world #######################

 # We add the task to the world here
world.add_task(RobotsPlaying(name="awesome_task"))
world.reset() # to call the task setup_scene,world.reset() (it replaces the call to await self._world.play_async() from the original example, as we are not using the simulation app but in a standalone app.)
 
#################### SETTING UP THE JETBOT #################### 
task_params = world.get_task("awesome_task").get_params()
jetbot = world.scene.get_object(task_params["jetbot_name"]["value"])
cube_name = task_params["cube_name"]["value"]
jetbot_controller = WheelBasePoseController(name="cool_controller",
                                                open_loop_wheel_controller=
                                                    DifferentialController(name="simple_control",
                                                                        wheel_radius=0.03, wheel_base=0.1125))

#################### SETTING UP THE FRANKA #################### 
# We need franka later to apply to it actions
franka = world.scene.get_object(task_params["franka_name"]["value"])
# We need the cube later on for the pick place controller
# Add Franka Controller
franka_controller = PickPlaceController(name="pick_place_controller",
                                            gripper=franka.gripper,
                                            robot_articulation=franka)

# Reset the world
world.reset()
jetbot_controller.reset() # here beucase it was in setup_post_reset in original example
franka_controller.reset() # here beucase it was in setup_post_reset in original example

######################## Main Loop #########################
# Navigate to the cube, pick it up, and place it in the goal position.
    
while simulation_app.is_running():
    world.step(render=True)
    current_observations = world.get_observations()
    if current_observations["task_event"] == 0:
        jetbot.apply_action(
        jetbot_controller.forward(
            start_position=current_observations[jetbot.name]["position"],
            start_orientation=current_observations[jetbot.name]["orientation"],
            goal_position=current_observations[jetbot.name]["goal_position"]))

    elif current_observations["task_event"] == 1:
        # Go backwards
        jetbot.apply_wheel_actions(ArticulationAction(joint_velocities=[-8, -8]))
    elif current_observations["task_event"] == 2:
        # Apply zero velocity to override the velocity applied before.
        # Note: target joint positions and target joint velocities will stay active unless changed
        jetbot.apply_wheel_actions(ArticulationAction(joint_velocities=[0.0, 0.0]))
        # Pick up the block
        actions = franka_controller.forward(
            picking_position=current_observations[cube_name]["position"],
            placing_position=current_observations[cube_name]["target_position"],
            current_joint_positions=current_observations[franka.name]["joint_positions"])
    # Pause once the controller is done
    if franka_controller.is_done():
        world.pause()
simulation_app.close()