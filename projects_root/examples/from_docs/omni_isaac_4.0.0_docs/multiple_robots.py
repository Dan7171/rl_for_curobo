"""

https://docs.isaacsim.omniverse.nvidia.com/4.0.0/core_api_tutorials/tutorial_core_adding_multiple_robots.html

This tutorial integrates two different types of robot into the same simulation. It details how to build program logic to switch between subtasks. 
After this tutorial, you will have experience building more complex simulations of robots interacting.

"""


"""
https://docs.isaacsim.omniverse.nvidia.com/4.0.0/core_api_tutorials/tutorial_core_adding_manipulator.html#what-is-a-task

Similar to the previous example (manpulilator.py, Setting a cube, Navigate to the cube, pick it up, and place it in the goal position.) 
but this time we implement a custom task and introduce the concept of tasks for the first time.
"""

############## STANDARD INIIATION IN ALL STGANDALONE PY APPS - DO NOT MODIFY ##############
import time
import isaacsim
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})   # https://docs.omniverse.nvidia.com/isaacsim/latest/manual_standalone_python.html#simulationapp

####################### IMPORTS ###################################################3
from omni.isaac.core import World
# This extension has franka related tasks and controllers as well
from omni.isaac.franka import Franka
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.franka.controllers import PickPlaceController
from omni.isaac.core.tasks import BaseTask
import numpy as np

##################  Implementing a custom task for pick and place (from BaseTask). Note: there is a built in PickPlace task in Isaac Sim. #########################
class FrankaPlaying(BaseTask):
    # NOTE: we only cover here a subset of the task functions that are available,
    # checkout the base class for all the available functions to override.
    # ex: calculate_metrics, is_done..etc.
    def __init__(self, name):
        super().__init__(name=name, offset=None)
        self._goal_position = np.array([-0.3, -0.3, 0.0515 / 2.0])
        self._task_achieved = False
        return

    # Here we setup all the assets that we care about in this task.
    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        scene.add_default_ground_plane()
        self._cube = scene.add(DynamicCuboid(prim_path="/World/random_cube",
                                            name="fancy_cube",
                                            position=np.array([0.3, 0.3, 0.3]),
                                            scale=np.array([0.0515, 0.0515, 0.0515]),
                                            color=np.array([0, 0, 1.0])))
        self._franka = scene.add(Franka(prim_path="/World/Fancy_Franka",
                                        name="fancy_franka"))
        return

    # Information exposed to solve the task is returned from the task through get_observations
    def get_observations(self):
        cube_position, _ = self._cube.get_world_pose()
        current_joint_positions = self._franka.get_joint_positions()
        observations = {
            self._franka.name: {
                "joint_positions": current_joint_positions,
            },
            self._cube.name: {
                "position": cube_position,
                "goal_position": self._goal_position
            }
        }
        return observations

    # Called before each physics step,
    # for instance we can check here if the task was accomplished by
    # changing the color of the cube once its accomplished
    def pre_step(self, control_index, simulation_time):
        cube_position, _ = self._cube.get_world_pose()
        if not self._task_achieved and np.mean(np.abs(self._goal_position - cube_position)) < 0.02:
            # Visual Materials are applied by default to the cube
            # in this case the cube has a visual material of type
            # PreviewSurface, we can set its color once the target is reached.
            self._cube.get_applied_visual_material().set_color(color=np.array([0, 1.0, 0]))
            self._task_achieved = True
        return

    # Called after each reset,
    # for instance we can always set the gripper to be opened at the beginning after each reset
    # also we can set the cube's color to be blue
    def post_reset(self):
        self._franka.gripper.set_joint_positions(self._franka.gripper.joint_opened_positions)
        self._cube.get_applied_visual_material().set_color(color=np.array([0, 0, 1.0]))
        self._task_achieved = False
        return


####################### Setup world ##########################

world = World()
world.scene.add_default_ground_plane()
###################### Add task to the world #######################

 # We add the task to the world here
world.add_task(FrankaPlaying(name="my_first_task"))
world.reset() # to call the task setup_scene,world.reset() (it replaces the call to await self._world.play_async() from the original example, as we are not using the simulation app but in a standalone app.)
 

###################### Add robot and a dynamic cube #######################
# The world already called the setup_scene from the task (with first reset of the world)
# so we can retrieve the task objects
franka = world.scene.get_object("fancy_franka") 
controller = PickPlaceController( # https://docs.isaacsim.omniverse.nvidia.com/4.0.0/py/source/extensions/omni.isaac.franka/docs/index.html?highlight=pickplacecontroller#omni.isaac.franka.controllers.PickPlaceController:~:text=end%20effector%20frame.-,Franka%20Controllers,%EF%83%81,-class
            name="pick_place_controller",
            gripper=franka.gripper,
            robot_articulation=franka,
        )
# Resetting the world needs to be called before querying anything related to an articulation specifically.
world.reset()
# open ee fingers
# franka.gripper.set_joint_positions(franka.gripper.joint_opened_positions) # NOTE :reset world before calling that.



######################## Main Loop #########################
# Navigate to the cube, pick it up, and place it in the goal position.
    
while simulation_app.is_running():
    world.step(render=True)
     # Gets all the tasks observations
    current_observations = world.get_observations() # task is added to the world, so we can get the observations from the world
    actions = controller.forward( 
        picking_position=current_observations["fancy_cube"]["position"],
        placing_position=current_observations["fancy_cube"]["goal_position"],
        current_joint_positions=current_observations["fancy_franka"]["joint_positions"],
    )
    franka.apply_action(actions)
    print(f"ts: {world.current_time_step_index}")
    if controller.is_done(): # Only for the pick and place controller, indicating if the state
    # machine reached the final state. (goal state)
        world.pause()
simulation_app.close()