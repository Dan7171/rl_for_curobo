"""
Reference: https://docs.isaacsim.omniverse.nvidia.com/4.0.0/core_api_tutorials/tutorial_core_adding_manipulator.html#use-the-pick-and-place-task
* This example  is similar to the previous one (manipulator_with_custom_task.py) but this time we use the built in pick and place task isaac sim provides (see reference above).



* Its not yet implemented in the standalone app. The its easy to adjust it to be a standalone app as well
As all the examples are...
The adjustment should be done very similar to the adjustment of manipulator_with_custom_task.py example (a standalone app) which inspired by this example.
To do this, First see how the manipulator_with_custom_task.py example is adjusted to a standalone app, from this non-standalone example: https://docs.isaacsim.omniverse.nvidia.com/4.0.0/core_api_tutorials/tutorial_core_adding_manipulator.html#:~:text=return-,What%20is%20a%20Task%3F,%EF%83%81,-The%20Task%20class.
And repeat it for this example, using the same logic, from the reference above.
You can take the manipulator_with_custom_task.py example as a template and repeat the same steps for this example.

 
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
# from omni.isaac.franka.tasks import PickPlace # NOTE here is the change from the previous example

# isaac 4.5 https://docs.isaacsim.omniverse.nvidia.com/latest/overview/extensions_renaming.html
from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController # omni.isaac.franka.controllers import PickPlaceController
from isaacsim.robot.manipulators.examples.franka.tasks import PickPlace 

import numpy as np

##################  Implementing a built in task for pick and place   #########################
# 


####################### Setup world ##########################

world = World()
world.scene.add_default_ground_plane()
###################### Add task to the world #######################

 # We add the task to the world here
world.add_task(PickPlace(name="awesome_task"))
world.reset() # to call the task setup_scene,world.reset() (it replaces the call to await self._world.play_async() from the original example, as we are not using the simulation app but in a standalone app.)
 

###################### Add robot and a dynamic cube #######################
# The world already called the setup_scene from the task (with first reset of the world)
# so we can retrieve the task objects
franka = world.scene.get_object("fancy_franka") 
controller = PickPlaceController(  # NOTE here is the change from the previous example
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