"""
https://docs.isaacsim.omniverse.nvidia.com/4.0.0/core_api_tutorials/tutorial_core_adding_manipulator.html#:~:text=return-,Using%20the%20PickAndPlace%20Controller,%EF%83%81,-Add%20a%20pickBasic pick and place example
Setting a cube, Navigate to the cube, pick it up, and place it in the goal position.

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
import numpy as np

####################### Setup world ##########################

world = World()
world.scene.add_default_ground_plane()

###################### Add robot and a dynamic cube #######################

# Robot specific class that provides extra functionalities
# such as having gripper and end_effector instances.
franka = world.scene.add(Franka(prim_path="/World/Fancy_Franka", name="fancy_franka"))
controller = PickPlaceController( # https://docs.isaacsim.omniverse.nvidia.com/4.0.0/py/source/extensions/omni.isaac.franka/docs/index.html?highlight=pickplacecontroller#omni.isaac.franka.controllers.PickPlaceController:~:text=end%20effector%20frame.-,Franka%20Controllers,%EF%83%81,-class
            name="pick_place_controller",
            gripper=franka.gripper,
            robot_articulation=franka,
        )
# Resetting the world needs to be called before querying anything related to an articulation specifically.
world.reset()
# open ee fingers
franka.gripper.set_joint_positions(franka.gripper.joint_opened_positions) # NOTE :reset world before calling that.
# add a cube for franka to pick up
world.scene.add(
    DynamicCuboid(
        prim_path="/World/random_cube",
        name="fancy_cube",
        position=np.array([0.3, 0.3, 0.3]),
        scale=np.array([0.0515, 0.0515, 0.0515]),
        color=np.array([0, 0, 1.0]),
    )
)



######################## Main Loop #########################
# Navigate to the cube, pick it up, and place it in the goal position.
    
goal_position = np.array([-0.3, -0.3, 0.0515 / 2.0])
while simulation_app.is_running():
    world.step(render=True)
    cube_position, cube_rotation = world.scene.get_object("fancy_cube").get_world_pose()
    actions = controller.forward(
            picking_position=cube_position,
            placing_position=goal_position,
            current_joint_positions=franka.get_joint_positions(),
        )
    franka.apply_action(actions)
    print(f"ts: {world.current_time_step_index}")
    if controller.is_done(): # Only for the pick and place controller, indicating if the state
    # machine reached the final state.
        world.pause()
simulation_app.close()