"""
https://docs.isaacsim.omniverse.nvidia.com/4.0.0/core_api_tutorials/tutorial_core_adding_controller.html#using-the-available-controllers

This example shows how to use the available controllers in the wheeled_robots extension.
Controller gets as a command the goal position and the current position and orientation of the robot (the final goal position)).
We drive the robot to a goal position and then rotate 180 degrees to the opposite direction.
For info about the controllers see:
https://docs.isaacsim.omniverse.nvidia.com/4.0.0/py/source/extensions/omni.isaac.wheeled_robots/docs/index.html?highlight=wheelbaseposecontroller


"""
############## STANDARD INIIATION IN ALL STGANDALONE PY APPS - DO NOT MODIFY ##############
import time
import isaacsim
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})   # https://docs.omniverse.nvidia.com/isaacsim/latest/manual_standalone_python.html#simulationapp

######################## IMPORTS ###############################
from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.core.utils.types import ArticulationAction
# This extension includes several generic controllers that could be used with multiple robots

# WheelBasePoseController: This controller closes the control loop, returning the wheel commands that will
#  drive the robot to a desired pose. It does this by exploiting an open loop controller for the robot passed at class initialization.
from omni.isaac.wheeled_robots.controllers.wheel_base_pose_controller import WheelBasePoseController # 
# Robot specific controller
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
import numpy as np
import carb


####################### Setup world ##########################

world = World()
world.scene.add_default_ground_plane()

################## Setup the path to the assets #################
assets_root_path = get_assets_root_path()
if assets_root_path is None:
# Use carb to log warnings, errors and infos in your application (shown on terminal)
    carb.log_error("Could not find nucleus server with /Isaac folder")

###################### Add robot #######################
asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
# This will create a new XFormPrim and point it to the usd file as a reference
# Similar to how pointers work in memory
add_reference_to_stage(usd_path=asset_path, prim_path="/World/Fancy_Robot")
# Wrap the jetbot prim root under a Robot class and add it to the Scene
# to use high level api to set/ get attributes as well as initializing
# physics handles needed..etc.
# Note: this call doesn't create the Jetbot in the stage window, it was already
# created with the add_reference_to_stage
jetbot_robot = world.scene.add(WheeledRobot(
                prim_path="/World/Fancy_Robot",
                name="fancy_robot",
                wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                create_robot=True,
                usd_path=asset_path,
            ))

# Note: before a reset is called, we can't access information related to an Articulation
# because physics handles are not initialized yet. setup_post_load is called after
# the first reset so we can do so there
print("Num of degrees of freedom before first reset: " + str(jetbot_robot.num_dof)) # prints None

# Resetting the world needs to be called before querying anything related to an articulation specifically.
# Its recommended to always do a reset after adding your assets, for physics handles to be propagated properly
world.reset()
print("Num of degrees of freedom after first reset: " + str(jetbot_robot.num_dof)) # prints 2
print("Joint Positions after first reset: " + str(jetbot_robot.get_joint_positions()))

######################## Main Loop - moving the robot #########################
goal_position = np.array([0.8, 0.8])
# https://docs.isaacsim.omniverse.nvidia.com/4.0.0/py/source/extensions/omni.isaac.wheeled_robots/docs/index.html?highlight=wheelbaseposecontroller#:~:text=Wheel%20Base%20Pose,%EF%83%81
# WheelBasePoseController: This controller closes the control loop, returning the wheel commands that will drive the robot to a desired pose. It does this by exploiting an open loop controller for the robot passed at class initialization.
controller = WheelBasePoseController(name="cool_controller", \
                                    open_loop_wheel_controller=\
                                        DifferentialController(\
                                            name="simple_control",wheel_radius=0.03, wheel_base=0.1125),\
                                                is_holonomic=False)
while simulation_app.is_running():
    world.step(render=True) # execute one physics step and one rendering step
    # In this example we control the robot by setting the goal position to the controller
    position, orientation = jetbot_robot.get_world_pose()
    x,y,_ = position # z is irrelevant for this example
    jetbot_robot.apply_action(controller.forward(start_position=position,\
                                            start_orientation=orientation,
                                            goal_position=goal_position))  # move to goal
    print(position, orientation, goal_position)
    if np.linalg.norm([x, y] - goal_position) < 0.1: # check if goal is reached
        print("Goal reached, rotating 180 degrees")
        time.sleep(3)
        goal_position = - goal_position # rotate 180 degrees
simulation_app.close() # close Isaac Sim