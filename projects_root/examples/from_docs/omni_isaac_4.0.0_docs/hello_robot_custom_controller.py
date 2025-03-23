"""
https://docs.isaacsim.omniverse.nvidia.com/4.0.0/core_api_tutorials/tutorial_core_adding_controller.html
"""
############## STANDARD INIIATION IN ALL STGANDALONE PY APPS - DO NOT MODIFY ##############
import isaacsim
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})   # https://docs.omniverse.nvidia.com/isaacsim/latest/manual_standalone_python.html#simulationapp

######################## IMPORTS ###############################
from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.robots import Robot
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.controllers import BaseController
import numpy as np
import carb

###################### IMLEMENT CUSTOM CONTROLLER #######################
class CoolController(BaseController):
    def __init__(self):
        super().__init__(name="my_cool_controller")
        # An open loop controller that uses a unicycle model
        self._wheel_radius = 0.03
        self._wheel_base = 0.1125
        return

    def forward(self, command):
        # command will have two elements, first element is the forward velocity
        # second element is the angular velocity (yaw only).
        joint_velocities = [0.0, 0.0]
        joint_velocities[0] = ((2 * command[0]) - (command[1] * self._wheel_base)) / (2 * self._wheel_radius)
        joint_velocities[1] = ((2 * command[0]) + (command[1] * self._wheel_base)) / (2 * self._wheel_radius)
        # A controller has to return an ArticulationAction
        return ArticulationAction(joint_velocities=joint_velocities)

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
controller = CoolController()
while simulation_app.is_running():
    world.step(render=True) # execute one physics step and one rendering step
    #apply the actions calculated by the controller
    jetbot_robot.apply_action(controller.forward(command=[0.20, np.pi / 4])) # moving in circle
simulation_app.close() # close Isaac Sim