"""
https://docs.isaacsim.omniverse.nvidia.com/4.0.0/core_api_tutorials/tutorial_core_hello_robot.html"""

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
while simulation_app.is_running():
    world.step(render=True) # execute one physics step and one rendering step
    
    # In Omniverse Isaac Sim, Robots are constructed of physically accurate articulated joints.
    # Applying actions to these articulations make them move.
    # Next, apply random velocities to the Jetbot articulation controller to get it moving.
    # Every articulation controller has apply_action method
    # which takes in ArticulationAction with joint_positions, joint_efforts and joint_velocities
    # as optional args. It accepts numpy arrays of floats OR lists of floats and None
    # None means that nothing is applied to this dof index in this step
    # ALTERNATIVELY, same method is called from self._jetbot.apply_action(...)

    # Both next 2 forms are ok. 
    # This is the more general but less simple form:    
    # jetbot_robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=None,
    #                                                                         joint_efforts=None,
    #                                                                         joint_velocities=5 * np.random.rand(2,)))

    # This is the simpler form for wheeled robots:
    # Omniverse Isaac Sim also has robot-specific extensions that provide further customized functions and access to other controllers and tasks (more on this later). Now you’ll re-write the previous code using the WheeledRobot class to make it simpler.
    jetbot_robot.apply_wheel_actions(ArticulationAction(joint_positions=None,joint_efforts=None,joint_velocities=5 * np.random.rand(2,)))
    
simulation_app.close() # close Isaac Sim