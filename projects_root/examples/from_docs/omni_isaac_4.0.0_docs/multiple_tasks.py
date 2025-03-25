"""
See: https://docs.isaacsim.omniverse.nvidia.com/4.0.0/core_api_tutorials/tutorial_core_multiple_tasks.html#scaling-to-many-tasks
In this example, 3 tasks are added to the world.
Each task has its own jetbot and franka.
The tasks are added to the world in the scene_setup method.
The setup_post_load method defines the controllers for each task.
The setup_post_reset method resets the controllers for each task.
The physics_step method applies the actions for each task.
The tasks are seperated by the offset parameter in the RobotsPlaying class.
In the main loop, the physics_step method is called for each task by it's turn.
"""



############## STANDARD INIIATION IN ALL STGANDALONE PY APPS - DO NOT MODIFY ##############
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
# Find a unique string name, to use it for prim paths and scene names
from omni.isaac.core.utils.string import find_unique_string_name        # Creates a unique prim path
from omni.isaac.core.utils.prims import is_prim_path_valid              # Checks if a prim path is valid
from omni.isaac.core.objects.cuboid import VisualCuboid

import numpy as np

class RobotsPlaying(BaseTask):
    # Adding offset to move the task objects with and the targets..etc
    def __init__(self, name, offset=None):
        super().__init__(name=name, offset=offset)
        self._task_event = 0
        self._jetbot_goal_position = np.array([1.3, 0.3, 0]) + self._offset #defined in the base task
        # Randomize the task a bit
        self._jetbot_goal_position = np.array([np.random.uniform(1.2, 1.6), 0.3, 0]) + self._offset
        self._pick_place_task = PickPlace(cube_initial_position=np.array([0.1, 0.3, 0.05]),
                                        target_position=np.array([0.7, -0.3, 0.0515 / 2.0]),
                                        offset=offset)
        return

    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        #This will already translate the pick and place assets by the offset
        self._pick_place_task.set_up_scene(scene)
        # Find a unique scene name
        jetbot_name = find_unique_string_name(
            initial_name="fancy_jetbot", is_unique_fn=lambda x: not self.scene.object_exists(x)
        )
        # Find a unique prim path
        jetbot_prim_path = find_unique_string_name(
            initial_name="/World/Fancy_Jetbot", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        assets_root_path = get_assets_root_path()
        jetbot_asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
        self._jetbot = scene.add(
            WheeledRobot(
                prim_path=jetbot_prim_path,
                name=jetbot_name,
                wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                create_robot=True,
                usd_path=jetbot_asset_path,
                position=np.array([0, 0.3, 0]),
            )
        )
        # Add Jetbot to this task objects
        self._task_objects[self._jetbot.name] = self._jetbot
        pick_place_params = self._pick_place_task.get_params()
        self._franka = scene.get_object(pick_place_params["robot_name"]["value"])
        # translate the franka by 100 in the x direction
        current_position, _ = self._franka.get_world_pose()
        self._franka.set_world_pose(position=current_position + np.array([1.0, 0, 0]))
        self._franka.set_default_state(position=current_position + np.array([1.0, 0, 0]))
        # This will only translate the task_objects by the offset specified (defined in the BaseTask)
        # Note: PickPlace task objects were already translated when setting up its scene
        self._move_task_objects_to_their_frame()
        return

    def get_observations(self):
        current_jetbot_position, current_jetbot_orientation = self._jetbot.get_world_pose()
        observations= {
            self.name + "_event": self._task_event, #change task event to make it unique
            # "task_event": self._task_event,
            self._jetbot.name: {
                "position": current_jetbot_position,
                "orientation": current_jetbot_orientation,
                "goal_position": self._jetbot_goal_position
            }
        }
        observations.update(self._pick_place_task.get_observations())
        return observations

    def get_params(self):
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
            if control_index - self._cube_arrive_step_index == 200:
                self._task_event += 1
        return

    def post_reset(self):
        self._franka.gripper.set_joint_positions(self._franka.gripper.joint_opened_positions)
        self._task_event = 0
        return


class HelloWorld:
    def __init__(self) -> None:
        world = World()
        world.scene.add_default_ground_plane()
        # Add lists for tasks,
        self._tasks = []
        self._num_of_tasks = 3
        #  Add lists for controllers
        self._franka_controllers = []
        self._jetbot_controllers = []
        # Add lists for variables needed for control
        self._jetbots = []
        self._frankas = []
        self._cube_names = []
        self._world = world
        
    

    def get_world(self):
        return self._world
    
    def scene_setup(self):
        world = self.get_world()
        for i in range(self._num_of_tasks):
            world.add_task(RobotsPlaying(name="my_awesome_task_" + str(i), offset=np.array([0, (i * 2) - 3, 0])))

        # For visualization purposes only, so we don't need to add it to the scene
        # Its there to only see where Franka was supposed to be vs after adding the offset
        VisualCuboid(prim_path="/new_cube_1",
                    name="visual_cube",
                    position=np.array([1.0, 0, 0.05]),
                    scale=np.array([0.1, 0.1, 0.1]))
        self._world.reset()
        
    def setup_post_load(self):
        for i in range(self._num_of_tasks):
            self._tasks.append(self._world.get_task(name="my_awesome_task_" + str(i)))
            # Get variables needed for control
            task_params = self._tasks[i].get_params()
            self._frankas.append(self._world.scene.get_object(task_params["franka_name"]["value"]))
            self._jetbots.append(self._world.scene.get_object(task_params["jetbot_name"]["value"]))
            self._cube_names.append(task_params["cube_name"]["value"])
            # Define controllers
            self._franka_controllers.append(PickPlaceController(name="pick_place_controller",
                                    gripper=self._frankas[i].gripper,
                                    robot_articulation=self._frankas[i],
                                    # Change the default events dt of the
                                    # pick and place controller to slow down some of the transitions
                                    # to pick up further blocks
                                    # Note: this is a simple pick and place state machine
                                    # based on events dt and not event success
                                    # check the different events description in the api
                                    # documentation
                                    events_dt=[0.008, 0.002, 0.5, 0.1, 0.05, 0.05, 0.0025, 1, 0.008, 0.08]))
            self._jetbot_controllers.append(WheelBasePoseController(name="cool_controller",
                                    open_loop_wheel_controller=
                                    DifferentialController(name="simple_control",
                                    wheel_radius=0.03, wheel_base=0.1125)))
        self._world.reset() # await self._world.play_async()
        return

    def setup_post_reset(self):
        for i in range(len(self._tasks)):
            # Reset all controllers
            self._franka_controllers[i].reset()
            self._jetbot_controllers[i].reset()
        self._world.reset() # await self._world.play_async()
        return

    def physics_step(self, step_size):
        current_observations = self._world.get_observations()
        for i in range(len(self._tasks)):
            # Apply actions for each task
            if current_observations[self._tasks[i].name + "_event"] == 0:
                self._jetbots[i].apply_wheel_actions(
                    self._jetbot_controllers[i].forward(
                        start_position=current_observations[self._jetbots[i].name]["position"],
                        start_orientation=current_observations[self._jetbots[i].name]["orientation"],
                        goal_position=current_observations[self._jetbots[i].name]["goal_position"]))
            elif current_observations[self._tasks[i].name + "_event"] == 1:
                self._jetbots[i].apply_wheel_actions(ArticulationAction(joint_velocities=[-8.0, -8.0]))
            elif current_observations[self._tasks[i].name + "_event"] == 2:
                self._jetbots[i].apply_wheel_actions(ArticulationAction(joint_velocities=[0.0, 0.0]))
                actions = self._franka_controllers[i].forward(
                    picking_position=current_observations[self._cube_names[i]]["position"],
                    placing_position=current_observations[self._cube_names[i]]["target_position"],
                    current_joint_positions=current_observations[self._frankas[i].name]["joint_positions"])
                self._frankas[i].apply_action(actions)
        return
    


####################### Setup world ##########################
world_manager = HelloWorld()
world = world_manager.get_world()
###################### Add task to the world #######################

# We add the task to the world here
world_manager.scene_setup() 
world_manager.setup_post_load()
world_manager.setup_post_reset() 

######################## Main Loop #########################
while simulation_app.is_running():
    world.step(render=True)
    world_manager.physics_step(None)
simulation_app.close()