# https://docs.omniverse.nvidia.com/isaacsim/latest/core_api_tutorials/tutorial_core_hello_robot.html#isaac-sim-app-tutorial-core-hello-robot


# Third Party
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp(
    {
        "headless":False,
        "width": "1920",
        "height": "1080",
    }
)

from omni.isaac.core.utils.extensions import enable_extension
def add_extensions(simulation_app, headless_mode=None):
    ext_list = [
        "omni.kit.asset_converter",
        "omni.kit.tool.asset_importer",
        "omni.isaac.asset_browser",
        "omni.isaac.examples.base_sample"
    ]
    if headless_mode is not None:
        ext_list += ["omni.kit.livestream." + headless_mode]
    [enable_extension(x) for x in ext_list]
    simulation_app.update()

    return True
add_extensions(simulation_app)
from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.robots import Robot
import numpy as np
import carb


class HelloWorld(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        return

    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find nucleus server with /Isaac folder")
        asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
        add_reference_to_stage(usd_path=asset_path, prim_path="/World/Fancy_Robot")
        jetbot_robot = world.scene.add(Robot(prim_path="/World/Fancy_Robot", name="fancy_robot"))
        return

    async def setup_post_load(self):
        self._world = self.get_world()
        self._jetbot = self._world.scene.get_object("fancy_robot")
        # This is an implicit PD controller of the jetbot/ articulation
        # setting PD gains, applying actions, switching control modes..etc.
        # can be done through this controller.
        # Note: should be only called after the first reset happens to the world
        self._jetbot_articulation_controller = self._jetbot.get_articulation_controller()
        # Adding a physics callback to send the actions to apply actions with every
        # physics step executed.
        self._world.add_physics_callback("sending_actions", callback_fn=self.send_robot_actions)
        return

    def send_robot_actions(self, step_size):
        # Every articulation controller has apply_action method
        # which takes in ArticulationAction with joint_positions, joint_efforts and joint_velocities
        # as optional args. It accepts numpy arrays of floats OR lists of floats and None
        # None means that nothing is applied to this dof index in this step
        # ALTERNATIVELY, same method is called from self._jetbot.apply_action(...)
        self._jetbot_articulation_controller.apply_action(ArticulationAction(joint_positions=None,
                                                                            joint_efforts=None,
                                                                            joint_velocities=5 * np.random.rand(2,)))
        return