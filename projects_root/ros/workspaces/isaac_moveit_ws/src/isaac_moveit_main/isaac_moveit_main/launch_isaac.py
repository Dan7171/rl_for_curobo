#!/isaac-sim/python.sh
"""
This script launches isaac sim with 
the relevant robot and action graph listening and publishing joint states and time.
"""
  
import sys
import re
import os
import carb
import numpy as np
from pathlib import Path
import argparse
import yaml

def add_robot_to_scene(
    my_world,
    subroot: str = "",
    robot_name: str = "robot",
    position: np.array = np.array([0, 0, 0]),
    initialize_world: bool = True,
    
):
    """
    Based on projects_root/examples/helper.py but aimed to load urdf, from any path unlike the original helper.py which only loads urdf from the assets path

    Args:
        my_world: World object
        subroot: str
        robot_name: str
        position: np.array
        initialize_world: bool
    """

    from curobo.util.usd_helper import set_prim_transform
    from curobo.util_file import  get_filename, get_path_of_dir, join_path
    from projects_root.examples.helper import find_articulation_root
    from omni.isaac.core.robots import Robot

    try:
        # Third Party
        from omni.isaac.urdf import _urdf  # isaacsim 2022.2
    
    except ImportError:
        # Third Party
        try:
            from omni.importer.urdf import _urdf  # isaac sim 2023.1 or above
        except ImportError:
            from isaacsim.asset.importer.urdf import _urdf  # isaac sim 4.5+
            ISAAC_SIM_45 = True

    urdf_interface = _urdf.acquire_urdf_interface()
    # Set the settings in the import config
    import_config = _urdf.ImportConfig()
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.fix_base = True
    import_config.make_default_prim = True
    import_config.self_collision = False
    import_config.create_physics_scene = True
    import_config.import_inertia_tensor = False
    import_config.default_drive_strength = 1047.19751
    import_config.default_position_drive_damping = 52.35988
    import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
    import_config.distance_scale = 1
    import_config.density = 0.0

    # urdf_path:
    # meshes_path:
    # meshes path should be a subset of urdf_path
    full_path = cfg["urdf_file_path"] # join_path(asset_path, robot_config["kinematics"]["urdf_path"])
    # full path contains the path to urdf
    # Get meshes path
    robot_path = get_path_of_dir(full_path)
    filename = get_filename(full_path)
    print(f"debug robot_path: {robot_path}")
    print(f"debug filename: {filename}")
    
    if ISAAC_SIM_45:
        from isaacsim.core.utils.extensions import get_extension_path_from_name
        import omni.kit.commands
        import omni.usd

        # Retrieve the path of the URDF file from the extension
        extension_path = get_extension_path_from_name("isaacsim.asset.importer.urdf")
        root_path = robot_path
        file_name = filename

        # Parse the robot's URDF file to generate a robot model

        dest_path = join_path(
            root_path, get_filename(file_name, remove_extension=True) + "_temp.usd"
        )
        result, robot_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path="{}/{}".format(root_path, file_name),
            import_config=import_config,
            dest_path=dest_path,
        )
        print(f"debug full_path (urdf path): {full_path}")
        print(f"debug result of import: {result}")
        print(f"debug dest_path: {dest_path}")
        print(f"debug robot_path: {robot_path}")
        print(f"my_world.scene.stage.GetDefaultPrim().GetPath(): {my_world.scene.stage.GetDefaultPrim().GetPath()}")
        prim_path = omni.usd.get_stage_next_free_path(
            my_world.scene.stage,
            str(my_world.scene.stage.GetDefaultPrim().GetPath()) + robot_path,
            False,
        )
        robot_prim = my_world.scene.stage.OverridePrim(prim_path)
        robot_prim.GetReferences().AddReference(dest_path)
        robot_path = prim_path
    else:

        imported_robot = urdf_interface.parse_urdf(robot_path, filename, import_config)
        dest_path = subroot

        robot_path = urdf_interface.import_robot(
            robot_path,
            filename,
            imported_robot,
            import_config,
            dest_path,
        )

    # Find the actual articulation root instead of assuming base_link
    robot_articulation_root_path = find_articulation_root(my_world.stage, robot_path)
    
    print(f"Robot imported at: {robot_path}")
    print(f"Articulation root found at: {robot_articulation_root_path}")

    robot_p = Robot(
        prim_path=robot_articulation_root_path,
        name=robot_name,
    )

    robot_prim = robot_p.prim
    stage = robot_prim.GetStage()
    linkp = stage.GetPrimAtPath(robot_path)
    set_prim_transform(linkp, [position[0], position[1], position[2], 1, 0, 0, 0])

    robot = my_world.scene.add(robot_p)
    if initialize_world:
        if ISAAC_SIM_45:
            my_world.initialize_physics()
            robot.initialize()

    return robot, robot_path, robot_articulation_root_path

    # original coding example used that:
    # # Loading the franka robot USD
    # prims.create_prim(
    #     robot_path,
    #     "Xform",
    #     position=np.array([0, -0.64, 0]),
    #     orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(0, 0, 1), 90)),
    #     usd_path=assets_root_path + ROBOT_USD_PATH,
    # )

def get_ros_domain_id():
    try:
        ros_domain_id = int(os.environ["ROS_DOMAIN_ID"])
        print("Using ROS_DOMAIN_ID: ", ros_domain_id)
    except ValueError:
        print("Invalid ROS_DOMAIN_ID integer value. Setting value to 0")
        ros_domain_id = 0
    except KeyError:
        print("ROS_DOMAIN_ID environment variable is not set. Setting value to 0")
        ros_domain_id = 0
    return ros_domain_id

def init_world():
    from omni.isaac.core import World
    world = World(stage_units_in_meters=1.0)
    xform = world.stage.DefinePrim("/World", "Xform")
    world.stage.SetDefaultPrim(xform)
    return world

def init_sim_app():
    # In older versions of Isaac Sim (prior to 4.0), SimulationApp is imported from
    # omni.isaac.kit rather than isaacsim.
    try:
        from isaacsim import SimulationApp
    except:
        from omni.isaac.kit import SimulationApp
    simulation_app = SimulationApp({"renderer": "RayTracedLighting", "headless": False})
    return simulation_app

def get_isaac_sim_version():
    # Use this flag to identify whether current release is Isaac Sim 4.5 or higher
    isaac_sim_ge_4_5_version = True
    # In older versions of Isaac Sim (prior to 4.5), get_version is imported from
    # omni.isaac.kit rather than isaacsim.core.version.
    try:
        from isaacsim.core.version import get_version
    except:
        from omni.isaac.version import get_version
        isaac_sim_ge_4_5_version = False
    # Check the major version number of Isaac Sim to see if it's four digits, corresponding
    # to Isaac Sim 2023.1.1 or older.  The version numbering scheme changed with the
    # Isaac Sim 4.0 release in 2024.
    is_legacy_isaacsim = len(get_version()[2]) == 4
    return isaac_sim_ge_4_5_version, is_legacy_isaacsim

def get_asset_database_path(simulation_app):
    # In older versions of Isaac Sim (prior to 4.5), nucleus is imported from
    # omni.isaac.core.utils rather than isaacsim.storage.native.
    if isaac_sim_ge_4_5_version:
        from isaacsim.storage.native import nucleus
    else:
        from omni.isaac.core.utils import nucleus  # noqa E402
    assets_root_path = nucleus.get_assets_root_path()
    if assets_root_path is None:
        carb.log_error("Could not find Isaac Sim assets folder")
        simulation_app.close()
        sys.exit()
    return assets_root_path

def enable_extensions(isaac_sim_ge_4_5_version):

    try:
        from isaacsim.core.utils import (  # noqa E402
            extensions,
        )
    except:
        from omni.isaac.core.utils import (  # noqa E402
            extensions,
        )

    if isaac_sim_ge_4_5_version:
        extensions.enable_extension("isaacsim.ros2.bridge")
        extensions.enable_extension("omni.graph.window.action")
    else:
        extensions.enable_extension("omni.isaac.ros2_bridge")

def add_props_from_database(assets_root_path):
    try:
        from isaacsim.core.utils import (  # noqa E402
            prims,
            rotations,

        )
    except:
        from omni.isaac.core.utils import (  # noqa E402
            prims,
            rotations,
        )
    # add some objects, spread evenly along the X axis
    # with a fixed offset from the robot in the Y and Z
    from pxr import Gf  # noqa E402

    prims.create_prim(
        "/cracker_box",
        "Xform",
        position=np.array([-0.2, -0.25, 0.15]),
        orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)),
        usd_path=assets_root_path
        + "/Isaac/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
    )
    prims.create_prim(
        "/sugar_box",
        "Xform",
        position=np.array([-0.07, -0.25, 0.1]),
        orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(0, 1, 0), -90)),
        usd_path=assets_root_path
        + "/Isaac/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
    )
    prims.create_prim(
        "/soup_can",
        "Xform",
        position=np.array([0.1, -0.25, 0.10]),
        orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)),
        usd_path=assets_root_path
        + "/Isaac/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
    )
    prims.create_prim(
        "/mustard_bottle",
        "Xform",
        position=np.array([0.0, 0.15, 0.12]),
        orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)),
        usd_path=assets_root_path
        + "/Isaac/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
    )

def init_action_graph(graph_path, simulation_app, joint_states_out_topic, joint_commands_in_topic, robot_path, robot_articulation_root_path, isaac_sim_ge_4_5_version, is_legacy_isaacsim, ros_domain_id):
    if isaac_sim_ge_4_5_version:
    # print(f"debug articulation_root_link: {robot_path + cfg['articulation_root_link']}")
    # Create an action graph with ROS component nodes from Isaac Sim 4.5 release and higher
        try:
            og_keys_set_values = [
            ("Context.inputs:domain_id", ros_domain_id),
            ("ArticulationController.inputs:robotPath", robot_path),
            ("PublishJointState.inputs:topicName", joint_states_out_topic),
            ("PublishJointState.inputs:targetPrim", robot_path), # + "/" + cfg["articulation_root_link"]),
            ("SubscribeJointState.inputs:topicName", joint_commands_in_topic),
            # ("createViewport.inputs:name", REALSENSE_VIEWPORT_NAME),
            # ("createViewport.inputs:viewportId", 1),
            # ("cameraHelperRgb.inputs:frameId", "sim_camera"),
            # ("cameraHelperRgb.inputs:topicName", "rgb"),
            # ("cameraHelperRgb.inputs:type", "rgb"),
            # ("cameraHelperInfo.inputs:frameId", "sim_camera"),
            # ("cameraHelperInfo.inputs:topicName", "camera_info"),
            # ("cameraHelperDepth.inputs:frameId", "sim_camera"),
            # ("cameraHelperDepth.inputs:topicName", "depth"),
            # ("cameraHelperDepth.inputs:type", "depth"),
        ]

        # In older versions of Isaac Sim, the articulation controller node contained a
        # "usePath" checkbox input that should be enabled.
            if is_legacy_isaacsim:
                og_keys_set_values.insert(
                1, ("ArticulationController.inputs:usePath", True)
            )

            og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    # ("OnImpulseEvent", "omni.graph.action.OnImpulseEvent"),
                    ("OnTick", "omni.graph.action.OnTick"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                    ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                    ("SubscribeJointState","isaacsim.ros2.bridge.ROS2SubscribeJointState",),
                    ("ArticulationController","isaacsim.core.nodes.IsaacArticulationController",),
                    ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                    # ("createViewport", "isaacsim.core.nodes.IsaacCreateViewport"),
                    # (
                    #     "getRenderProduct",
                    #     "isaacsim.core.nodes.IsaacGetViewportRenderProduct",
                    # ),
                    # ("setCamera", "isaacsim.core.nodes.IsaacSetCameraOnRenderProduct"),
                    # ("cameraHelperRgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    # ("cameraHelperInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
                    # ("cameraHelperDepth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ],
                og.Controller.Keys.CONNECT: [
                    (
                        # "OnImpulseEvent.outputs:execOut",
                        "OnTick.outputs:tick",
                        "PublishJointState.inputs:execIn",
                    ),
                    (
                        # "OnImpulseEvent.outputs:execOut",
                        "OnTick.outputs:tick",
                        "SubscribeJointState.inputs:execIn",
                    ),
                    (
                        # "OnImpulseEvent.outputs:execOut", 
                        "OnTick.outputs:tick",
                        "PublishClock.inputs:execIn"),
                    (
                        # "OnImpulseEvent.outputs:execOut",
                        "OnTick.outputs:tick",
                        "ArticulationController.inputs:execIn",
                    ),
                    ("Context.outputs:context", "PublishJointState.inputs:context"),
                    ("Context.outputs:context", "SubscribeJointState.inputs:context"),
                    ("Context.outputs:context", "PublishClock.inputs:context"),
                    ("ReadSimTime.outputs:simulationTime","PublishJointState.inputs:timeStamp",),
                    ("ReadSimTime.outputs:simulationTime","PublishClock.inputs:timeStamp",),
                    ("SubscribeJointState.outputs:jointNames","ArticulationController.inputs:jointNames",),
                    ("SubscribeJointState.outputs:positionCommand","ArticulationController.inputs:positionCommand",),
                    ("SubscribeJointState.outputs:velocityCommand","ArticulationController.inputs:velocityCommand",),
                    ("SubscribeJointState.outputs:effortCommand","ArticulationController.inputs:effortCommand",),
                    # ("OnTick.outputs:tick", "createViewport.inputs:execIn"),
                    # (
                    #     "createViewport.outputs:execOut",
                    #     "getRenderProduct.inputs:execIn",
                    # ),
                    # (
                    #     "createViewport.outputs:viewport",
                    #     "getRenderPrort.outputs:viewport",
                    #     "getRenderProduct.inputs:viewport",
                    # ),
                    # ("getRenderProduct.outputs:execOut", "setCamera.inputs:execIn"),
                    # (
                    #     "getRenderProduct.outputs:renderProductPath",
                    #     "setCamera.inputs:renderProductPath",
                    # ),
                    # ("setCamera.outputs:execOut", "cameraHelperRgb.inputs:execIn"),
                    # ("setCamera.outputs:execOut", "cameraHelperInfo.inputs:execIn"),
                    # ("setCamera.outputs:execOut", "cameraHelperDepth.inputs:execIn"),
                    # ("Context.outputs:context", "cameraHelperRgb.inputs:context"),
                    # ("Context.outputs:context", "cameraHelperInfo.inputs:context"),
                    # ("Context.outputs:context", "cameraHelperDepth.inputs:context"),
                    # (
                    #     "getRenderProduct.outputs:renderProductPath",
                    #     "cameraHelperRgb.inputs:renderProductPath",
                    # ),
                    # (
                    #     "getRenderProduct.outputs:renderProductPath",
                    #     "cameraHelperInfo.inputs:renderProductPath",
                    # ),
                    # (
                    #     "getRenderProduct.outputs:renderProductPath",
                    #     "cameraHelperDepth.inputs:renderProductPath",
                    # ),
                ],
                og.Controller.Keys.SET_VALUES: og_keys_set_values,
            },
        )

        except Exception as e:
            print(e)

    else:
    # Create an action graph with ROS component nodes from a pre Isaac Sim 4.5 release
        try:
            og_keys_set_values = [
            ("Context.inputs:domain_id", ros_domain_id),
            # Set the /Franka target prim to Articulation Controller node
            ("ArticulationController.inputs:robotPath", robot_path),
            ("PublishJointState.inputs:topicName", joint_states_out_topic),
            ("SubscribeJointState.inputs:topicName", joint_commands_in_topic),
            # ("createViewport.inputs:name", REALSENSE_VIEWPORT_NAME),
            ("createViewport.inputs:viewportId", 1),
            ("cameraHelperRgb.inputs:frameId", "sim_camera"),
            ("cameraHelperRgb.inputs:topicName", "rgb"),
            ("cameraHelperRgb.inputs:type", "rgb"),
            ("cameraHelperInfo.inputs:frameId", "sim_camera"),
            ("cameraHelperInfo.inputs:topicName", "camera_info"),
            ("cameraHelperInfo.inputs:type", "camera_info"),
            ("cameraHelperDepth.inputs:frameId", "sim_camera"),
            ("cameraHelperDepth.inputs:topicName", "depth"),
            ("cameraHelperDepth.inputs:type", "depth"),
        ]

        # In older versions of Isaac Sim, the articulation controller node contained a
        # "usePath" checkbox input that should be enabled.
            if is_legacy_isaacsim:
                og_keys_set_values.insert(
                1, ("ArticulationController.inputs:usePath", True)
            )

            og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnImpulseEvent", "omni.graph.action.OnImpulseEvent"),
                    ("ReadSimTime", "omni.isaac.core_nodes.IsaacReadSimulationTime"),
                    ("Context", "omni.isaac.ros2_bridge.ROS2Context"),
                    (
                        "PublishJointState",
                        "omni.isaac.ros2_bridge.ROS2PublishJointState",
                    ),
                    (
                        "SubscribeJointState",
                        "omni.isaac.ros2_bridge.ROS2SubscribeJointState",
                    ),
                    (
                        "ArticulationController",
                        "omni.isaac.core_nodes.IsaacArticulationController",
                    ),
                    ("PublishClock", "omni.isaac.ros2_bridge.ROS2PublishClock"),
                    ("OnTick", "omni.graph.action.OnTick"),
                    ("createViewport", "omni.isaac.core_nodes.IsaacCreateViewport"),
                    (
                        "getRenderProduct",
                        "omni.isaac.core_nodes.IsaacGetViewportRenderProduct",
                    ),
                    (
                        "setCamera",
                        "omni.isaac.core_nodes.IsaacSetCameraOnRenderProduct",
                    ),
                    ("cameraHelperRgb", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                    ("cameraHelperInfo", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                    ("cameraHelperDepth", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                ],
                og.Controller.Keys.CONNECT: [
                    (
                        "OnImpulseEvent.outputs:execOut",
                        "PublishJointState.inputs:execIn",
                    ),
                    (
                        "OnImpulseEvent.outputs:execOut",
                        "SubscribeJointState.inputs:execIn",
                    ),
                    ("OnImpulseEvent.outputs:execOut", "PublishClock.inputs:execIn"),
                    (
                        "OnImpulseEvent.outputs:execOut",
                        "ArticulationController.inputs:execIn",
                    ),
                    ("Context.outputs:context", "PublishJointState.inputs:context"),
                    ("Context.outputs:context", "SubscribeJointState.inputs:context"),
                    ("Context.outputs:context", "PublishClock.inputs:context"),
                    (
                        "ReadSimTime.outputs:simulationTime",
                        "PublishJointState.inputs:timeStamp",
                    ),
                    (
                        "ReadSimTime.outputs:simulationTime",
                        "PublishClock.inputs:timeStamp",
                    ),
                    (
                        "SubscribeJointState.outputs:jointNames",
                        "ArticulationController.inputs:jointNames",
                    ),
                    (
                        "SubscribeJointState.outputs:positionCommand",
                        "ArticulationController.inputs:positionCommand",
                    ),
                    (
                        "SubscribeJointState.outputs:velocityCommand",
                        "ArticulationController.inputs:velocityCommand",
                    ),
                    (
                        "SubscribeJointState.outputs:effortCommand",
                        "ArticulationController.inputs:effortCommand",
                    ),
                    ("OnTick.outputs:tick", "createViewport.inputs:execIn"),
                    (
                        "createViewport.outputs:execOut",
                        "getRenderProduct.inputs:execIn",
                    ),
                    (
                        "createViewport.outputs:viewport",
                        "getRenderProduct.inputs:viewport",
                    ),
                    ("getRenderProduct.outputs:execOut", "setCamera.inputs:execIn"),
                    (
                        "getRenderProduct.outputs:renderProductPath",
                        "setCamera.inputs:renderProductPath",
                    ),
                    ("setCamera.outputs:execOut", "cameraHelperRgb.inputs:execIn"),
                    ("setCamera.outputs:execOut", "cameraHelperInfo.inputs:execIn"),
                    ("setCamera.outputs:execOut", "cameraHelperDepth.inputs:execIn"),
                    ("Context.outputs:context", "cameraHelperRgb.inputs:context"),
                    ("Context.outputs:context", "cameraHelperInfo.inputs:context"),
                    ("Context.outputs:context", "cameraHelperDepth.inputs:context"),
                    (
                        "getRenderProduct.outputs:renderProductPath",
                        "cameraHelperRgb.inputs:renderProductPath",
                    ),
                    (
                        "getRenderProduct.outputs:renderProductPath",
                        "cameraHelperInfo.inputs:renderProductPath",
                    ),
                    (
                        "getRenderProduct.outputs:renderProductPath",
                        "cameraHelperDepth.inputs:renderProductPath",
                    ),
                ],
                og.Controller.Keys.SET_VALUES: og_keys_set_values,
            },
        )
        except Exception as e:
            print(e)

        simulation_app.update()

    if isaac_sim_ge_4_5_version:
    # Setting the  target prim to Publish JointState node
        set_targets(
        prim=stage.get_current_stage().GetPrimAtPath("/ActionGraph/PublishJointState"),
        attribute="inputs:targetPrim",
        target_prim_paths=[robot_articulation_root_path], # target_prim_paths=[robot_path],
    )
    else:
        from omni.isaac.core_nodes.scripts.utils import set_target_prims  # noqa E402
        set_target_prims(
        primPath="/ActionGraph/PublishJointState", target_prim_paths=[robot_articulation_root_path],# targetPrimPaths=[robot_path]
    )

def spawn_targets(num_targets):
    from omni.isaac.core.objects import cuboid
    targets = []
    for i in range(num_targets):
        targets.append(cuboid.VisualCuboid(
            f"/World/target_{i}",
            position=np.array([0.5, 0, 0.5]),
            orientation=np.array([0, 1, 0, 0]),
            color=np.array([1.0, 0, 0]),
            size=0.05,
        ))
    
    return targets

BACKGROUND_STAGE_PATH = "/background"
BACKGROUND_USD_PATH = "/Isaac/Environments/Simple_Room/simple_room.usd"
GRAPH_PATH = "/ActionGraph"
REALSENSE_VIEWPORT_NAME = "realsense_viewport"
ADD_PROPS = True

# get cfg path
args = argparse.ArgumentParser()
args.add_argument("--cfg_file", type=str, required=True)
args = args.parse_args()

# get cfg
cfg_file = args.cfg_file
with open(cfg_file, "r") as f:
    cfg = yaml.safe_load(f)

simulation_app = init_sim_app()
world = init_world()
robot, robot_path, robot_articulation_root_path = add_robot_to_scene(world, position=np.array([0, 0, 0]))
isaac_sim_ge_4_5_version, is_legacy_isaacsim = get_isaac_sim_version()
assets_root_path = get_asset_database_path(simulation_app)
num_targets = cfg.get("num_targets",1)
targets = spawn_targets(num_targets)


try:
    from isaacsim.core.api import SimulationContext  # noqa E402
    from isaacsim.core.utils.prims import set_targets  # noqa E402
    from isaacsim.core.utils import (  # noqa E402
        stage,
        viewports,
    )
except:
    from omni.isaac.core import SimulationContext  # noqa E402
    from omni.isaac.core.utils.prims import set_targets  # noqa E402
    from omni.isaac.core.utils import (  # noqa E402
        stage,
        viewports,
    )
enable_extensions(isaac_sim_ge_4_5_version)
import omni.graph.core as og  # noqa E402
import omni
simulation_context = SimulationContext(stage_units_in_meters=1.0)

# Loading the simple_room environment
stage.add_reference_to_stage(
    assets_root_path + BACKGROUND_USD_PATH, BACKGROUND_STAGE_PATH
)

# Preparing stage
viewports.set_camera_view(eye=np.array([1.2, 1.2, 0.8]), target=np.array([0, 0, 0.5]))
if ADD_PROPS:
    add_props_from_database(assets_root_path)

simulation_app.update()
ros_domain_id = get_ros_domain_id()

from projects_root.ros.workspaces.isaac_moveit_ws.src.isaac_moveit_main.isaac_moveit_main.target_state_publisher import TargetStatePublisher
target_state_publisher = TargetStatePublisher(targets)

init_action_graph(GRAPH_PATH, simulation_app, cfg["joint_states_out_topic"], cfg["joint_commands_in_topic"], robot_path, robot_articulation_root_path, isaac_sim_ge_4_5_version, is_legacy_isaacsim, ros_domain_id)

# Fix camera settings since the defaults in the realsense model are inaccurate
# realsense_prim = camera_prim = UsdGeom.Camera(
#     stage.get_current_stage().GetPrimAtPath(CAMERA_PRIM_PATH)
# )
# realsense_prim.GetHorizontalApertureAttr().Set(20.955)
# realsense_prim.GetVerticalApertureAttr().Set(15.7)
# realsense_prim.GetFocalLengthAttr().Set(18.8)
# realsense_prim.GetFocusDistanceAttr().Set(400)

# set_targets(
#     prim=stage.get_current_stage().GetPrimAtPath(graph_path + "/setCamera"),
#     attribute="inputs:cameraPrim",
#     target_prim_paths=[CAMERA_PRIM_PATH],
# )

# Run app update for multiple frames to re-initialize the ROS action graph after setting new prim inputs
simulation_app.update()
simulation_app.update()

# need to initialize physics getting any articulation..etc
simulation_context.initialize_physics()

simulation_context.play()
simulation_app.update()

# Dock the second camera window
# viewport = omni.ui.Workspace.get_window("Viewport")
# rs_viewport = omni.ui.Workspace.get_window(REALSENSE_VIEWPORT_NAME)
# rs_viewport.dock_in(viewport, omni.ui.DockPosition.RIGHT)


while simulation_app.is_running():

    # Run with a fixed step size
    simulation_context.step(render=True)
    # IMPULSE EVENT WAS REPLACED BY ON TICK NODE
    # # Tick the Publish/Subscribe JointState, Publish TF and Publish Clock nodes each frame
    # og.Controller.set(
    #     og.Controller.attribute("/ActionGraph/OnImpulseEvent.state:enableImpulse"), True
    # )

simulation_context.stop()
simulation_app.close()


def main():
    """
    Main function to launch Isaac Sim with MoveIt integration.
    """
    print("Launching Isaac Sim with MoveIt integration...")
    # The main execution is already in the global scope
    # This function is mainly for ROS2 entry point compatibility


if __name__ == '__main__':
    main()
