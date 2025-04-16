#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

try:
    # Third Party
    import isaacsim
except ImportError:
    pass

# Third Party
import cv2
import torch
import numpy as np
from matplotlib import cm
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": False,
        "width": "1920",
        "height": "1080",
    }
)

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.camera import CameraObservation
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.types.state import JointState
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
import matplotlib.pyplot as plt

simulation_app.update()

# Standard Library
import argparse

# Third Party
import carb
from helper import VoxelManager, add_robot_to_scene
from omni.isaac.core import World
from omni.isaac.core.materials import OmniPBR
from omni.isaac.core.objects import cuboid, sphere
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.sensor import Camera
import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.isaac.core.objects import DynamicCuboid

from curobo.util.usd_helper import UsdHelper

class SimulatedRealsense:
    def __init__(self, clipping_distance_m=0.7):
        self.clipping_distance = clipping_distance_m
        self.frame_count = 0
        self.intrinsics = torch.tensor([
            [640, 0, 320],
            [0, 640, 240],
            [0, 0, 1]
        ], dtype=torch.float32)
    
    def get_data(self):
        # Create simulated depth image (grayscale)
        depth = np.zeros((480, 640), dtype=np.float32)
        # Add some simple animated pattern
        x, y = np.meshgrid(np.linspace(0, 639, 640), np.linspace(0, 479, 480))
        depth = ((np.sin(x/100 + self.frame_count/10) + np.cos(y/100)) * 1000).astype(np.float32)
        
        # Create simulated color image
        color = np.zeros((480, 640, 3), dtype=np.uint8)
        color[:, :, 0] = ((np.sin(x/100) + 1) * 127).astype(np.uint8)
        color[:, :, 1] = ((np.cos(y/100) + 1) * 127).astype(np.uint8)
        color[:, :, 2] = 128
        
        # Convert to tensor for depth
        depth_tensor = torch.from_numpy(depth) / 1000.0  # Convert to meters
        
        self.frame_count += 1
        
        return {
            "raw_depth": depth,
            "raw_rgb": color,
            "depth": depth_tensor,
            "intrinsics": self.intrinsics
        }
    
    def stop_device(self):
        pass

parser = argparse.ArgumentParser()
parser.add_argument(
    "--show-window",
    action="store_true",
    help="When True, shows camera image in a CV window",
    default=False,
)
parser.add_argument("--robot", type=str, default="franka.yml", help="robot configuration to load")
parser.add_argument(
    "--use-debug-draw",
    action="store_true",
    help="When True, sets robot in static mode",
    default=False,
)
args = parser.parse_args()

def draw_points(voxels):
    from omni.isaac.debug_draw import _debug_draw
    draw = _debug_draw.acquire_debug_draw_interface()
    draw.clear_points()
    if len(voxels) == 0:
        return

    jet = cm.get_cmap("plasma").reversed()
    cpu_pos = voxels[..., :3].view(-1, 3).cpu().numpy()
    z_val = cpu_pos[:, 0]
    jet_colors = jet(z_val)

    b, _ = cpu_pos.shape
    point_list = []
    colors = []
    for i in range(b):
        point_list += [(cpu_pos[i, 0], cpu_pos[i, 1], cpu_pos[i, 2])]
        colors += [(jet_colors[i][0], jet_colors[i][1], jet_colors[i][2], 0.8)]
    sizes = [20.0 for _ in range(b)]
    draw.draw_points(point_list, colors, sizes)

def clip_camera(camera_data):
    h_ratio = 0.05
    w_ratio = 0.05
    depth = camera_data["raw_depth"]
    depth_tensor = camera_data["depth"]
    h, w = depth_tensor.shape
    depth[: int(h_ratio * h), :] = 0.0
    depth[int((1 - h_ratio) * h) :, :] = 0.0
    depth[:, : int(w_ratio * w)] = 0.0
    depth[:, int((1 - w_ratio) * w) :] = 0.0

    depth_tensor[: int(h_ratio * h), :] = 0.0
    depth_tensor[int(1 - h_ratio * h) :, :] = 0.0
    depth_tensor[:, : int(w_ratio * w)] = 0.0
    depth_tensor[:, int(1 - w_ratio * w) :] = 0.0

if __name__ == "__main__":
    radius = 0.05
    act_distance = 0.4
    voxel_size = 0.05
    render_voxel_size = 0.02
    clipping_distance = 0.7
    
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage
    my_world.scene.add_default_ground_plane()

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    target_material = OmniPBR("/World/looks/t", color=np.array([0, 1, 0]))
    target_material_2 = OmniPBR("/World/looks/t2", color=np.array([0, 1, 0]))

    target = cuboid.VisualCuboid(
        "/World/target_1",
        position=np.array([0.4, -0.5, 0.2]),
        orientation=np.array([0, 1.0, 0, 0]),
        size=0.04,
        visual_material=target_material,
    )

    target_2 = cuboid.VisualCuboid(
        "/World/target_2",
        position=np.array([0.4, 0.5, 0.2]),
        orientation=np.array([0.0, 1, 0.0, 0.0]),
        size=0.04,
        visual_material=target_material_2,
    )

    camera_marker = cuboid.VisualCuboid(
        "/World/camera_nvblox",
        position=np.array([-0.05, 0.0, 0.45]),
        orientation=np.array([0.5, -0.5, 0.5, -0.5]),
        color=np.array([0, 0, 1]),
        size=0.01,
    )
    camera_marker.set_visibility(False)
    

    

    collision_checker_type = CollisionCheckerType.BLOX
    world_cfg = WorldConfig.from_dict(
        {
            "blox": {
                "world": {
                    "pose": [0, 0, 0, 1, 0, 0, 0],
                    "integrator_type": "occupancy",
                    "voxel_size": 0.02,
                }
            }
        }
    )
    tensor_args = TensorDeviceType()

    robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]
    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    robot, _ = add_robot_to_scene(robot_cfg, my_world, "/World/world_robot/")

    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_wall.yml"))
    )

    world_cfg_table.cuboid[0].pose[2] -= 0.01
    usd_help = UsdHelper()
    usd_help.load_stage(my_world.stage)
    usd_help.add_world_to_stage(world_cfg_table.get_mesh_world(), base_frame="/World")
    world_cfg.add_obstacle(world_cfg_table.cuboid[0])
    world_cfg.add_obstacle(world_cfg_table.cuboid[1])

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        trajopt_tsteps=32,
        collision_checker_type=CollisionCheckerType.BLOX,
        use_cuda_graph=True,
        num_trajopt_seeds=12,
        num_graph_seeds=12,
        interpolation_dt=0.03,
        collision_activation_distance=0.025,
        acceleration_scale=1.0,
        self_collision_check=True,
        maximum_trajectory_dt=0.25,
        finetune_dt_scale=1.05,
        fixed_iters_trajopt=True,
        finetune_trajopt_iters=300,
        minimize_jerk=True,
    )
    motion_gen = MotionGen(motion_gen_config)
    print("warming up..")
    motion_gen.warmup(warmup_js_trajopt=False)

    world_model = motion_gen.world_collision
    realsense_data = SimulatedRealsense(clipping_distance_m=clipping_distance)
    data = realsense_data.get_data()

    camera_pose = Pose.from_list([0, 0, 0, 0.707, 0.707, 0, 0])
    i = 0
    tensor_args = TensorDeviceType()
    target_list = [target, target_2]
    target_material_list = [target_material, target_material_2]
    for material in target_material_list:
        material.set_color(np.array([0.1, 0.1, 0.1]))
    target_idx = 0
    cmd_idx = 0
    cmd_plan = None
    articulation_controller = robot.get_articulation_controller()
    plan_config = MotionGenPlanConfig(
        enable_graph=False, enable_graph_attempt=4, max_attempts=2, enable_finetune_trajopt=True
    )
    if not args.use_debug_draw:
        voxel_viewer = VoxelManager(100, size=render_voxel_size)
    cmd_step_idx = 0

    cube_2 = my_world.scene.add(
     DynamicCuboid(
        prim_path="/new_cube_2",
        name="cube_1",
        position=np.array([5.0, 3, 1.0]),
        scale=np.array([0.6, 0.5, 0.2]),
        size=1.0,
            color=np.array([255, 0, 0]),
        )
    )

    cube_3 = my_world.scene.add(
        DynamicCuboid(
            prim_path="/new_cube_3",
            name="cube_2",
            position=np.array([-5, 1, 3.0]),
            scale=np.array([0.1, 0.1, 0.1]),
            size=1.0,
            color=np.array([0, 0, 255]),
            linear_velocity=np.array([0, 0, 0.4]),
        )
    )

    camera = Camera(
        prim_path="/World/camera",
        position=np.array([0.0, 0.0, .0]),
        frequency=20,
        resolution=(256, 256),
        orientation=rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True),
    )
    camera.initialize()
    camera.add_motion_vectors_to_frame()
    i = 0
    while simulation_app.is_running():
        i+=1
        my_world.step(render=True)
        try:
            if i == 100:
                points_2d = camera.get_image_coords_from_world_points(
                np.array([cube_3.get_world_pose()[0], cube_2.get_world_pose()[0]])
                )
                points_3d = camera.get_world_points_from_image_coords(points_2d, np.array([24.94, 24.9]))
                print(points_2d)
                print(points_3d)
                imgplot = plt.imshow(camera.get_rgba()[:, :, :3])
                plt.show()
                print(camera.get_current_frame()["motion_vectors"])
        except:
            pass
        if not my_world.is_playing():
            if i % 100 == 0:
                print("**** Click Play to start simulation *****")
            i += 1
            continue
        step_index = my_world.current_time_step_index

        if step_index <= 2:
            my_world.reset()
            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(default_config, idx_list)
            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
            )

        if step_index % 5 == 0.0:
            world_model.decay_layer("world")
            data = realsense_data.get_data()
            clip_camera(data)
            cube_position, cube_orientation = camera_marker.get_local_pose()
            camera_pose = Pose(
                position=tensor_args.to_device(cube_position),
                quaternion=tensor_args.to_device(cube_orientation),
            )

            data_camera = CameraObservation(
                depth_image=data["depth"], intrinsics=data["intrinsics"], pose=camera_pose
            )
            data_camera = data_camera.to(device=tensor_args.device)
            world_model.add_camera_frame(data_camera, "world")
            world_model.process_camera_frames("world", False)
            torch.cuda.synchronize()
            world_model.update_blox_hashes()

            bounding = Cuboid("t", dims=[1, 1, 1.0], pose=[0, 0, 0, 1, 0, 0, 0])
            voxels = world_model.get_voxels_in_bounding_box(bounding, voxel_size)
            if voxels.shape[0] > 0:
                voxels = voxels[voxels[:, 2] > voxel_size]
                voxels = voxels[voxels[:, 0] > 0.0]
                if args.use_debug_draw:
                    draw_points(voxels)
                else:
                    voxels = voxels.cpu().numpy()
                    voxel_viewer.update_voxels(voxels[:, :3])
            else:
                if not args.use_debug_draw:
                    voxel_viewer.clear()

        if args.show_window:
            depth_image = data["raw_depth"]
            color_image = data["raw_rgb"]
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=100), cv2.COLORMAP_VIRIDIS
            )
            color_image = cv2.flip(color_image, 1)
            depth_colormap = cv2.flip(depth_colormap, 1)

            images = np.hstack((color_image, depth_colormap))

            cv2.namedWindow("NVBLOX Example", cv2.WINDOW_NORMAL)
            cv2.imshow("NVBLOX Example", images)
            key = cv2.waitKey(1)
            if key & 0xFF == ord("q") or key == 27:
                cv2.destroyAllWindows()
                break

        if cmd_plan is None and step_index % 10 == 0:
            for ks in range(len(target_material_list)):
                if ks == target_idx:
                    target_material_list[ks].set_color(np.ravel([0, 1.0, 0]))
                else:
                    target_material_list[ks].set_color(np.ravel([0.1, 0.1, 0.1]))

            sim_js = robot.get_joints_state()
            sim_js_names = robot.dof_names
            cu_js = JointState(
                position=tensor_args.to_device(sim_js.positions),
                velocity=tensor_args.to_device(sim_js.velocities) * 0.0,
                acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
                jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
                joint_names=sim_js_names,
            )
            cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)

            cube_position, cube_orientation = target_list[target_idx].get_world_pose()
            ee_translation_goal = cube_position
            ee_orientation_teleop_goal = cube_orientation

            ik_goal = Pose(
                position=tensor_args.to_device(ee_translation_goal),
                quaternion=tensor_args.to_device(ee_orientation_teleop_goal),
            )

            result = motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)
            succ = result.success.item()
            if succ:
                cmd_plan = result.get_interpolated_plan()
                cmd_plan = motion_gen.get_full_js(cmd_plan)
                idx_list = []
                common_js_names = []
                for x in sim_js_names:
                    if x in cmd_plan.joint_names:
                        idx_list.append(robot.get_dof_index(x))
                        common_js_names.append(x)
                cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)
                cmd_idx = 0
                target_idx += 1
                if target_idx >= len(target_list):
                    target_idx = 0
            else:
                carb.log_warn("Plan did not converge to a solution. No action is being taken.")

        if cmd_plan is not None:
            cmd_state = cmd_plan[cmd_idx]
            art_action = ArticulationAction(
                cmd_state.position.cpu().numpy(),
                cmd_state.velocity.cpu().numpy(),
                joint_indices=idx_list,
            )
            articulation_controller.apply_action(art_action)
            cmd_step_idx += 1
            if cmd_step_idx == 2:
                cmd_idx += 1
                cmd_step_idx = 0
            if cmd_idx >= len(cmd_plan.position):
                cmd_idx = 0
                cmd_plan = None

    realsense_data.stop_device()
    print("finished program")
    simulation_app.close() 