#!/usr/bin/env python3
"""
Multi-Robot MPC with ROS2 Publisher-Subscriber Architecture

This module implements a distributed multi-robot MPC system using ROS2 for communication.
Each robot publishes its plan and subscribes to plans from robots it needs to avoid.
Uses clean architecture with pre-populated robot context to eliminate code duplication.
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
import threading
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import json

# ROS2 message types
from std_msgs.msg import String
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion

# Isaac Sim and CuRobo imports
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
# Third party modules
import time
import signal
from typing import List, Optional, Tuple
import torch
import os
import numpy as np
# Initialize isaacsim app and load extensions
from projects_root.utils.issacsim import init_app, wait_for_playing, activate_gpu_dynamics
simulation_app = init_app() # must happen before importing other isaac sim modules, or any other module which imports isaac sim modules.
from projects_root.utils.helper import add_extensions # available only after app initiation
add_extensions(simulation_app, None if not HEADLESS_ISAAC else 'true') # in all of the examples of curobo it happens somwhere around here, before the simulation begins. I am not sure why, but I kept it as that. 
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

# import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core import World
from omni.isaac.core.utils.types import ArticulationAction
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.types.base import TensorDeviceType
from curobo.types.robot import JointState
from curobo.geom.types import WorldConfig

from projects_root.autonomous_arm import ArmMpc
from projects_root.utils.usd_utils import UsdHelper
from projects_root.projects.dynamic_obs.dynamic_obs_predictor.obstacle import Obstacle
from projects_root.projects.dynamic_obs.dynamic_obs_predictor.runtime_topics import init_runtime_topics, get_topics
from projects_root.utils.colors import get_color_palette

# System configuration constants
PHYSICS_STEP_DT = 1.0/60.0  # Physics simulation step size
RENDER_DT = 1.0/60.0        # Rendering step size
MPC_DT = 0.1               # MPC control frequency 
DEBUG = False
ENABLE_GPU_DYNAMICS = True


def calculate_robot_sphere_count(robot_cfg):
    """Calculate the number of collision spheres for a robot from its configuration"""
    base_spheres = len(robot_cfg.get("kinematics", {}).get("collision_spheres", []))
    extra_spheres = len(robot_cfg.get("kinematics", {}).get("extra_collision_spheres", []))
    return (base_spheres, extra_spheres)


def parse_meta_configs(meta_config_paths: List[str]) -> Tuple[List[str], List[str]]:
    """Parse meta configuration files to get robot and MPC config paths"""
    robot_config_paths = []
    mpc_config_paths = []
    
    for meta_path in meta_config_paths:
        if os.path.exists(meta_path):
            meta_config = load_yaml(meta_path)
            robot_config_paths.append(meta_config['robot'])
            mpc_config_paths.append(meta_config['mpc'])
        else:
            raise FileNotFoundError(f"Meta config file not found: {meta_path}")
    
    return robot_config_paths, mpc_config_paths


def define_run_setup(n_robots: int):
    """Define robot poses, collision prediction relationships, and targets"""
    # Robot base poses
    X_robots = np.array([
        [2.0, -1.5, 0.0, 1.0, 0.0, 0.0, 0.0],  # Robot 0
        [2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],   # Robot 1  
        [2.0, 1.5, 0.0, 1.0, 0.0, 0.0, 0.0]    # Robot 2
    ])
    
    # Collision prediction relationships (which robots each robot should avoid)
    col_pred_with = [
        [1, 2],  # Robot 0 avoids robots 1 and 2
        [0, 2],  # Robot 1 avoids robots 0 and 2
        [0, 1]   # Robot 2 avoids robots 0 and 1
    ]
    
    # Target poses for robots
    X_targets_R = [
        [0.4, -1.2, 0.5, 1.0, 0.0, 0.0, 0.0],  # Robot 0 target
        [0.4, 0.3, 0.5, 1.0, 0.0, 0.0, 0.0],   # Robot 1 target
        [0.4, 1.8, 0.5, 1.0, 0.0, 0.0, 0.0]    # Robot 2 target
    ]
    
    # Cost plotting and colors
    plot_costs = [True] * n_robots
    target_colors = get_color_palette(n_robots)
    
    return X_robots[:n_robots], col_pred_with[:n_robots], X_targets_R[:n_robots], plot_costs, target_colors


# Custom ROS2 message for robot plans
class RobotPlanMsg:
    """Custom message structure for robot plan data"""
    def __init__(self, robot_id: int, plan_data: dict, timestamp: float):
        self.robot_id = robot_id
        self.plan_data = plan_data
        self.timestamp = timestamp
    
    def to_json(self) -> str:
        """Convert to JSON string for publishing"""
        return json.dumps({
            'robot_id': self.robot_id,
            'plan_data': self.plan_data,
            'timestamp': self.timestamp
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> 'RobotPlanMsg':
        """Create from JSON string"""
        data = json.loads(json_str)
        return cls(data['robot_id'], data['plan_data'], data['timestamp'])


class RobotNode(Node):
    """
    ROS2 Node for individual robot MPC control with plan publishing/subscribing
    """
    
    def __init__(self, robot_id: int, robot: ArmMpc, col_pred_with: List[int], 
                 world: World, env_obstacles: List, tensor_args):
        super().__init__(f'robot_{robot_id}_node')
        
        self.robot_id = robot_id
        self.robot = robot
        self.col_pred_with = col_pred_with
        self.world = world
        self.env_obstacles = env_obstacles
        self.tensor_args = tensor_args
        
        # Callback group for concurrent execution
        self.callback_group = ReentrantCallbackGroup()
        
        # Plan storage for other robots
        self.other_robot_plans: Dict[int, dict] = {}
        self.plan_timestamps: Dict[int, float] = {}
        
        # Publisher for this robot's plan
        self.plan_publisher = self.create_publisher(
            String, 
            f'/robot_{robot_id}/plan', 
            10,
            callback_group=self.callback_group
        )
        
        # Subscribers for other robots' plans
        self.plan_subscribers = {}
        for other_robot_id in col_pred_with:
            subscriber = self.create_subscription(
                String,
                f'/robot_{other_robot_id}/plan',
                lambda msg, rid=other_robot_id: self.plan_callback(msg, rid),
                10,
                callback_group=self.callback_group
            )
            self.plan_subscribers[other_robot_id] = subscriber
        
        # Control loop timer
        self.control_timer = self.create_timer(
            MPC_DT,  # Use MPC_DT for control frequency
            self.control_loop,
            callback_group=self.callback_group
        )
        
        # Plan publishing timer
        self.plan_timer = self.create_timer(
            MPC_DT / 2,  # Publish plans at 2x control frequency
            self.publish_plan,
            callback_group=self.callback_group
        )
        
        self.get_logger().info(f'Robot {robot_id} node initialized')
        self.get_logger().info(f'Subscribing to plans from robots: {col_pred_with}')
    
    def plan_callback(self, msg: String, robot_id: int):
        """Callback for receiving other robots' plans"""
        try:
            plan_msg = RobotPlanMsg.from_json(msg.data)
            self.other_robot_plans[robot_id] = plan_msg.plan_data
            self.plan_timestamps[robot_id] = plan_msg.timestamp
            
            self.get_logger().debug(f'Received plan from robot {robot_id}')
        except Exception as e:
            self.get_logger().error(f'Error parsing plan from robot {robot_id}: {e}')
    
    def publish_plan(self):
        """Publish this robot's current plan"""
        try:
            if hasattr(self.robot, 'solver') and self.robot.solver is not None:
                plan_data = self.robot.get_plan(valid_spheres_only=False)
                
                # Convert tensor data to serializable format
                serializable_plan = self.convert_plan_to_serializable(plan_data)
                
                plan_msg = RobotPlanMsg(
                    robot_id=self.robot_id,
                    plan_data=serializable_plan,
                    timestamp=time.time()
                )
                
                msg = String()
                msg.data = plan_msg.to_json()
                self.plan_publisher.publish(msg)
                
        except Exception as e:
            self.get_logger().error(f'Error publishing plan: {e}')
    
    def convert_plan_to_serializable(self, plan_data: dict) -> dict:
        """Convert tensor data to JSON-serializable format"""
        serializable = {}
        
        for key, value in plan_data.items():
            if isinstance(value, dict):
                serializable[key] = self.convert_plan_to_serializable(value)
            elif isinstance(value, torch.Tensor):
                serializable[key] = value.cpu().numpy().tolist()
            else:
                serializable[key] = value
        
        return serializable
    
    def reconstruct_plan_tensors(self, plan_data: dict) -> dict:
        """Reconstruct tensors from serializable format"""
        reconstructed = {}
        
        for key, value in plan_data.items():
            if isinstance(value, dict):
                reconstructed[key] = self.reconstruct_plan_tensors(value)
            elif isinstance(value, list):
                # Convert back to tensor
                reconstructed[key] = torch.tensor(value, device=self.tensor_args.device)
            else:
                reconstructed[key] = value
        
        return reconstructed
    
    def control_loop(self):
        """Main robot control loop"""
        try:
            # Update environment obstacles
            for obstacle in self.env_obstacles:
                obstacle.update_registered_ccheckers()
            
            # Prepare plans for collision prediction - Fixed: create proper list structure
            max_robot_id = max(self.col_pred_with) if self.col_pred_with else -1
            plans = [None] * (max_robot_id + 1) if max_robot_id >= 0 else []
            
            # Reconstruct plans from received data
            for robot_id in self.col_pred_with:
                if robot_id in self.other_robot_plans:
                    plans[robot_id] = self.reconstruct_plan_tensors(self.other_robot_plans[robot_id])
            
            # Update robot state and execute control
            if self.robot.use_col_pred and any(p is not None for p in plans):
                self.robot.update(plans, self.col_pred_with, 0, self.tensor_args, self.robot_id)
            else:
                self.robot.update()
            
            # Plan and execute action - Fixed: use correct method name
            action = self.robot.plan(max_attempts=2)
            self.robot.command(action, num_times=1)
            
        except Exception as e:
            self.get_logger().error(f'Error in control loop: {e}')


class MultiRobotMPCSystem:
    """
    Main system coordinator for multi-robot MPC with ROS2
    Uses clean architecture with pre-populated robot context
    """
    
    def __init__(self, meta_config_paths: List[str]):
        # Initialize ROS2
        rclpy.init()
        
        # Parse configurations
        self.robot_config_paths, self.mpc_config_paths = parse_meta_configs(meta_config_paths)
        self.n_robots = len(self.robot_config_paths)
        
        print(f"Starting multi-robot ROS2 simulation with {self.n_robots} robots")
        for i in range(self.n_robots):
            print(f"Robot {i}: robot_config='{self.robot_config_paths[i]}', mpc_config='{self.mpc_config_paths[i]}'")
        
        # Setup simulation and robots
        self.setup_simulation()
        self.setup_robots()
        
        # Create ROS2 nodes for each robot
        self.robot_nodes: List[RobotNode] = []
        self.setup_ros_nodes()
        
        # ROS2 executor
        self.executor = MultiThreadedExecutor()
    
    def setup_simulation(self):
        """Setup Isaac Sim world and environment"""
        # Isaac sim setup
        self.usd_help = UsdHelper()
        self.my_world = World(stage_units_in_meters=1.0)
        self.my_world.scene.add_default_ground_plane()
        self.my_world.set_simulation_dt(PHYSICS_STEP_DT, RENDER_DT)
        
        if ENABLE_GPU_DYNAMICS:
            from projects_root.utils.issacsim import activate_gpu_dynamics
            activate_gpu_dynamics(self.my_world)
        
        stage = self.my_world.stage
        self.usd_help.load_stage(stage)
        xform = stage.DefinePrim("/World", "Xform")
        stage.SetDefaultPrim(xform)
        stage.DefinePrim("/curobo", "Xform")
        
        # CuRobo setup
        setup_curobo_logger("warn")
        self.tensor_args = TensorDeviceType()
        
        # Setup collision obstacles
        collision_obstacles_cfg_path = "projects_root/projects/dynamic_obs/dynamic_obs_predictor/cfgs/collision_obstacles.yml"
        col_ob_cfg = load_yaml(collision_obstacles_cfg_path)
        self.env_obstacles = []
        
        for obstacle in col_ob_cfg:
            obs = Obstacle(self.my_world, **obstacle)
            self.env_obstacles.append(obs)
    
    def setup_robots(self):
        """Setup robot instances with clean architecture"""
        # Basic setup of the scenario
        self.X_robots, self.col_pred_with, self.X_targets_R, self.plot_costs, self.target_colors = define_run_setup(self.n_robots)
        
        # Runtime topics for clean architecture
        init_runtime_topics(n_envs=1, robots_per_env=self.n_robots)
        runtime_topics = get_topics()
        self.env_topics = runtime_topics.get_default_env() if runtime_topics is not None else []
        
        # Robot setup
        self.robots: List[ArmMpc] = []
        self.robot_world_models = [WorldConfig() for _ in range(self.n_robots)]
        self.robots_collision_caches = [{"obb": 5, "mesh": 5} for _ in range(self.n_robots)]
        
        # Calculate sphere counts for all robots BEFORE creating instances
        robot_cfgs = [load_yaml(robot_path)["robot_cfg"] for robot_path in self.robot_config_paths]
        robot_sphere_counts_split = [calculate_robot_sphere_count(robot_cfg) for robot_cfg in robot_cfgs]
        robot_sphere_counts = [split[0] + split[1] for split in robot_sphere_counts_split]
        robot_sphere_counts_valid = [split[0] for split in robot_sphere_counts_split]
        
        # Pre-populate robot context BEFORE robot creation
        for i in range(self.n_robots):
            # Get MPC config values for this specific robot
            mpc_config = load_yaml(self.mpc_config_paths[i])
            
            # Check if this robot has DynamicObsCost enabled
            has_dynamic_obs_cost = (
                "cost" in mpc_config and 
                "custom" in mpc_config["cost"] and 
                "arm_base" in mpc_config["cost"]["custom"] and 
                "dynamic_obs_cost" in mpc_config["cost"]["custom"]["arm_base"]
            )
            
            if has_dynamic_obs_cost:
                n_obstacle_spheres = sum(robot_sphere_counts[j] for j in self.col_pred_with[i])
                
                # Populate robot context directly in env_topics[i]
                self.env_topics[i]["env_id"] = 0
                self.env_topics[i]["robot_id"] = i
                self.env_topics[i]["robot_pose"] = self.X_robots[i].tolist()
                self.env_topics[i]["n_obstacle_spheres"] = n_obstacle_spheres
                self.env_topics[i]["n_own_spheres"] = robot_sphere_counts[i]
                self.env_topics[i]["horizon"] = mpc_config["model"]["horizon"]
                self.env_topics[i]["n_rollouts"] = mpc_config["mppi"]["num_particles"]
                self.env_topics[i]["col_pred_with"] = self.col_pred_with[i]
                
                # Add new fields for sparse sphere functionality
                self.env_topics[i]["mpc_config_paths"] = self.mpc_config_paths
                self.env_topics[i]["robot_config_paths"] = self.robot_config_paths
                self.env_topics[i]["robot_sphere_counts"] = robot_sphere_counts_split
        
        # Create robot instances
        for i in range(self.n_robots):
            robot = ArmMpc(
                robot_cfgs[i],
                self.my_world,
                self.usd_help,
                env_id=0,
                robot_id=i,
                p_R=self.X_robots[i][:3],
                q_R=self.X_robots[i][3:],
                p_T_R=np.array(self.X_targets_R[i][:3]),
                q_T_R=np.array(self.X_targets_R[i][3:]),
                target_color=self.target_colors[i],
                plot_costs=self.plot_costs[i],
                override_particle_file=self.mpc_config_paths[i],
                n_coll_spheres=robot_sphere_counts[i],
                n_coll_spheres_valid=robot_sphere_counts_valid[i],
                use_col_pred=len(self.col_pred_with[i]) > 0
            )
            
            self.robots.append(robot)
        
        # Add environment obstacles to robot world models
        for obstacle in self.env_obstacles:
            for i, world_model in enumerate(self.robot_world_models):
                world_model_idx = obstacle.add_to_world_model(world_model, self.X_robots[i])
                print(f"Obstacle {obstacle.name} added to world model {world_model_idx}")
        
        # Initialize robots (after pre-population)
        self.robot_idx_lists = [None for _ in range(self.n_robots)]
        self.ccheckers = []
        
        for i, robot in enumerate(self.robots):
            # Set robots in initial joint configuration
            self.robot_idx_lists[i] = [robot.robot.get_dof_index(x) for x in robot.j_names]
            if self.robot_idx_lists[i] is None:
                raise RuntimeError(f"Failed to get DOF indices for robot {i}")
            
            idx_list = self.robot_idx_lists[i]
            assert idx_list is not None
            robot.init_joints(idx_list)
            
            # Init robot mpc solver
            robot.init_solver(self.robot_world_models[i], self.robots_collision_caches[i], MPC_DT, DEBUG)
            robot.robot._articulation_view.initialize()
            
            # Get initialized collision checker
            checker = robot.get_cchecker()
            self.ccheckers.append(checker)
        
        # Register collision checkers with obstacles
        for obstacle in self.env_obstacles:
            obstacle.register_ccheckers(self.ccheckers)
    
    def setup_ros_nodes(self):
        """Create ROS2 nodes for each robot"""
        for i, robot in enumerate(self.robots):
            node = RobotNode(
                robot_id=i,
                robot=robot,
                col_pred_with=self.col_pred_with[i],
                world=self.my_world,
                env_obstacles=self.env_obstacles,
                tensor_args=self.tensor_args
            )
            
            self.robot_nodes.append(node)
            self.executor.add_node(node)
    
    def run(self):
        """Run the multi-robot system"""
        print("Starting multi-robot MPC system with ROS2...")
        
        # Wait for simulation to be ready
        from projects_root.utils.issacsim import wait_for_playing
        wait_for_playing(self.my_world, simulation_app, autoplay=True)
        
        try:
            # Run ROS2 executor in separate thread
            executor_thread = threading.Thread(target=self.executor.spin)
            executor_thread.daemon = True
            executor_thread.start()
            
            # Main simulation loop
            while simulation_app.is_running():
                self.my_world.step(render=True)
                time.sleep(0.01)  # Small delay to prevent excessive CPU usage
                
        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Clean shutdown"""
        self.executor.shutdown()
        for node in self.robot_nodes:
            node.destroy_node()
        rclpy.shutdown()
        simulation_app.close()


def resolve_meta_config_path(robot_model: str) -> str:
    """Resolves the meta-configuration paths to the absolute paths."""
    root_path = "projects_root/projects/dynamic_obs/dynamic_obs_predictor/cfgs/meta_cfgs"
    return os.path.join(root_path, f"{robot_model}.yml")


def main():
    """Main entry point"""
    # Example robot configurations
    input_args = ['franka', 'ur5e', 'franka']
    meta_config_paths = [resolve_meta_config_path(robot_model) for robot_model in input_args]
    
    system = MultiRobotMPCSystem(meta_config_paths)
    system.run()


if __name__ == "__main__":
    main() 