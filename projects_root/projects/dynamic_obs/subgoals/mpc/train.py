import argparse
import torch

def curobo_isaac_sim_apps_initiation_protocol(parser:argparse.ArgumentParser):
    """
    This function is used to initiate the Isaac Sim simulation environment.
    It is used in all curobo apps that use Isaac Sim.
    """
    parser.add_argument("--headless_mode",type=str,default=None,help="To run headless, use one of [native, websocket], webrtc might not work.",)
    parser.add_argument("--use_isaac_sim", type=bool, default=True, help="Use Isaac Sim simulation (default) instead of real robot")
    
    from omni.isaac.kit import SimulationApp
    args = parser.parse_args()
    simulation_app = None
    if args.use_isaac_sim:
        try:
            import isaacsim
        except ImportError:
            pass    
        simulation_app = SimulationApp(
        {
            "headless": args.headless_mode is not None,
            "width": "1920",
            "height": "1080",
        })    
    return args, simulation_app



# initialize parser as follows:
parser = argparse.ArgumentParser(description="argument parser for curobo apps")
# define as much as argmuments as you need and read them
parser.add_argument("--num_episodes", type=int, default=1000)
parser.add_argument("--max_steps", type=int, default=500)
parser.add_argument("--num_obstacles", type=int, default=3)
parser.add_argument("--model_path", type=str, default="models")
parser.add_argument("--robot_ip", type=str, help="IP address of the real robot if not using simulation")


a = torch.zeros(4, device="cuda:0") # "this is necessary to allow isaac sim to use this torch instance" (said at simple_stacking.py example)
# call the standard initiation protocol, get your args updated with headless mode as well as the simulation app object (or None if using real robot)
# (this is the standard initiation protocol for curobo apps that use isaac sim. Should be called at the beginning of the main function, on any  app).
args, simulation_app = curobo_isaac_sim_apps_initiation_protocol(parser)
assert simulation_app is not None, "Simulation app not initialized"


# Only now you can import the necessary omni.isaac.core modules you need (note: linter errors are expected here, that's normal. TODO: fix this when time allows.)
from omni.isaac.core import World  
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.objects import cuboid
from omni.isaac.core.robots import Robot

# Next, import the helper required helper functions, and call the extension adding function.
from projects_root.utils.helper import add_extensions, add_robot_to_scene
add_extensions(simulation_app, args.headless_mode)

# Import your necessary curobo modules
from curobo.util.usd_helper import UsdHelper #  see projects_root/notes/understanding_the_usd_interface.txt
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig, Cuboid
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.state import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig

# Next, add other all your personal required imports as you would normally do
import numpy as np
import os
from typing import Dict, List, Optional, Tuple, Union, Any, TYPE_CHECKING
from dataclasses import dataclass, field
import random
from abc import ABC, abstractmethod
from rl import RLMPCAgent, RLMPCConfig

#####################################################################################################################
# Now we can start with our actual code
#####################################################################################################################

class RobotStateProvider(ABC):
    """Abstract base class for getting robot state from different sources."""
    
    @abstractmethod
    def get_joint_state(self) -> JointState:
        """Get the current joint state of the robot."""
        pass

class IsaacSimStateProvider(RobotStateProvider):
    """Get robot state from Isaac Gym simulation."""
    
    def __init__(self, robot: "Robot", world: "World"):
        """Initialize with Isaac Gym robot instance.
        
        Args:
            robot: Isaac Sim Robot instance
            world: Isaac Sim World instance
        """
        self.robot = robot
        self.world = world
        self.tensor_args = TensorDeviceType()
        
        # Get initial state to validate joint names
        robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
        self.joint_names = robot_cfg["kinematics"]["cspace"]["joint_names"]  # Use robot config joint names
        
    def get_joint_state(self):
        """Get joint state from Isaac Gym simulation."""
        # Make sure simulation is playing
        if not self.world.is_playing():
            self.world.play()
        
        # Get current joint state from Isaac Sim - this returns JointsState
        sim_js = self.robot.get_joints_state()
        
        if sim_js is None or np.any(np.isnan(sim_js.positions)):
            print("Warning: Invalid joint state from Isaac Sim, using zeros")
            # Return zero state as fallback
            return ArticulationAction(
                positions=np.zeros(len(self.joint_names)),
                velocities=np.zeros(len(self.joint_names)),
                joint_names=self.robot.dof_names
            )
        
        return sim_js

class RealRobotStateProvider(RobotStateProvider):
    """Get robot state from a real robot."""
    
    def __init__(self, robot_interface):
        """Initialize with real robot interface."""
        self.robot_interface = robot_interface
        self.tensor_args = TensorDeviceType()
    
    def get_joint_state(self) -> JointState:
        """Get joint state from real robot."""
        # This would be implemented based on your specific robot interface
        # For example, using ROS2 controllers or direct robot APIs
        real_js = self.robot_interface.get_joint_states()  # Implement this based on your robot
        return JointState(
            position=self.tensor_args.to_device(real_js.position),
            velocity=self.tensor_args.to_device(real_js.velocity),
            acceleration=self.tensor_args.to_device(real_js.velocity) * 0.0,  # Typically not provided by real robots
            jerk=self.tensor_args.to_device(real_js.velocity) * 0.0,  # Typically not provided by real robots
            joint_names=real_js.joint_names
        )

@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Training parameters
    num_episodes: int = 1000
    max_steps_per_episode: int = 500
    eval_interval: int = 10
    save_interval: int = 100
    
    # Environment parameters
    num_obstacles: int = 3
    obstacle_speed_range: List[float] = field(default_factory=lambda: [-0.1, 0.1])  # Min/max speed in m/s
    workspace_bounds: List[List[float]] = field(default_factory=lambda: [[-0.5, 0.5], [-0.5, 0.5], [0.0, 0.5]])  # x, y, z bounds
    
    # Paths
    model_save_path: str = "models"
    robot_config: str = "franka.yml"
    obstacle_velocities: Dict[str, List[float]] = field(default_factory=lambda: {})  # Store velocities per obstacle
    
    # Robot state source
    use_isaac_sim: bool = True  # If False, uses real robot

def setup_environment(world_config: WorldConfig, 
                    config: TrainingConfig,
                    mpc_solver: MpcSolver) -> Tuple[WorldConfig, List[Cuboid]]:
    """Set up the simulation environment with obstacles."""
    obstacles = []
    config.obstacle_velocities.clear()  # Reset velocities
    
    # Create random obstacles
    for i in range(config.num_obstacles):
        # Random position within workspace bounds
        pos = [
            random.uniform(config.workspace_bounds[0][0], config.workspace_bounds[0][1]),  # x bounds
            random.uniform(config.workspace_bounds[1][0], config.workspace_bounds[1][1]),  # y bounds
            random.uniform(config.workspace_bounds[2][0], config.workspace_bounds[2][1])   # z bounds
        ]
        # Fixed orientation (upright)
        quat = [1, 0, 0, 0]
        
        # Create cuboid obstacle
        name = f"obstacle_{i}"
        obstacle = Cuboid(
            name=name,
            pose=pos + quat,  # Combine position and orientation
            dims=[0.1, 0.1, 0.3]  # Fixed size
        )
        
        # Store velocity separately if enabled
        if config.obstacle_speed_range is not None:
            min_speed = config.obstacle_speed_range[0]
            max_speed = config.obstacle_speed_range[1]
            config.obstacle_velocities[name] = [
                random.uniform(min_speed, max_speed),
                random.uniform(min_speed, max_speed),
                0.0  # No vertical movement
            ]
        
        obstacles.append(obstacle)
    
    # Update world configuration with obstacles
    world_config = WorldConfig(cuboid=obstacles)
    if mpc_solver.world_collision is not None:
        mpc_solver.world_collision.load_collision_model(world_config)
    
    return world_config, obstacles

def update_obstacles(world_config: WorldConfig, 
                    obstacles: List[Cuboid], 
                    config: TrainingConfig,
                    dt: float,
                    mpc_solver: MpcSolver):
    """Update obstacle positions based on their velocities."""
    updated = False
    for obstacle in obstacles:
        if obstacle.name in config.obstacle_velocities:
            vel = config.obstacle_velocities[obstacle.name]
            if obstacle.pose is not None:
                # Update position based on velocity
                new_position = [
                    obstacle.pose[0] + vel[0] * dt,
                    obstacle.pose[1] + vel[1] * dt,
                    obstacle.pose[2] + vel[2] * dt
                ]
                # Keep within workspace bounds
                new_position[0] = max(-0.5, min(0.5, new_position[0]))  # x bounds
                new_position[1] = max(-0.5, min(0.5, new_position[1]))  # y bounds
                new_position[2] = max(0.0, min(0.5, new_position[2]))   # z bounds
                
                # Keep orientation unchanged
                obstacle.pose = new_position + obstacle.pose[3:7]
                updated = True
            
    # Update world collision checker with new obstacle poses only if needed
    if updated and mpc_solver.world_collision is not None:
        # Create new world config with updated obstacles
        world_config = WorldConfig(cuboid=obstacles)
        mpc_solver.world_collision.load_collision_model(world_config)

def draw_points(rollouts: torch.Tensor):
    if rollouts is None:
        return

    # Third Party
    from omni.isaac.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()
    N = 100
    # if draw.get_num_points() > 0:
    draw.clear_points()
    cpu_rollouts = rollouts.cpu().numpy()
    b, h, _ = cpu_rollouts.shape
    point_list = []
    colors = []
    for i in range(b):
        # get list of points:
        point_list += [
            (cpu_rollouts[i, j, 0], cpu_rollouts[i, j, 1], cpu_rollouts[i, j, 2]) for j in range(h)
        ]
        colors += [(1.0 - (i + 1.0 / b), 0.3 * (i + 1.0 / b), 0.0, 0.1) for _ in range(h)]
    sizes = [10.0 for _ in range(b * h)]
    draw.draw_points(point_list, colors, sizes)



def init_world(my_world: World):
    while simulation_app.is_running():
        for _ in range(10):
            my_world.step(render=True)
        return

def block_until_play_button_pressed(my_world: World):
    while simulation_app.is_running():
        if args.autoplay:
            my_world.play()
        my_world.step(render=True)
        if my_world.is_playing():
            return
      
            
def train(config: TrainingConfig, robot_state_provider: RobotStateProvider, usd_help: UsdHelper):
    """Main training loop."""
    
    # Initialize MPC solver
    mpc, mpc_current_state, goal = init_mpc_solver(robot_cfg, world_cfg)
    goal_buffer = mpc.setup_solve_single(goal, 1)
    mpc.update_goal(goal_buffer)
    mpc_result = mpc.step(mpc_current_state, max_attempts=2)
        
    
    usd_help.load_stage(my_world.stage)
    init_world(my_world) 
   
    cmd_state_full = None 
    
    # Initialize RL agent
    rl_config = RLMPCConfig()
    rl_agent = RLMPCAgent(rl_config, mpc)
    
    # Training loop
    episode_rewards = []
    step = 0
    
    block_until_play_button_pressed(my_world)
    while simulation_app.is_running():
    
        init_curobo = False # TODO this is found in all examples, why?

        for episode in range(config.num_episodes):
            print(f"Starting episode {episode}")
            
            # Reset environment and get initial state
            # world_config, obstacles = setup_environment(world_cfg, config, mpc_solver)
            
            # Create current state from retract config
            # current_state = JointState.from_position(retract_cfg, joint_names=joint_names)
            current_joint_state = current_state
            
            # # Set random goal
            # goal_position = tensor_args.to_device(torch.tensor([
            #     random.uniform(config.workspace_bounds[0][0], config.workspace_bounds[0][1]),
            #     random.uniform(config.workspace_bounds[1][0], config.workspace_bounds[1][1]),
            #     random.uniform(config.workspace_bounds[2][0], config.workspace_bounds[2][1])
            # ], dtype=torch.float32)).unsqueeze(0)
            
            # goal_quaternion = tensor_args.to_device(torch.tensor(
            #     [1.0, 0.0, 0.0, 0.0], dtype=torch.float32
            # )).unsqueeze(0)
            
            original_goal = Pose(position=goal_position, quaternion=goal_quaternion)
            
            # Episode variables
            episode_reward = 0
            states = []
            actions = []
            rewards = []
            dones = []
            next_states = []
            
            # Episode loop
            
            for step in range(config.max_steps_per_episode):
                step_index = my_world.current_time_step_index
                if step_index <= 2:
                    my_world.reset()
                    idx_list = [robot.get_dof_index(x) for x in j_names]
                    robot.set_joint_positions(default_config, idx_list)

                    robot._articulation_view.set_max_efforts(
                        values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
                    )

                if not init_curobo:
                    init_curobo = True
                step += 1
                step_index = step
                # Get state representation
                state_repr = rl_agent.get_state_representation(
                    robot_state=current_joint_state,
                    original_goal=original_goal,
                    current_goal=original_goal,
                    world_config=world_config
                )
                
                # Select action and update goal
                current_goal = rl_agent.select_action(state_repr)
                mpc_goal = Goal(
                    current_state=current_joint_state,
                    goal_pose=current_goal
                )
                goal_buffer = mpc.setup_solve_single(mpc_goal, num_seeds=1)
                mpc.update_goal(goal_buffer)
                
                # Run MPC step
                mpc_result = mpc_solver.step(current_joint_state, max_attempts=2)
                
                # Get next state from robot
                sim_js = robot_state_provider.get_joint_state()
                js_names = robot_state_provider.robot.dof_names
                
                # Create next state using robot state and MPC result
                next_js = JointState(
                    position=tensor_args.to_device(torch.tensor(sim_js.positions)),
                    velocity=tensor_args.to_device(torch.tensor(sim_js.velocities)).unsqueeze(0) * 0.0,  # Add batch dimension
                    acceleration=tensor_args.to_device(torch.tensor(sim_js.velocities)).unsqueeze(0) * 0.0,
                    jerk=tensor_args.to_device(torch.tensor(sim_js.velocities)).unsqueeze(0) * 0.0,
                    joint_names=js_names
                )
                next_js = next_js.get_ordered_joint_state(mpc_solver.rollout_fn.joint_names)
                
                # Create next state
                next_joint_state = JointState.from_position(
                    tensor_args.to_device(next_js.position).unsqueeze(0),
                    joint_names=mpc_solver.rollout_fn.joint_names
                )
                
                # Update obstacles
                update_obstacles(world_config, obstacles, config, dt=0.02, mpc_solver=mpc_solver)
                
                # Compute reward and done flag
                reward = rl_agent.compute_reward(
                    robot_state=next_joint_state,
                    original_goal=original_goal,
                    current_goal=current_goal,
                    collision_checker=None,
                    done=(step == config.max_steps_per_episode - 1)
                )
                done = (step == config.max_steps_per_episode - 1)
                
                # Store transition
                states.append(state_repr)
                actions.append(current_goal)
                rewards.append(reward)
                dones.append(done)
                next_states.append(next_joint_state)
                
                episode_reward += reward
                
                # Update current state for next iteration
                current_joint_state = next_joint_state
                
                if done:
                    break
            
            # Update RL agent
            update_info = rl_agent.update(states, actions, rewards, dones, next_states)
            episode_rewards.append(episode_reward)
            
            # Logging
            print(f"Episode {episode} - Reward: {episode_reward:.2f}")
            print(f"Actor Loss: {update_info['actor_loss']:.4f}")
            print(f"Critic Loss: {update_info['critic_loss']:.4f}")
            
            # Save model periodically
            if episode % config.save_interval == 0:
                os.makedirs(config.model_save_path, exist_ok=True)
                rl_agent.save(os.path.join(config.model_save_path, f"model_ep_{episode}.pt"))
            
            # Evaluation
            if episode % config.eval_interval == 0:
                eval_reward = evaluate(rl_agent, mpc_solver, config, robot_state_provider)
                print(f"Evaluation Reward: {eval_reward:.2f}")

def evaluate(rl_agent: RLMPCAgent, 
            mpc_solver: MpcSolver, 
            config: TrainingConfig,
            robot_state_provider: RobotStateProvider,
            num_eval_episodes: int = 5) -> float:
    """Evaluate the current policy."""
    eval_rewards = []
    tensor_args = TensorDeviceType()
    
    for _ in range(num_eval_episodes):
        # Reset environment
        world_config = WorldConfig()
        world_config, obstacles = setup_environment(world_config, config, mpc_solver)
        mpc_solver.update_world(world_config)
        
        # Get current state from robot
        sim_js = robot_state_provider.get_joint_state()
        js_names = robot_state_provider.robot.dof_names
        
        # Create JointState from sim_js
        cu_js = JointState(
            position=tensor_args.to_device(torch.tensor(sim_js.positions)),
            velocity=tensor_args.to_device(torch.tensor(sim_js.velocities)).unsqueeze(0) * 0.0,  # Add batch dimension
            acceleration=tensor_args.to_device(torch.tensor(sim_js.velocities)).unsqueeze(0) * 0.0,
            jerk=tensor_args.to_device(torch.tensor(sim_js.velocities)).unsqueeze(0) * 0.0,
            joint_names=js_names
        )
        cu_js = cu_js.get_ordered_joint_state(mpc_solver.rollout_fn.joint_names)
        
        # Create current state
        current_state = JointState.from_position(
            tensor_args.to_device(cu_js.position).unsqueeze(0),
            joint_names=mpc_solver.rollout_fn.joint_names
        )
        
        # Set random goal
        goal_position = tensor_args.to_device(torch.tensor([
            random.uniform(config.workspace_bounds[0][0], config.workspace_bounds[0][1]),
            random.uniform(config.workspace_bounds[1][0], config.workspace_bounds[1][1]),
            random.uniform(config.workspace_bounds[2][0], config.workspace_bounds[2][1])
        ], dtype=torch.float32)).unsqueeze(0)
        
        goal_quaternion = tensor_args.to_device(torch.tensor(
            [1.0, 0.0, 0.0, 0.0], dtype=torch.float32
        )).unsqueeze(0)
        
        original_goal = Pose(position=goal_position, quaternion=goal_quaternion)
        
        episode_reward = 0
        current_joint_state = current_state
        
        for step in range(config.max_steps_per_episode):
            # Get state representation
            state_repr = rl_agent.get_state_representation(
                robot_state=current_joint_state,
                original_goal=original_goal,
                current_goal=original_goal,
                world_config=world_config
            )
            
            # Select action with deterministic policy
            current_goal = rl_agent.select_action(state_repr, deterministic=True)
            
            # Create MPC goal and update
            mpc_goal = Goal(
                current_state=current_joint_state,
                goal_pose=current_goal
            )
            goal_buffer = mpc_solver.setup_solve_single(mpc_goal, num_seeds=1)
            mpc_solver.update_goal(goal_buffer)
            
            # Run MPC step
            mpc_result = mpc_solver.step(current_joint_state, max_attempts=2)
            
            # Get next state from robot
            sim_js = robot_state_provider.get_joint_state()
            js_names = robot_state_provider.robot.dof_names
            
            # Create next state using robot state and MPC result
            next_js = JointState(
                position=tensor_args.to_device(torch.tensor(sim_js.positions)),
                velocity=tensor_args.to_device(torch.tensor(sim_js.velocities)).unsqueeze(0) * 0.0,  # Add batch dimension
                acceleration=tensor_args.to_device(torch.tensor(sim_js.velocities)).unsqueeze(0) * 0.0,
                jerk=tensor_args.to_device(torch.tensor(sim_js.velocities)).unsqueeze(0) * 0.0,
                joint_names=js_names
            )
            next_js = next_js.get_ordered_joint_state(mpc_solver.rollout_fn.joint_names)
            
            # Create next state
            next_joint_state = JointState.from_position(
                tensor_args.to_device(next_js.position).unsqueeze(0),
                joint_names=mpc_solver.rollout_fn.joint_names
            )
            
            # Update obstacles
            update_obstacles(world_config, obstacles, config, dt=0.02, mpc_solver=mpc_solver)
            
            # Compute reward
            reward = rl_agent.compute_reward(
                robot_state=next_joint_state,
                original_goal=original_goal,
                current_goal=current_goal,
                collision_checker=None,
                done=(step == config.max_steps_per_episode - 1)
            )
            
            episode_reward += reward
            
            # Update current state for next iteration
            current_joint_state = next_joint_state
            
            if step == config.max_steps_per_episode - 1:
                break
            
        eval_rewards.append(episode_reward)
    
    return float(np.mean(eval_rewards))


# def Stage_wrapper(my_world: World) -> omni.isaac.core.pxr.Usd.Stage:
    # https://docs.omniverse.nvidia.com/kit/docs/pxr-usd-api/latest/pxr/Usd.html#
    # https://docs.isaacsim.omniverse.nvidia.com/latest/replicator_tutorials/tutorial_replicator_getting_started.html#custom-writer-and-annotators-with-multiple-cameras
    # https://docs.omniverse.nvidia.com/usd/latest/learn-openusd/terms/prim.html    

# def GetPrimAtPath_wrapper(stage, path):
    # # https://docs.omniverse.nvidia.com/kit/docs/pxr-usd-api/latest/pxr/Usd.html#pxr.Usd.Stage.GetPrimAtPath
    # return stage.GetPrimAtPath(path)

def SetDefaultPrim_wrapper(stage, prim):
    # https://docs.omniverse.nvidia.com/kit/docs/pxr-usd-api/latest/pxr/Usd.html#pxr.Usd.Stage.SetDefaultPrim
    stage.SetDefaultPrim(prim)

def DefinePrim_wrapper(stage, path, typeName):
    # https://docs.omniverse.nvidia.com/kit/docs/pxr-usd-api/latest/pxr/Usd.html#pxr.Usd.Stage.DefinePrim
    stage.DefinePrim(path, typeName)


def setup_stage(my_world: World):
    """
    Setup the stage for the simulation
    """
    stage = my_world.stage
    xform = DefinePrim_wrapper(stage, "/World", "Xform")
    SetDefaultPrim_wrapper(stage, xform)
    DefinePrim_wrapper(stage, "/curobo", "Xform")
    stage = my_world.stage
    return stage

def setup_world():
    # Create world
    my_world = World(stage_units_in_meters=1.0)
    setup_stage(my_world)
    my_world.scene.add_default_ground_plane()
    return my_world

def setup_robot_cfg(robot):
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]
    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]
    robot_cfg["kinematics"]["collision_sphere_buffer"] += 0.02
    
    return robot_cfg, j_names, default_config

def get_articulation_controller_wrapper(robot):
    """
    Overview
    
    Articulation controller is the low level controller that controls joint position, joint velocity, and joint effort in Isaac Sim. The articulation controller can be interfaced using Python and Omnigraph.

    https://docs.isaacsim.omniverse.nvidia.com/latest/robot_simulation/articulation_controller.html
    https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.core/docs/index.html#omni.isaac.core.controllers.ArticulationController  
    """
    return robot.get_articulation_controller()

def init_curobo_world_representation() -> WorldConfig:
    """
    WorldConfig is the way curobo describes the world.
    It is used to define the world for collision checking in curobo for example.
    This function initiales it with the collision table.
    """
    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_cfg_table.cuboid[0].pose[2] -= 0.04  # Adjust table height
    world_cfg1 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_mesh_world()
    world_cfg1.mesh[0].name += "_mesh"
    world_cfg1.mesh[0].pose[2] = -10.5  # Place mesh below ground

    world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh) # representation of the world for use in curobo
    return world_cfg


def make_moving_obstacle(world_cfg: WorldConfig, obstacle_type: str, position: List[float], size: List[float], color: List[float]):
    """
    Add an obstacle to the world.
    """
    def create_moving_obstacle(world, position, size=0.1, obstacle_type="cuboid", color=None, enable_physics=False, mass=1.0):
        
        """
        Create a moving obstacle in the simulation.
        
        Args:
            world: Isaac Sim world instance
            position: Initial position [x, y, z]
            size: Size of obstacle (diameter for sphere, side length for cube)
            obstacle_type: "cuboid" or "sphere"
            color: RGB color array (defaults to blue if None)
            enable_physics: If True, creates a physical obstacle that can collide and follow physics.
                        If False, creates a visual-only obstacle that moves without physics.
            mass: Mass in kg (only used if enable_physics=True)
        """
        
        
        def init_sphere_obstacle(world, position, size, color, enable_physics=False, mass=1.0):
            """
            Initialize a sphere obstacle.
            
            Args:
                world: Isaac Sim world instance
                position: Initial position [x, y, z]
                size: Diameter of sphere
                color: RGB color array
                enable_physics: If True, creates a physical obstacle that can collide and follow physics.
                            If False, creates a visual-only obstacle that moves without physics.
                mass: Mass in kg (only used if enable_physics=True)
            """
            from omni.isaac.core.objects import sphere
            if enable_physics:
                from omni.isaac.core.objects import DynamicSphere
                obstacle = world.scene.add(
                    DynamicSphere(
                        prim_path="/World/moving_obstacle",
                        name="moving_obstacle",
                        position=position,
                        radius=size/2,
                        color=color,
                        mass=mass,
                        density=0.9
                    )
                )
            else:
                obstacle = world.scene.add(
                    sphere.VisualSphere(
                        prim_path="/World/moving_obstacle",
                        name="moving_obstacle",
                        position=position,
                        radius=size/2,
                        color=color,
                    )
                )
            return obstacle
        def init_cube_obstacle(world, position, size, color, enable_physics=False, mass=1.0):
            """
            Initialize a cube obstacle.
            
            Args:
                world: Isaac Sim world instance
                position: Initial position [x, y, z]
                size: Side length of cube
                color: RGB color array
                enable_physics: If True, creates a physical obstacle that can collide and follow physics.
                            If False, creates a visual-only obstacle that moves without physics.
                mass: Mass in kg (only used if enable_physics=True)
                friction: Friction coefficient (only used if enable_physics=True)
                restitution: Bounciness coefficient (only used if enable_physics=True)
            """
            if enable_physics:
                from omni.isaac.core.objects import DynamicCuboid
                obstacle = world.scene.add(
                    DynamicCuboid( # https://docs.isaacsim.omniverse.nvidia.com/4.5.0/py/source/extensions/isaacsim.core.api/docs/index.html#isaacsim.core.api.objects.DynamicCuboid:~:text=Dynamic%20cuboids%20(Cube%20shape)%20have%20collisions%20(Collider%20API)%20and%20rigid%20body%20dynamics%20(Rigid%20Body%20API) 
                        prim_path="/World/moving_obstacle",
                        name="moving_obstacle",
                        position=position,
                        size=size,
                        color=color,
                        mass=mass,
                        density=0.9
                    )
                )
            else:
                obstacle = world.scene.add(
                    cuboid.VisualCuboid(
                        prim_path="/World/moving_obstacle",
                        name="moving_obstacle",
                        position=position,
                        size=size,
                        color=color,
                    )
                )
            return obstacle
        
        # debug print
        print(f"create_moving_obstacle enable_physics value: {enable_physics}")
        print(f"create_moving_obstacle enable_physics type: {type(enable_physics)}")
        
        if color is None:
            color = np.array([0.0, 0.0, 0.1])  # Default blue color
        if obstacle_type == "cuboid":
            return init_cube_obstacle(world, position, size, color, enable_physics, mass)
        elif obstacle_type == "sphere":
            return init_sphere_obstacle(world, position, size, color, enable_physics, mass)
    initial_pos = np.array(args.obstacle_initial_pos)
    # Add debug print
    print(f"main() enable_physics value: {args.enable_physics}")
    print(f"main() enable_physics type: {type(args.enable_physics)}")
    obstacle = create_moving_obstacle(
        my_world,
        initial_pos,
        args.obstacle_size,
        args.obstacle_type,
        np.array(args.obstacle_color),
        args.enable_physics,
        args.obstacle_mass
    )
    
    # Set up obstacle movement
    obstacle_velocity = np.array(args.obstacle_velocity)
    
    if args.enable_physics:
        # For physical obstacles, use Isaac Sim's physics engine
        obstacle.set_linear_velocity(obstacle_velocity)
    else:
        # For non-physical obstacles, manually update position
        current_position = initial_pos
        dt = 1.0/60.0  # Simulation timestep (60 Hz)

    # Add obstacle to CuRobo's collision checker
    if args.obstacle_type == "cuboid":
        moving_obstacle = Cuboid(
            name="moving_obstacle",
            pose=[initial_pos[0], initial_pos[1], initial_pos[2], 1.0, 0.0, 0.0, 0.0],
            dims=[args.obstacle_size, args.obstacle_size, args.obstacle_size],
        )
    else:  # sphere
        from curobo.geom.types import Sphere
        moving_obstacle = Sphere(
            name="moving_obstacle",
            pose=[initial_pos[0], initial_pos[1], initial_pos[2], 1.0, 0.0, 0.0, 0.0],
            radius=args.obstacle_size/2,
        )
    return moving_obstacle


def init_mpc_solver(robot_cfg, world_cfg: WorldConfig):

    """
    Initialize the MPC solver 
    """
    mpc_config = MpcSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        use_cuda_graph=True,  # Use CUDA graphs for faster execution
        use_cuda_graph_metrics=True,
        use_cuda_graph_full_step=False,
        self_collision_check=True,
        collision_checker_type=CollisionCheckerType.MESH,
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        use_mppi=True,  # Use Model Predictive Path Integral for optimization
        use_lbfgs=False,
        use_es=False,
        store_rollouts=True,  # Store trajectories for visualization
        step_dt=0.02,  # MPC timestep
    )

    mpc = MpcSolver(mpc_config)

    # Set up initial robot state and goal
    retract_cfg = mpc.rollout_fn.dynamics_model.retract_config.clone().unsqueeze(0)
    joint_names = mpc.rollout_fn.joint_names

    state = mpc.rollout_fn.compute_kinematics(
        JointState.from_position(retract_cfg, joint_names=joint_names)
    )
    current_state = JointState.from_position(retract_cfg, joint_names=joint_names)
    retract_pose = Pose(state.ee_pos_seq, quaternion=state.ee_quat_seq)
    goal = Goal(
        current_state=current_state,
        goal_state=JointState.from_position(retract_cfg, joint_names=joint_names),
        goal_pose=retract_pose,
    )

    
    return mpc, current_state, goal

if __name__ == "__main__":
    
    
    config = TrainingConfig(
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
        num_obstacles=args.num_obstacles,
        model_save_path=args.model_path,
        use_isaac_sim=args.use_isaac_sim
    )
    


    if config.use_isaac_sim:
        
        # Create a world with ground plane
        my_world = setup_world()
        stage = my_world.stage
        
        # Make a target to follow
        target = cuboid.VisualCuboid("/World/target",position=np.array([0.5, 0, 0.5]),orientation=np.array([0, 1, 0, 0]),color=np.array([1.0, 0, 0]),size=0.05)
       

        # Configure CuRobo logging and parameters
        setup_curobo_logger("warn")
        past_pose = None
        n_obstacle_cuboids = 30  # Number of collision boxes for obstacle approximation
        n_obstacle_mesh = 10     # Number of mesh triangles for obstacle approximation

        # warmup curobo instance
        usd_help = UsdHelper()
        # target_pose = None
        
        # ?
        tensor_args = TensorDeviceType()
        
        # Load robot configuration
        robot_cfg, j_names, default_config = setup_robot_cfg(args.robot)
        
        # Add robot to scene 
        robot, robot_prim_path = add_robot_to_scene(robot_cfg,my_world)
        
        # Get articulation controller: 
        articulation_controller = get_articulation_controller_wrapper(robot)

        # prepare world representation as used by curobo,
        world_cfg = init_curobo_world_representation()
        
        # create one moving obstacle and add it to curobo's world representation
        world_cfg.add_obstacle(make_moving_obstacle(world_cfg, args.obstacle_type, args.obstacle_initial_pos, args.obstacle_size, args.obstacle_color))
    

        train(config, state_provider, usd_help)
     


    #     # Set default joint positions
    #     idx_list = [robot.get_dof_index(x) for x in joint_names]
    #     robot.set_joint_positions(default_config, idx_list)
        
    #     # Set maximum joint efforts
    #     robot._articulation_view.set_max_efforts(
    #         values=np.array([5000 for i in range(len(idx_list))]), 
    #         joint_indices=idx_list
    #     )
        
    #     # Initialize robot and start simulation
    #     robot.initialize()
    #     my_world.play()
        
    #     # Create Isaac Gym state provider
    #     state_provider = IsaacSimStateProvider(robot, my_world)
        
    #     # Let simulation settle
    #     for _ in range(50):
    #         my_world.step(render=True)
    # else:
    #     if args.robot_ip is None:
    #         raise ValueError("Robot IP address must be provided when not using simulation")
        
    #     # Initialize real robot interface (implement based on your robot)
    #     from robot_interface import RealRobotInterface  # You need to implement this
    #     robot_interface = RealRobotInterface(args.robot_ip)
    #     state_provider = RealRobotStateProvider(robot_interface)
    
    # train(config, state_provider)





