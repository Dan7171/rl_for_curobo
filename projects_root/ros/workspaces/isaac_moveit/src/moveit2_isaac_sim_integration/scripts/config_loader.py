"""
Configuration loader for MoveIt2 Isaac Sim integration.
Loads robot configurations from YAML files and provides easy access to settings.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np


class RobotConfig:
    """
    A class to handle robot configuration data.
    """
    
    def __init__(self, config_data: Dict[str, Any]):
        """
        Initialize with configuration data dictionary.
        
        Args:
            config_data: Dictionary containing robot configuration
        """
        self.config = config_data
        self.name = config_data.get("name", "unknown_robot")
        # Will be filled by ConfigLoader so that external processes can
        # reference the YAML file that defined this RobotConfig.
        self.config_file_path: str | None = None
        
    def get_isaac_sim_config(self) -> Dict[str, Any]:
        """Get Isaac Sim specific configuration."""
        return self.config.get("isaac_sim", {})
        
    def get_moveit2_config(self) -> Dict[str, Any]:
        """Get MoveIt2 specific configuration."""
        return self.config.get("moveit2", {})
        
    def get_ros2_topics(self) -> Dict[str, str]:
        """Get ROS2 topic configuration."""
        return self.config.get("ros2_topics", {})
        
    def get_planning_config(self) -> Dict[str, Any]:
        """Get planning configuration."""
        return self.config.get("planning", {})
        
    def get_simulation_config(self) -> Dict[str, Any]:
        """Get simulation configuration."""
        return self.config.get("simulation", {})
        
    def get_end_effectors(self) -> List[Dict[str, Any]]:
        """Get list of end effector configurations."""
        return self.get_moveit2_config().get("end_effectors", [])
        
    def get_joint_names(self) -> List[str]:
        """Get list of joint names."""
        return self.get_moveit2_config().get("joint_names", [])
        
    def get_home_position(self) -> List[float]:
        """Get home position for all joints."""
        return self.get_moveit2_config().get("home_position", [])
        
    def get_planning_group(self) -> str:
        """Get main planning group name."""
        return self.get_moveit2_config().get("planning_group", "manipulator")
        
    def get_default_planner(self) -> str:
        """Get default planner name."""
        return self.get_planning_config().get("default_planner", "RRTConnect")
        
    def get_isaac_stage_path(self) -> str:
        """Get Isaac Sim stage path."""
        return self.get_isaac_sim_config().get("stage_path", "/Robot")
        
    def get_isaac_usd_path(self) -> str:
        """Get Isaac Sim USD path."""
        return self.get_isaac_sim_config().get("usd_path", "")
        
    def get_robot_position(self) -> List[float]:
        """Get robot position in Isaac Sim."""
        return self.get_isaac_sim_config().get("position", [0.0, 0.0, 0.0])
        
    def get_robot_orientation(self) -> List[float]:
        """Get robot orientation in Isaac Sim."""
        return self.get_isaac_sim_config().get("orientation", [0.0, 0.0, 0.0, 0.0])
        
    def get_urdf_path(self) -> str:
        """Get URDF file path."""
        return self.config.get("urdf_path", "")
        
    def get_target_visualization_size(self) -> float:
        """Get target visualization cuboid size."""
        return self.get_simulation_config().get("target_visualization_size", 0.1)
        
    def get_target_change_interval(self) -> int:
        """Get target change interval in simulation steps."""
        return self.get_simulation_config().get("target_change_interval", 100)
        
    def get_target_pose_tolerance(self) -> float:
        """Get target pose tolerance for replanning."""
        return self.get_simulation_config().get("target_pose_tolerance", 0.01)


class ConfigLoader:
    """
    Configuration loader for robot configurations.
    """
    
    def __init__(self, config_file_path: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_file_path: Path to configuration file. If None, uses default.
        """
        if config_file_path is None:
            # Default path relative to package
            package_dir = Path(__file__).parent.parent
            config_file_path = package_dir / "config" / "robot_config.yaml"
            
        self.config_file_path = Path(config_file_path)
        self.configs = {}
        
        if self.config_file_path.exists():
            self.load_configs()
        else:
            raise FileNotFoundError(f"Config file not found: {self.config_file_path}")
    
    def load_configs(self):
        """Load all configurations from the YAML file."""
        try:
            with open(self.config_file_path, 'r') as file:
                all_configs = yaml.safe_load(file)
                
            # Convert each configuration to RobotConfig objects
            for config_name, config_data in all_configs.items():
                rc = RobotConfig(config_data)
                # Expose original YAML file path so external tools (simulation_runner)
                # can forward it when spawning sub-processes.
                rc.config_file_path = str(self.config_file_path)
                self.configs[config_name] = rc
                
        except Exception as e:
            raise RuntimeError(f"Error loading config file: {e}")
    
    def get_robot_config(self, robot_name: str = "default_robot") -> RobotConfig:
        """
        Get robot configuration by name.
        
        Args:
            robot_name: Name of the robot configuration
            
        Returns:
            RobotConfig object
        """
        if robot_name not in self.configs:
            available_configs = list(self.configs.keys())
            raise ValueError(f"Robot config '{robot_name}' not found. Available configs: {available_configs}")
        
        return self.configs[robot_name]
    
    def list_available_robots(self) -> List[str]:
        """List all available robot configurations."""
        return list(self.configs.keys())
    
    def validate_config(self, robot_name: str) -> bool:
        """
        Validate a robot configuration.
        
        Args:
            robot_name: Name of the robot configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            config = self.get_robot_config(robot_name)
            
            # Check required fields
            required_fields = [
                "name",
                "isaac_sim",
                "moveit2",
                "ros2_topics"
            ]
            
            for field in required_fields:
                if field not in config.config:
                    print(f"Missing required field: {field}")
                    return False
            
            # Check end effectors
            end_effectors = config.get_end_effectors()
            if not end_effectors:
                print("No end effectors configured")
                return False
            
            # Check joint names
            joint_names = config.get_joint_names()
            if not joint_names:
                print("No joint names configured")
                return False
            
            # Check home position matches joint count
            home_pos = config.get_home_position()
            if len(home_pos) != len(joint_names):
                print(f"Home position length ({len(home_pos)}) doesn't match joint count ({len(joint_names)})")
                return False
            
            print(f"Configuration '{robot_name}' is valid")
            return True
            
        except Exception as e:
            print(f"Error validating config: {e}")
            return False


def create_default_config_from_urdf(urdf_path: str, robot_name: str = "custom_robot") -> Dict[str, Any]:
    """
    Create a default configuration dictionary from a URDF file.
    This is a basic implementation that would need to be expanded for full URDF parsing.
    
    Args:
        urdf_path: Path to URDF file
        robot_name: Name for the robot configuration
        
    Returns:
        Dictionary with default configuration
    """
    # Attempt to parse the URDF so we can auto-fill joint information.
    try:
        from urdf_parser_py.urdf import URDF  # type: ignore

        robot = URDF.from_xml_file(urdf_path)

        # Collect movable joints (exclude fixed)
        joint_names = []
        limits = {}
        for joint in robot.joints:
            if joint.type == "fixed":
                continue
            joint_names.append(joint.name)
            if joint.limit is not None:
                limits[joint.name] = [float(joint.limit.lower), float(joint.limit.upper)]

        # Default home pose = mid-point of limits if available else zeros
        home_position = []
        for name in joint_names:
            if name in limits:
                lower, upper = limits[name]
                home_position.append((lower + upper) / 2.0)
            else:
                home_position.append(0.0)

        print(f"Parsed {len(joint_names)} movable joints from URDF")

    except Exception as e:
        print(f"Warning: Failed to parse URDF ({e}). Falling back to empty joint list")
        joint_names = []
        home_position = []

    config = {
        "name": robot_name,
        "urdf_path": urdf_path,

        # Leave usd_path empty so that start_sim.py triggers URDF import
        "isaac_sim": {
            "stage_path": f"/{robot_name}",
            "usd_path": "",  # auto-import
            "position": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0, 0.0],
        },

        "moveit2": {
            "planning_group": "manipulator",
            "end_effectors": [
                {
                    "name": "end_effector",
                    "planning_group": "manipulator",
                    "link_name": "end_effector_link",
                    "reference_frame": "base_link",
                    "default_target_pose": {
                        "position": [0.5, 0.0, 0.5],
                        "orientation": [0.0, 0.0, 0.0, 1.0],
                    },
                }
            ],
            "joint_names": joint_names,
            "home_position": home_position,
        },

        "ros2_topics": {
            "isaac_joint_commands": "/isaac_joint_commands",
            "isaac_joint_states": "/isaac_joint_states",
            "moveit_joint_trajectory": "/joint_trajectory_controller/joint_trajectory",
        },

        "planning": {
            "default_planner": "RRTConnect",
            "planning_time": 5.0,
            "max_velocity_scaling_factor": 0.1,
            "max_acceleration_scaling_factor": 0.1,
        },

        "simulation": {
            "step_size": 0.01,
            "target_change_interval": 100,
            "target_pose_tolerance": 0.01,
            "target_visualization_size": 0.1,
        },
    }

    return config


def main():
    """Test function for configuration loader."""
    try:
        # Load configurations
        loader = ConfigLoader()
        
        # List available robots
        robots = loader.list_available_robots()
        print(f"Available robot configurations: {robots}")
        
        # Test each configuration
        for robot_name in robots:
            print(f"\nTesting configuration: {robot_name}")
            config = loader.get_robot_config(robot_name)
            
            print(f"  Name: {config.name}")
            print(f"  Planning group: {config.get_planning_group()}")
            print(f"  End effectors: {len(config.get_end_effectors())}")
            print(f"  Joint count: {len(config.get_joint_names())}")
            print(f"  Default planner: {config.get_default_planner()}")
            
            # Validate configuration
            is_valid = loader.validate_config(robot_name)
            print(f"  Valid: {is_valid}")
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 