#!/usr/bin/env python3
"""
Multi-arm robot configuration generator for CuRobo.

This utility generates robot configurations for arbitrary numbers of arms
from individual arm URDF files and configurations.
"""

import os
import yaml
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ArmConfig:
    """Configuration for a single arm in the multi-arm system."""
    arm_id: str  # e.g., "left", "right", "arm_0", etc.
    urdf_path: str  # Path to single arm URDF
    base_position: List[float]  # [x, y, z] position offset
    base_orientation: List[float]  # [x, y, z, w] quaternion
    joint_prefix: str  # Prefix for joint names (e.g., "left_", "right_")
    link_prefix: str  # Prefix for link names
    ee_link_name: str  # End-effector link name for this arm
    
    
class MultiArmConfigGenerator:
    """Generate multi-arm robot configurations from individual arm configs."""
    
    def __init__(self, output_dir: str = "generated_configs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_multi_arm_urdf(self, arms: List[ArmConfig], system_name: str) -> str:
        """Generate a URDF file for multi-arm system from individual arm URDFs.
        
        Args:
            arms: List of arm configurations
            system_name: Name for the multi-arm system
            
        Returns:
            Path to generated URDF file
        """
        # Create base URDF structure matching working dual-arm exactly
        root = ET.Element("robot", name=system_name)
        root.set("xmlns:xacro", "http://www.ros.org/wiki/xacro")
        
        # Add comment like working dual-arm
        comment = ET.Comment(f" Base link for the {system_name} system ")
        root.append(comment)
        
        # Add empty base link (like working dual-arm - no geometry)
        base_link = ET.SubElement(root, "link", name="base_link")
        
        # Process each arm using simplified merge like dual-arm
        for arm_config in arms:
            self._merge_arm_urdf_simple(root, arm_config)
            
        # Write URDF file
        urdf_path = self.output_dir / f"{system_name}.urdf"
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ", level=0)
        tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
        
        return str(urdf_path)
        
    def _merge_arm_urdf_simple(self, root_robot: ET.Element, arm_config: ArmConfig):
        """Merge an individual arm URDF into the multi-arm URDF following dual-arm pattern exactly."""
        # Load the arm URDF
        arm_tree = ET.parse(arm_config.urdf_path)
        arm_root = arm_tree.getroot()
        
        # Add comment for this arm assembly like dual-arm
        comment = ET.Comment(f" {arm_config.arm_id.title()} arm assembly ")
        root_robot.append(comment)
        
        # Find the root link of this arm (first link with geometry)
        arm_root_link = self._find_arm_root_link(arm_root)
        
        # Handle connection differently based on whether arm root is base_link
        if arm_root_link == 'base_link':
            # For robots where base_link is the root, add connection from main base_link to renamed arm base_link
            base_joint = ET.SubElement(root_robot, "joint", 
                                     name=f"{arm_config.arm_id}_arm_fixed", 
                                     type="fixed")
            
            # Set arm position/orientation
            origin = ET.SubElement(base_joint, "origin")
            pos = " ".join(map(str, arm_config.base_position))
            origin.set("rpy", "0 0 0")
            origin.set("xyz", pos)
            
            # Connect main base_link to renamed arm base_link
            joint_parent = ET.SubElement(base_joint, "parent", link="base_link")
            joint_child = ET.SubElement(base_joint, "child", 
                                      link=f"{arm_config.link_prefix}base_link")
            axis = ET.SubElement(base_joint, "axis", xyz="0 0 0")
        else:
            # Add joint connecting arm to base (like dual-arm pattern for non-base_link roots)
            base_joint = ET.SubElement(root_robot, "joint", 
                                     name=f"{arm_config.arm_id}_arm_fixed", 
                                     type="fixed")
            
            # Set arm position/orientation
            origin = ET.SubElement(base_joint, "origin")
            pos = " ".join(map(str, arm_config.base_position))
            origin.set("rpy", "0 0 0")
            origin.set("xyz", pos)
            
            # Parent and child links
            joint_parent = ET.SubElement(base_joint, "parent", link="base_link")
            joint_child = ET.SubElement(base_joint, "child", 
                                      link=f"{arm_config.link_prefix}{arm_root_link}")
            axis = ET.SubElement(base_joint, "axis", xyz="0 0 0")
        
        # Copy all links and joints from arm URDF with prefixes and fix mesh paths
        for element in arm_root:
            if element.tag in ["link", "joint"]:
                element_name = element.get("name", "")
                
                # Skip redundant joints that would conflict with our arm_fixed joints
                skip_redundant_joints = ['panda_fixed', 'arm_fixed', 'world_fixed', 'mount_fixed']
                
                # Handle base_link differently based on arm root
                if arm_root_link == 'base_link':
                    # For base_link roots, skip original base_link that's not the arm root, but include the arm root base_link
                    skip_base_elements = ['world', 'root', 'mount']  # Don't skip base_link
                else:
                    # For non-base_link roots, skip base_link as usual
                    skip_base_elements = ['base_link', 'world', 'root', 'mount']
                
                if (element.tag == "link" and element_name in skip_base_elements) or \
                   (element.tag == "joint" and element_name in skip_redundant_joints):
                    continue
                
                # Clone the element
                new_element = ET.SubElement(root_robot, element.tag, element.attrib)
                
                # Update name with prefix
                if "name" in new_element.attrib:
                    if element.tag == "link":
                        new_element.set("name", f"{arm_config.link_prefix}{element.get('name')}")
                    elif element.tag == "joint":
                        new_element.set("name", f"{arm_config.joint_prefix}{element.get('name')}")
                
                # Copy all child elements and fix mesh paths
                for child in element:
                    new_child = self._copy_element_recursive_with_mesh_fix(child, arm_config)
                    
                    # Update parent/child references in joints
                    if element.tag == "joint":
                        if child.tag == "parent" and "link" in child.attrib:
                            parent_link = child.get('link')
                            # Always prefix parent link references
                            new_child.set("link", f"{arm_config.link_prefix}{parent_link}")
                        elif child.tag == "child" and "link" in child.attrib:
                            new_child.set("link", f"{arm_config.link_prefix}{child.get('link')}")
                    
                    new_element.append(new_child)
    
    def _find_arm_root_link(self, urdf_root: ET.Element) -> str:
        """Find the actual root link of an arm URDF (should be base_link if it has geometry)."""
        # Strategy: For single-arm URDFs, the robot's base_link is usually the actual root
        
        # First, check if base_link exists and has geometry - if so, use it
        base_link_elem = urdf_root.find(".//link[@name='base_link']")
        if base_link_elem is not None:
            has_visual = base_link_elem.find('.//visual') is not None
            has_collision = base_link_elem.find('.//collision') is not None
            if has_visual or has_collision:
                return 'base_link'
        
        # If base_link doesn't have geometry, find first link with geometry connected from base_link
        base_candidates = ['base_link', 'world', 'root', 'mount']
        
        for joint in urdf_root.findall('.//joint'):
            parent = joint.find('parent')
            child = joint.find('child')
            
            if parent is not None and child is not None:
                parent_link = parent.get('link')
                child_link = child.get('link')
                
                # If parent is a base-like link, check if child has geometry
                if parent_link in base_candidates and child_link:
                    child_link_elem = urdf_root.find(f".//link[@name='{child_link}']")
                    if child_link_elem is not None:
                        # Check if this link has visual or collision geometry
                        has_visual = child_link_elem.find('.//visual') is not None
                        has_collision = child_link_elem.find('.//collision') is not None
                        if has_visual or has_collision:
                            return child_link
        
        # Fallback: Find first link with geometry that's not a base-like link
        for link in urdf_root.findall('.//link'):
            link_name = link.get('name')
            if link_name and link_name not in base_candidates:
                has_visual = link.find('.//visual') is not None
                has_collision = link.find('.//collision') is not None
                if has_visual or has_collision:
                    return link_name
        
        # Final fallback: just return first non-base link
        for link in urdf_root.findall('.//link'):
            link_name = link.get('name')
            if link_name and link_name not in base_candidates:
                return link_name
                
        raise ValueError("No suitable arm root link found in URDF")
    
    def _copy_element_recursive(self, element: ET.Element) -> ET.Element:
        """Recursively copy an XML element."""
        new_element = ET.Element(element.tag, element.attrib)
        new_element.text = element.text
        new_element.tail = element.tail
        
        for child in element:
            new_element.append(self._copy_element_recursive(child))
            
        return new_element
    
    def _copy_element_recursive_with_mesh_fix(self, element: ET.Element, arm_config: ArmConfig) -> ET.Element:
        """Recursively copy an XML element and fix mesh paths to match dual-arm structure."""
        new_element = ET.Element(element.tag, element.attrib)
        new_element.text = element.text
        new_element.tail = element.tail
        
        # Fix mesh paths for visual and collision elements
        if element.tag == "mesh" and "filename" in element.attrib:
            filename = element.get("filename")
            if filename:
                # Extract robot folder name from arm_config.urdf_path
                # e.g., "curobo/.../robot/kinova/kinova_gen3_7dof.urdf" -> "kinova"
                urdf_path_parts = arm_config.urdf_path.split('/')
                robot_folder = None
                for i, part in enumerate(urdf_path_parts):
                    if part == "robot" and i + 1 < len(urdf_path_parts):
                        robot_folder = urdf_path_parts[i + 1]
                        break
                
                if robot_folder:
                    # Build robot-agnostic relative path from test_generated_configs/
                    new_element.set("filename", f"../curobo/src/curobo/content/assets/robot/{robot_folder}/{filename}")
                else:
                    # Fallback to original filename if robot folder not found
                    new_element.set("filename", filename)
        
        for child in element:
            new_element.append(self._copy_element_recursive_with_mesh_fix(child, arm_config))
            
        return new_element
        
    def generate_curobo_config(self, arms: List[ArmConfig], system_name: str, 
                             urdf_path: str) -> str:
        """Generate CuRobo YAML configuration for multi-arm system.
        
        Args:
            arms: List of arm configurations
            system_name: Name for the multi-arm system
            urdf_path: Path to the generated URDF file
            
        Returns:
            Path to generated YAML config file
        """
        # Build joint names list
        joint_names = []
        for arm_config in arms:
            # Load single arm config to get joint names
            arm_joints = self._get_arm_joint_names(arm_config)
            joint_names.extend([f"{arm_config.joint_prefix}{joint}" for joint in arm_joints])
        
        # Build link names for end-effectors
        ee_link_names = [arm_config.ee_link_name for arm_config in arms]
        
        # Build collision link names
        collision_link_names = []
        for arm_config in arms:
            arm_links = self._get_arm_collision_links(arm_config)
            collision_link_names.extend([f"{arm_config.link_prefix}{link}" for link in arm_links])
        
        # Create configuration dictionary
        config = {
            "robot_cfg": {
                "kinematics": {
                    "use_usd_kinematics": False,
                    "urdf_path": f"robot/{system_name}/{os.path.basename(urdf_path)}",
                    "asset_root_path": f"robot/{system_name}",
                    "base_link": "base_link",
                    "ee_link": ee_link_names[0],  # Primary end-effector
                    "link_names": ee_link_names,  # All end-effectors for multi-arm
                    "collision_link_names": collision_link_names,
                    "collision_sphere_buffer": 0.004,
                    "use_global_cumul": True,
                    "cspace": {
                        "joint_names": joint_names,
                        "retract_config": [0.0] * len(joint_names),  # Default neutral pose
                        "null_space_weight": [1.0] * len(joint_names),
                        "cspace_distance_weight": [1.0] * len(joint_names),
                        "max_jerk": 500.0,
                        "max_acceleration": 15.0
                    }
                }
            }
        }
        
        # Write YAML file
        yaml_path = self.output_dir / f"{system_name}.yml"
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
            
        return str(yaml_path)
    
    def _get_arm_joint_names(self, arm_config: ArmConfig) -> List[str]:
        """Extract joint names from a single arm URDF by parsing the actual file."""
        try:
            tree = ET.parse(arm_config.urdf_path)
            root = tree.getroot()
            
            joint_names = []
            for joint in root.findall('.//joint'):
                joint_type = joint.get('type', '')
                if joint_type in ['revolute', 'prismatic', 'continuous']:
                    joint_names.append(joint.get('name'))
            
            return joint_names
        except Exception as e:
            print(f"Warning: Could not parse {arm_config.urdf_path}: {e}")
            # Fallback to empty list - will be handled upstream
            return []
    
    def _get_arm_collision_links(self, arm_config: ArmConfig) -> List[str]:
        """Extract collision link names from a single arm URDF by parsing the actual file."""
        try:
            tree = ET.parse(arm_config.urdf_path)
            root = tree.getroot()
            
            link_names = []
            for link in root.findall('.//link'):
                link_name = link.get('name')
                if link_name:
                    # Include links that have collision geometry or are structural
                    collision = link.find('.//collision')
                    visual = link.find('.//visual')
                    if collision is not None or visual is not None:
                        link_names.append(link_name)
            
            return link_names
        except Exception as e:
            print(f"Warning: Could not parse {arm_config.urdf_path}: {e}")
            # Fallback to empty list - will be handled upstream
            return []
    
    def generate_particle_mpc_config(self, num_arms: int, system_name: str) -> str:
        """Generate particle MPC configuration for K arms.
        
        Args:
            num_arms: Number of arms in the system
            system_name: Name for the multi-arm system
            
        Returns:
            Path to generated particle MPC config file
        """
        config = {
            "model": {
                "horizon": 30,
                "state_filter_cfg": {
                    "filter_coeff": {
                        "position": 0.1,
                        "velocity": 0.1,
                        "acceleration": 0.0
                    },
                    "enable": True
                },
                "dt_traj_params": {
                    "base_dt": 0.01,
                    "base_ratio": 0.5,
                    "max_dt": 0.04
                },
                "vel_scale": 1.0,
                "control_space": "ACCELERATION",
                "teleport_mode": False,
                "state_finite_difference_mode": "CENTRAL"
            },
            "cost": {
                "pose_cfg": {
                    "vec_weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    "run_vec_weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    "weight": [60, 300.0, 20, 20],
                    "vec_convergence": [0.0, 0.00],
                    "terminal": True,
                    "run_weight": 1.0,
                    "use_metric": True,
                    "num_arms": num_arms  # Enable multi-arm pose cost
                },
                "cspace_cfg": {
                    "weight": 1000.0,
                    "terminal": True,
                    "run_weight": 1.0
                },
                "bound_cfg": {
                    "weight": [5000.0, 5000.0, 5000.0, 5000.0],
                    "activation_distance": [0.1, 0.1, 0.1, 0.1],
                    "smooth_weight": [0.0, 50.0, 0.0, 0.0],
                    "run_weight_velocity": 0.0,
                    "run_weight_acceleration": 1.0,
                    "run_weight_jerk": 1.0,
                    "null_space_weight": [10.0]
                },
                "primitive_collision_cfg": {
                    "weight": 100000.0,
                    "use_sweep": True,
                    "sweep_steps": 4,
                    "classify": False,
                    "use_sweep_kernel": True,
                    "use_speed_metric": False,
                    "speed_dt": 0.1,
                    "activation_distance": 0.025
                },
                "self_collision_cfg": {
                    "weight": 50000.0,
                    "classify": False
                },
                "stop_cfg": {
                    "weight": 100.0,
                    "max_nlimit": 0.25
                }
            }
        }
        
        # Write YAML file
        yaml_path = self.output_dir / f"particle_mpc_{system_name}.yml"
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
            
        return str(yaml_path)


# Define available robot models with their configurations
ROBOT_MODELS = {
    "franka": {
        "urdf_path": "curobo/src/curobo/content/assets/robot/franka_description/franka_panda.urdf",
        "config_path": "franka.yml",
        "ee_link_suffix": "hand",
        "joint_names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"],
        "default_spacing": 1.0
    },
    "ur5e": {
        "urdf_path": "curobo/src/curobo/content/assets/robot/ur_description/ur5e.urdf", 
        "config_path": "ur5e.yml",
        "ee_link_suffix": "ee_link",
        "joint_names": ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
        "default_spacing": 0.8
    },
    "ur10e": {
        "urdf_path": "curobo/src/curobo/content/assets/robot/ur_description/ur10e.urdf",
        "config_path": "ur10e.yml", 
        "ee_link_suffix": "ee_link",
        "joint_names": ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
        "default_spacing": 1.2
    },
    "kinova": {
        "urdf_path": "curobo/src/curobo/content/assets/robot/kinova/kinova_gen3_7dof.urdf",
        "config_path": "kinova_gen3.yml",
        "ee_link_suffix": "end_effector_link", 
        "joint_names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"],
        "default_spacing": 0.9
    },
    "iiwa": {
        "urdf_path": "curobo/src/curobo/content/assets/robot/iiwa_allegro_description/iiwa.urdf",
        "config_path": "iiwa.yml",
        "ee_link_suffix": "iiwa_link_ee",
        "joint_names": ["iiwa_joint_1", "iiwa_joint_2", "iiwa_joint_3", "iiwa_joint_4", "iiwa_joint_5", "iiwa_joint_6", "iiwa_joint_7"],
        "default_spacing": 1.0
    },
    "jaco": {
        "urdf_path": "curobo/src/curobo/content/assets/robot/jaco/jaco_7s.urdf",
        "config_path": "jaco7.yml", 
        "ee_link_suffix": "j2s7s300_end_effector",
        "joint_names": ["j2s7s300_joint_1", "j2s7s300_joint_2", "j2s7s300_joint_3", "j2s7s300_joint_4", "j2s7s300_joint_5", "j2s7s300_joint_6", "j2s7s300_joint_7"],
        "default_spacing": 0.9
    },
    "tm12": {
        "urdf_path": "curobo/src/curobo/content/assets/robot/techman/tm_description/urdf/tm12-nominal.urdf",
        "config_path": "tm12.yml",
        "ee_link_suffix": "tool0",
        "joint_names": ["shoulder_1_joint", "shoulder_2_joint", "elbow_1_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
        "default_spacing": 1.1
    }
}


def create_k_arm_system(robot_model: str, num_arms: int, arm_spacing: Optional[float] = None, 
                       layout: str = "line") -> Tuple[List[ArmConfig], str]:
    """Create a K-arm system configuration for any robot model.
    
    Args:
        robot_model: Robot model name (franka, ur5e, ur10e, kinova, iiwa, jaco, tm12)
        num_arms: Number of arms to include
        arm_spacing: Spacing between arms in meters (uses default if None)
        layout: Layout pattern ("line", "circle", "grid")
        
    Returns:
        Tuple of (arm_configs, system_name)
    """
    if robot_model not in ROBOT_MODELS:
        raise ValueError(f"Unsupported robot model: {robot_model}. Available: {list(ROBOT_MODELS.keys())}")
    
    model_info = ROBOT_MODELS[robot_model]
    spacing = arm_spacing if arm_spacing is not None else model_info["default_spacing"]
    
    arms = []
    system_name = f"{robot_model}_{num_arms}_arm"
    
    # Generate arm positions based on layout
    positions = _generate_arm_positions(num_arms, spacing, layout)
    
    for i in range(num_arms):
        arm_config = ArmConfig(
            arm_id=f"arm_{i}",
            urdf_path=model_info["urdf_path"],
            base_position=positions[i],
            base_orientation=[0.0, 0.0, 0.0, 1.0],
            joint_prefix=f"arm_{i}_",
            link_prefix=f"arm_{i}_", 
            ee_link_name=f"arm_{i}_{model_info['ee_link_suffix']}"
        )
        arms.append(arm_config)
    
    return arms, system_name


def _generate_arm_positions(num_arms: int, spacing: float, layout: str) -> List[List[float]]:
    """Generate positions for arms based on layout pattern."""
    positions = []
    
    if layout == "line":
        # Arrange arms in a line along X-axis
        for i in range(num_arms):
            x = i * spacing - (num_arms - 1) * spacing / 2  # Center the line
            positions.append([x, 0.0, 0.0])
            
    elif layout == "circle":
        # Arrange arms in a circle
        import math
        radius = spacing * num_arms / (2 * math.pi)
        for i in range(num_arms):
            angle = 2 * math.pi * i / num_arms
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            positions.append([x, y, 0.0])
            
    elif layout == "grid":
        # Arrange arms in a grid pattern
        import math
        cols = math.ceil(math.sqrt(num_arms))
        for i in range(num_arms):
            row = i // cols
            col = i % cols
            x = col * spacing - (cols - 1) * spacing / 2
            y = row * spacing
            positions.append([x, y, 0.0])
            
    else:
        raise ValueError(f"Unsupported layout: {layout}. Available: line, circle, grid")
    
    return positions


def create_mixed_arm_system(arm_specs: List[Dict], system_name: str) -> Tuple[List[ArmConfig], str]:
    """Create a mixed-model K-arm system.
    
    Args:
        arm_specs: List of arm specifications, each containing:
                  {"model": "franka", "position": [x, y, z], "orientation": [x, y, z, w]}
        system_name: Name for the multi-arm system
        
    Returns:
        Tuple of (arm_configs, system_name)
    """
    arms = []
    
    for i, spec in enumerate(arm_specs):
        model = spec["model"]
        if model not in ROBOT_MODELS:
            raise ValueError(f"Unsupported robot model: {model}")
            
        model_info = ROBOT_MODELS[model]
        
        arm_config = ArmConfig(
            arm_id=f"arm_{i}_{model}",
            urdf_path=model_info["urdf_path"],
            base_position=spec.get("position", [0.0, 0.0, 0.0]),
            base_orientation=spec.get("orientation", [0.0, 0.0, 0.0, 1.0]),
            joint_prefix=f"arm_{i}_",
            link_prefix=f"arm_{i}_",
            ee_link_name=f"arm_{i}_{model_info['ee_link_suffix']}"
        )
        arms.append(arm_config)
    
    return arms, system_name


# Backward compatibility
def create_franka_k_arm_system(num_arms: int, arm_spacing: float = 1.0) -> Tuple[List[ArmConfig], str]:
    """Create a K-arm Franka system configuration (backward compatibility)."""
    return create_k_arm_system("franka", num_arms, arm_spacing)


if __name__ == "__main__":
    # Example usage: Generate various multi-arm systems
    generator = MultiArmConfigGenerator("generated_multi_arm_configs")
    
    print("Generating example multi-arm configurations...")
    
    examples = [
        # Single model systems
        ("franka", 4, "line"),
        ("ur5e", 3, "circle"),
        ("kinova", 5, "grid"),
        
        # Mixed model system
        ("mixed", None, None)
    ]
    
    for robot_model, num_arms, layout in examples:
        print(f"\n--- {robot_model.upper()} System ---")
        
        if robot_model == "mixed":
            # Create mixed arm system
            mixed_specs = [
                {"model": "franka", "position": [-1.0, 0.0, 0.0]},
                {"model": "ur5e", "position": [0.0, 0.0, 0.0]},
                {"model": "kinova", "position": [1.0, 0.0, 0.0]}
            ]
            arms, system_name = create_mixed_arm_system(mixed_specs, "mixed_example")
        else:
            # Create single model system
            arms, system_name = create_k_arm_system(robot_model, num_arms, layout=layout)
        
        # Generate URDF
        urdf_path = generator.generate_multi_arm_urdf(arms, system_name)
        print(f"✓ Generated URDF: {urdf_path}")
        
        # Generate CuRobo config
        config_path = generator.generate_curobo_config(arms, system_name, urdf_path)
        print(f"✓ Generated CuRobo config: {config_path}")
        
        # Generate particle MPC config
        mpc_config_path = generator.generate_particle_mpc_config(len(arms), system_name)
        print(f"✓ Generated MPC config: {mpc_config_path}")
    
    print(f"\n✓ All configurations generated in: generated_multi_arm_configs/")
    print(f"\nAvailable robot models: {list(ROBOT_MODELS.keys())}")
    print(f"Available layouts: line, circle, grid")
    print(f"\nTest with: python projects_root/tests/load_urdf.py --robot franka --num_arms 3") 