import argparse
import os
import sys
import subprocess
import tempfile
import yaml
from pathlib import Path
from xml.etree import ElementTree as ET
import time

try:
    from urdf_parser_py.urdf import URDF  # type: ignore
except ImportError as e:
    print("[moveit_config_autogen] Missing dependency 'urdf_parser_py'. Install with: pip install urdf_parser_py")
    raise e

__all__ = [
    "generate_moveit_config",
    "launch_move_group",
]

DEFAULT_CFG_ROOT = Path(__file__).parents[1] / "config" / "moveit_configs_for_testing" / "auto_gen"

def _extract_joint_data(robot: URDF):
    """Return lists of (name, limits) for movable joints."""
    joints = []
    for j in robot.joints:
        if j.type == "fixed":
            continue
        limit = j.limit if j.limit is not None else None
        joints.append((j.name, limit))
    return joints

def _create_srdf(robot_name: str, joint_names: list[str]) -> str:
    """Generate a minimal SRDF string with a single planning group including all movable joints."""
    robot_elem = ET.Element("robot", name=robot_name)
    group_elem = ET.SubElement(robot_elem, "group", name="manipulator")
    for jn in joint_names:
        ET.SubElement(group_elem, "joint", name=jn)
    return ET.tostring(robot_elem, encoding="unicode")

def _create_joint_limits_yaml(joint_entries):
    jl = {"joint_limits": {}}
    for name, lim in joint_entries:
        # Defaults if limits missing
        entry = {
            "has_position_limits": bool(lim and (lim.lower is not None) and (lim.upper is not None)),
            "min_position": float(lim.lower) if lim and lim.lower is not None else -3.14,
            "max_position": float(lim.upper) if lim and lim.upper is not None else 3.14,
            "has_velocity_limits": bool(lim and (lim.velocity is not None)),
            "max_velocity": float(lim.velocity) if lim and lim.velocity is not None else 1.0,
            "has_acceleration_limits": False,
        }
        jl["joint_limits"][name] = entry
    return yaml.dump(jl)

def generate_moveit_config(urdf_path: str, robot_name: str | None = None, tmp_root: str | None = None):
    """Generate minimal MoveIt config for *urdf_path*.

    Returns dictionary with keys:
        - config_dir: root of generated files
        - srdf_path
        - joint_limits_path
        - param_yaml_path (combined params for move_group)
    """
    urdf_path = os.path.abspath(urdf_path)
    if not os.path.isfile(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    robot = URDF.from_xml_file(urdf_path)
    if robot_name is None:
        robot_name = robot.name or "autogen_robot"

    joint_entries = _extract_joint_data(robot)
    joint_names = [n for n, _ in joint_entries]

    srdf_str = _create_srdf(robot_name, joint_names)
    joint_limits_yaml = _create_joint_limits_yaml(joint_entries)

    # Write temp files
    # Decide where to place generated config
    if tmp_root is not None:
        root_dir = Path(tmp_root)
    else:
        # Use package-local auto_gen directory so results remain persistent for testing
        root_dir = DEFAULT_CFG_ROOT

    root_dir.mkdir(parents=True, exist_ok=True)
    config_dir = root_dir / f"{robot_name}_moveit_config_{str(time.time())}" 
    config_dir.mkdir(exist_ok=True)

    # Save URDF (copy) + SRDF + yaml
    urdf_out = config_dir / f"{robot_name}.urdf"
    if urdf_out.resolve() != Path(urdf_path).resolve():
        # copy
        urdf_out.write_text(Path(urdf_path).read_text())
    srdf_out = config_dir / f"{robot_name}.srdf"
    srdf_out.write_text(srdf_str)

    jl_out = config_dir / "joint_limits.yaml"
    jl_out.write_text(joint_limits_yaml)

    # Compose parameter YAML consumable by move_group
    params = {
        "move_group": {
            "ros__parameters": {
                "robot_description": Path(urdf_out).read_text(),
                "robot_description_semantic": srdf_str,
                "robot_description_planning": yaml.safe_load(joint_limits_yaml),
                # Minimal planning pipeline config for OMPL
                "planning_pipelines": ["ompl"],
                "ompl": {
                    "planning_plugin": "ompl_interface/OMPLPlanner",
                    "request_adapters": "default_planner_request_adapters/AddTimeOptimalParameterization default_planner_request_adapters/FixWorkspaceBounds default_planner_request_adapters/FixStartStateBounds default_planner_request_adapters/FixStartStateCollision default_planner_request_adapters/FixStartStatePathConstraints",
                    "start_state_max_bounds_error": 0.1,
                },
            }
        }
    }
    param_yaml_path = config_dir / "move_group_params.yaml"
    param_yaml_path.write_text(yaml.dump(params))

    return {
        "config_dir": str(config_dir),
        "srdf_path": str(srdf_out),
        "joint_limits_path": str(jl_out),
        "param_yaml_path": str(param_yaml_path),
    }


def launch_move_group(param_yaml_path: str):
    """Launch move_group as a background subprocess with the given parameter YAML."""
    cmd = [
        "ros2",
        "run",
        "moveit_ros_move_group",
        "move_group",
        "--ros-args",
        "--params-file",
        param_yaml_path,
    ]
    print(f"[moveit_config_autogen] Launching move_group: {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def main():
    parser = argparse.ArgumentParser(description="Auto-generate minimal MoveIt config from URDF and optionally start move_group.")
    parser.add_argument("urdf", help="Path to URDF file")
    parser.add_argument("--robot-name", help="Override robot name (defaults to URDF name)")
    parser.add_argument("--launch", action="store_true", help="Launch move_group after generation")
    args = parser.parse_args()

    res = generate_moveit_config(args.urdf, robot_name=args.robot_name)
    print("Generated config in", res["config_dir"])

    if args.launch:
        proc = launch_move_group(res["param_yaml_path"])
        print("move_group PID:", proc.pid)
        print("Press Ctrl+C to stop.")
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()

if __name__ == "__main__":
    main() 