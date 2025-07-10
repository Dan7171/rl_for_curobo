#!/usr/bin/env python3
"""
Launch file for MoveIt2 Isaac Sim integration.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Generate launch description for MoveIt2 Isaac Sim integration."""
    
    # Declare arguments
    robot_arg = DeclareLaunchArgument(
        "robot",
        default_value="default_robot",
        description="Robot configuration name"
    )
    
    config_arg = DeclareLaunchArgument(
        "config",
        default_value="",
        description="Path to robot configuration file"
    )
    
    planner_arg = DeclareLaunchArgument(
        "planner",
        default_value="RRTConnect",
        description="Motion planner name"
    )
    
    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz",
        default_value="false",
        description="Use RViz for goal setting"
    )
    
    headless_arg = DeclareLaunchArgument(
        "headless",
        default_value="false",
        description="Run Isaac Sim in headless mode"
    )
    
    real_hardware_arg = DeclareLaunchArgument(
        "real_hardware",
        default_value="false",
        description="Send commands to real hardware"
    )
    
    # Get package directory
    pkg_dir = get_package_share_directory("moveit2_isaac_sim_integration")

    # Path to simulation runner script
    simulation_runner_script = os.path.join(pkg_dir, "..", "..", "scripts", "simulation_runner.py")

    # Create simulation runner node
    simulation_runner = ExecuteProcess(
        cmd=[
            simulation_runner_script,
            "--robot", LaunchConfiguration("robot"),
            "--config", LaunchConfiguration("config"),
            "--planner", LaunchConfiguration("planner"),
        ],
        condition=IfCondition(LaunchConfiguration("use_rviz")),
        output="screen"
    )

    # Alternative without RViz
    simulation_runner_no_rviz = ExecuteProcess(
        cmd=[
            simulation_runner_script,
            "--robot", LaunchConfiguration("robot"),
            "--config", LaunchConfiguration("config"),
            "--planner", LaunchConfiguration("planner"),
        ],
        condition=IfCondition(LaunchConfiguration("use_rviz")),
        output="screen"
    )
    
    # Static transform publisher (world to robot base)
    static_tf_world_to_base = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher_world_to_base",
        arguments=["0", "0", "0", "0", "0", "0", "world", "base_link"],
        output="log"
    )
    
    # Optional: Launch RViz
    rviz_config_path = os.path.join(pkg_dir, "config", "moveit_rviz_config.rviz")
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config_path],
        condition=IfCondition(LaunchConfiguration("use_rviz")),
        output="log"
    )
    
    return LaunchDescription([
        # Arguments
        robot_arg,
        config_arg,
        planner_arg,
        use_rviz_arg,
        headless_arg,
        real_hardware_arg,
        
        # Info
        LogInfo(
            msg=[
                "Starting MoveIt2 Isaac Sim Integration with:",
                "\n  Robot: ", LaunchConfiguration("robot"),
                "\n  Planner: ", LaunchConfiguration("planner"),
                "\n  Use RViz: ", LaunchConfiguration("use_rviz")
            ]
        ),
        
        # Nodes
        static_tf_world_to_base,
        simulation_runner,
        rviz_node,
    ]) 