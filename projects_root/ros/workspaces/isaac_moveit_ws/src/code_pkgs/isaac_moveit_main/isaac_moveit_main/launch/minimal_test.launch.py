#!/usr/bin/env python3
"""
Minimal test launch file for Isaac Sim + MoveIt integration.
This launch file only starts essential components to test basic functionality.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    """Generate minimal launch description for testing."""
    
    # Declare launch arguments
    robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='ur5',
        description='Name of the robot (e.g., ur5, ur10, panda)'
    )
    
    # Get robot name from launch configuration
    robot_name = LaunchConfiguration('robot_name')
    
    # Launch only the move_group (no RViz, no sensors)
    move_group_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ur5_moveit_config'),
                'launch',
                'move_group.launch.py'
            ])
        ]),
        launch_arguments={
            'robot_name': robot_name,
        }.items()
    )
    
    # Launch controller manager
    controller_manager = Node(
        package='controller_manager',
        executable='ros2_control_node',
        name='controller_manager',
        output='screen',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('ur5_moveit_config'),
                'config',
                'ros2_controllers.yaml'
            ])
        ]
    )
    
    # Launch spawner for arm controller
    arm_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['arm_controller'],
        output='screen'
    )
    
    # Launch spawner for joint state broadcaster
    joint_state_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen'
    )
    
    return LaunchDescription([
        robot_name_arg,
        move_group_launch,
        controller_manager,
        arm_spawner,
        joint_state_spawner,
    ]) 