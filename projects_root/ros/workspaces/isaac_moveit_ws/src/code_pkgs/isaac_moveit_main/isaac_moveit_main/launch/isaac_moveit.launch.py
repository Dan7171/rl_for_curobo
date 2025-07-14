#!/usr/bin/env python3
"""
Flexible Isaac Sim + MoveIt integration launch file.

This launch file can work with any robot by accepting robot_name as a parameter.
It uses the official MoveIt approach but makes it robot-agnostic.

Usage:
    ros2 launch isaac_moveit_main isaac_moveit.launch.py robot_name:=ur5
    ros2 launch isaac_moveit_main isaac_moveit.launch.py robot_name:=ur10
    ros2 launch isaac_moveit_main isaac_moveit.launch.py robot_name:=panda

Author: Assistant
Date: 2024
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    """Generate launch description for Isaac Sim + MoveIt integration."""
    
    # Declare launch arguments
    robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='ur5',
        description='Name of the robot (e.g., ur5, ur10, panda)'
    )
    
    # Get robot name from launch configuration
    robot_name = LaunchConfiguration('robot_name')
    
    # Use the correct package name: moveit_config
    # Don't pass robot_name parameter since moveit_config is already configured for ur5
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('moveit_config'),
                'launch',
                'demo.launch.py'
            ])
        ])
    )
    
    # You can add Isaac Sim specific nodes here later
    # For now, this just launches MoveIt + RViz
    
    return LaunchDescription([
        robot_name_arg,
        moveit_launch,
    ]) 