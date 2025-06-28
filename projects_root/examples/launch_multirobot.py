#!/usr/bin/env python3
"""
ROS2 Launch file for Multi-Robot MPC System

Usage:
    ros2 launch projects_root/examples/launch_multirobot.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
import os

def generate_launch_description():
    """Generate launch description for multi-robot MPC system"""
    
    # Declare launch arguments
    robot_configs_arg = DeclareLaunchArgument(
        'robot_configs',
        default_value='franka,ur5e,franka',
        description='Comma-separated list of robot configurations'
    )
    
    visualize_arg = DeclareLaunchArgument(
        'visualize',
        default_value='true',
        description='Enable visualization'
    )
    
    # Get launch configurations
    robot_configs = LaunchConfiguration('robot_configs')
    visualize = LaunchConfiguration('visualize')
    
    # Create launch description
    ld = LaunchDescription()
    
    # Add launch arguments
    ld.add_action(robot_configs_arg)
    ld.add_action(visualize_arg)
    
    # Launch the main multi-robot system
    multirobot_node = Node(
        package='rl_for_curobo',  # Your package name
        executable='mpc_async_multirobot_ros.py',
        name='multirobot_mpc_system',
        output='screen',
        parameters=[{
            'robot_configs': robot_configs,
            'visualize': visualize
        }]
    )
    
    ld.add_action(multirobot_node)
    
    # Optional: Launch RViz for visualization
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(os.path.dirname(__file__), 'multirobot_config.rviz')],
        condition=lambda context: LaunchConfiguration('visualize').perform(context) == 'true'
    )
    
    ld.add_action(rviz_node)
    
    return ld 