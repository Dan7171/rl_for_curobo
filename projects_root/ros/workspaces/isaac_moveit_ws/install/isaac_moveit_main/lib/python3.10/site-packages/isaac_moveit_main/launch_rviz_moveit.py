#!/usr/bin/env python3
"""
Launch RViz with MoveIt integration.
This script launches RViz with the MoveIt plugin and necessary components.
"""

import rclpy
from rclpy.node import Node
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    """Generate launch description for RViz with MoveIt."""
    
    # Declare launch arguments
    robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='ur5',
        description='Name of the robot (e.g., ur5, ur10, panda)'
    )
    
    # Get robot name from launch configuration
    robot_name = LaunchConfiguration('robot_name')
    
    # Launch the full demo (includes RViz)
    demo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ur5_moveit_config'),
                'launch',
                'demo.launch.py'
            ])
        ]),
        launch_arguments={
            'robot_name': robot_name,
        }.items()
    )
    
    return LaunchDescription([
        robot_name_arg,
        demo_launch,
    ])


def main():
    """Main function to launch RViz with MoveIt."""
    print("Launching RViz with MoveIt integration...")
    
    # For now, just print a message and suggest using the launch file directly
    print("This script is a wrapper for the demo launch.")
    print("You can also run: ros2 launch isaac_moveit_main isaac_moveit.launch.py")
    print("Or: ros2 launch ur5_moveit_config demo.launch.py")
    
    # Return success
    return 0


if __name__ == '__main__':
    main() 