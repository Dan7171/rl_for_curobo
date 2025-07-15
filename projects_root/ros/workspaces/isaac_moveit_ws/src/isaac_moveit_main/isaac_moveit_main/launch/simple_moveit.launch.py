#!/usr/bin/env python3
"""
Simple MoveIt launch file that manually launches components.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    """Generate simple launch description for MoveIt."""
    
    # Declare launch arguments
    robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='ur5',
        description='Name of the robot'
    )
    
    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': LaunchConfiguration('robot_description')
        }]
    )
    
    # MoveGroup
    move_group = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        name='move_group',
        output='screen',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('moveit_config'),
                'config',
                'move_group.yaml'
            ]),
            PathJoinSubstitution([
                FindPackageShare('moveit_config'),
                'config',
                'moveit_controllers.yaml'
            ]),
            PathJoinSubstitution([
                FindPackageShare('moveit_config'),
                'config',
                'kinematics.yaml'
            ]),
            PathJoinSubstitution([
                FindPackageShare('moveit_config'),
                'config',
                'joint_limits.yaml'
            ]),
            PathJoinSubstitution([
                FindPackageShare('moveit_config'),
                'config',
                'pilz_cartesian_limits.yaml'
            ]),
            PathJoinSubstitution([
                FindPackageShare('moveit_config'),
                'config',
                'ur5.srdf'
            ]),
            PathJoinSubstitution([
                FindPackageShare('moveit_config'),
                'config',
                'ur5.urdf'
            ])
        ]
    )
    
    # RViz
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=[
            '-d', PathJoinSubstitution([
                FindPackageShare('moveit_config'),
                'config',
                'moveit.rviz'
            ])
        ]
    )
    
    # Controller manager
    controller_manager = Node(
        package='controller_manager',
        executable='ros2_control_node',
        name='controller_manager',
        output='screen',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('moveit_config'),
                'config',
                'ros2_controllers.yaml'
            ])
        ]
    )
    
    # Spawners
    arm_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['arm_controller'],
        output='screen'
    )
    
    gripper_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['gripper_controller'],
        output='screen'
    )
    
    joint_state_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen'
    )
    
    return LaunchDescription([
        robot_name_arg,
        robot_state_publisher,
        move_group,
        rviz,
        controller_manager,
        arm_spawner,
        gripper_spawner,
        joint_state_spawner,
    ]) 