from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_demo_launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration

def launch_setup(context, *args, **kwargs):
    # Use hardcoded robot name since this config is specifically for ur5
    robot_name = "ur5"
    moveit_config = MoveItConfigsBuilder(robot_name, package_name="moveit_config").to_moveit_configs()
    return generate_demo_launch(moveit_config).entities

def generate_launch_description():
    robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='ur5',
        description='Name of the robot (e.g., ur5, ur10, panda)'
    )
    return LaunchDescription([
        robot_name_arg,
        OpaqueFunction(function=launch_setup)
    ])
