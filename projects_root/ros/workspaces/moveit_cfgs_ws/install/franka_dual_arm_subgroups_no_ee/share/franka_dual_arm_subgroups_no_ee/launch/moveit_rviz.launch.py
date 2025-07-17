from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_moveit_rviz_launch


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder("franka_dual_arm", package_name="franka_dual_arm_subgroups_no_ee").to_moveit_configs()
    return generate_moveit_rviz_launch(moveit_config)
