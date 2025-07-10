from moveit2_utils import MoveIt2Planner, create_pose_from_list
from config_loader import ConfigLoader
import rclpy, time

def main():
    rclpy.init()
    cfg = ConfigLoader().get_robot_config("default_robot")
    planner = MoveIt2Planner(cfg)

    # Wait until Isaac Sim sends its first joint-state
    while planner.current_joint_states is None:
        rclpy.spin_once(planner, timeout_sec=0.1)

    # Example: send the robot to a fixed Cartesian goal
    target = create_pose_from_list([0.5, 0.0, 0.4])
    planner.set_target_pose(cfg.get_end_effectors()[0]["name"], target)
    plan = planner.plan_to_targets()
    if plan:
        for res in plan.values():
            planner.execute_plan(res, use_isaac_direct=True)

    print("Planner node running — use Ctrl-C to stop")
    rclpy.spin(planner)

if __name__ == "__main__":
    main()