# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import PoseStamped
# from moveit2 import MoveGroupInterface

# class MotionPlanner(Node):
#     def __init__(self):
#         super().__init__('motion_planner')
#         self.move_group = MoveGroupInterface(node=self, joint_names=[], base_link_name="base_link", end_effector_name="ee_link", group_name="arm")

#     def send_goal(self):
#         # Define target pose (world frame)
#         pose = PoseStamped()
#         pose.header.frame_id = "world"
#         pose.pose.position.x = 0.5
#         pose.pose.position.y = 0.0
#         pose.pose.position.z = 0.5
#         pose.pose.orientation.w = 1.0

#         # Send goal to MoveIt
#         self.move_group.move_to_pose(pose, "ee_link")
#         self.move_group.wait_until_executed()

# def main():
#     rclpy.init()
#     node = MotionPlanner()
#     node.send_goal()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()


import rclpy
from rclpy.node import Node
from moveit_msgs.action import MoveGroup
from geometry_msgs.msg import Pose
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import Constraints, PositionConstraint, Constraints
from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rclpy.action import ActionClient


class MoveGroupClient(Node):
    def __init__(self):
        super().__init__('move_group_client')
        self._client = ActionClient(self, MoveGroup, '/move_action')
        self._client.wait_for_server()

    def send_goal(self):
        goal_msg = MoveGroup.Goal()

        # Set planning group
        goal_msg.request.group_name = 'arm'

        # Set planning time
        goal_msg.request.allowed_planning_time = 5.0

        # Set start state (optional but often required)
        joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_2_joint',
            'wrist_3_joint',
            'wrist_1_joint'
        ]
        joint_positions = [0.6236, -2.5956, 1.6396, 0.2071, 0.2581, -0.5722]
        goal_msg.request.start_state.joint_state.name = joint_names
        goal_msg.request.start_state.joint_state.position = joint_positions
        goal_msg.request.start_state.is_diff = False

        # Position constraint (box region around target)
        constraint = PositionConstraint()
        constraint.header.frame_id = 'world'
        constraint.link_name = 'ee_link'

        # Define a small box
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.001, 0.001, 0.001]  # tiny box

        constraint.constraint_region.primitives.append(box)

        pose = Pose()
        pose.position.x = 0.5
        pose.position.y = 0.0
        pose.position.z = 0.5
        pose.orientation.w = 1.0
        constraint.constraint_region.primitive_poses.append(pose)

        constraint.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(constraint)

        goal_msg.request.goal_constraints.append(constraints)

        # Planning options
        goal_msg.planning_options.plan_only = False
        goal_msg.planning_options.planning_scene_diff.is_diff = True

        # Send and handle result
        future = self._client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            return

        self.get_logger().info('Goal accepted, waiting for result...')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result
        self.get_logger().info(f"Planning result error code: {result.error_code.val}")


def main():
    rclpy.init()
    node = MoveGroupClient()
    node.send_goal()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
