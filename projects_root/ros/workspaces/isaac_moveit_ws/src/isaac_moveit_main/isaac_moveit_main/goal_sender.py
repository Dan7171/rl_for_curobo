import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint, MotionPlanRequest, PlanningOptions, RobotState
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

class MultiArmPlanner(Node):
    def __init__(self):
        super().__init__('multi_arm_move_group_client')

        self.client = ActionClient(self, MoveGroup, '/move_group')

        # Wait until server is ready
        print("waiting for server")
        # self.client.wait_for_server()
        while not self.client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('Waiting for MoveGroup action server...')

        # Send the goal
        print("sending goal")
        self.send_goal()

    def send_goal(self):
        goal = MoveGroup.Goal()
        goal.request.group_name = 'all_arms'  # 👈 Replace with your group name
        goal.request.num_planning_attempts = 1
        goal.request.allowed_planning_time = 5.0

        # --- Start State (must set!) ---
        goal.request.start_state = RobotState()
        goal.request.start_state.joint_state.name = [
            'left_panda_joint1', 'left_panda_joint2', 'left_panda_joint3', 'left_panda_joint4', 'left_panda_joint5', 'left_panda_joint6','left_panda_joint7',
            'right_panda_joint1', 'right_panda_joint2', 'right_panda_joint3', 'right_panda_joint4', 'right_panda_joint5', 'right_panda_joint6','right_panda_joint7'
        ]
        goal.request.start_state.joint_state.position = [0.0] * len(goal.request.start_state.joint_state.name)  # Adjust if needed

        # --- Constraints for LEFT arm ---
        left_constraint = self.make_position_constraint(
            link_name='left_ee_link',  # 👈 Your left arm's end-effector
            x=0.4, y=0.2, z=0.4
        )

        # --- Constraints for RIGHT arm ---
        right_constraint = self.make_position_constraint(
            link_name='right_ee_link',  # 👈 Your right arm's end-effector
            x=0.4, y=-0.2, z=0.4
        )

        goal.request.goal_constraints = [Constraints()]
        goal.request.goal_constraints[0].position_constraints.append(left_constraint)
        goal.request.goal_constraints[0].position_constraints.append(right_constraint)

        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = True  # Only plan, don't execute

        self.future = self.client.send_goal_async(goal, feedback_callback=self.feedback_callback)
        self.future.add_done_callback(self.goal_response_callback)

    def make_position_constraint(self, link_name, x, y, z):
        pc = PositionConstraint()
        pc.link_name = link_name
        pc.target_point_offset.x = 0.0
        pc.target_point_offset.y = 0.0
        pc.target_point_offset.z = 0.0

        pc.constraint_region.primitives.append(SolidPrimitive(
            type=SolidPrimitive.BOX,
            dimensions=[0.01, 0.01, 0.01]
        ))

        pose = PoseStamped()
        pose.header.frame_id = 'world'
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.w = 1.0

        pc.constraint_region.primitive_poses.append(pose)
        pc.weight = 1.0

        return pc

    def feedback_callback(self, feedback):
        self.get_logger().info(f"Feedback: {feedback}")

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f"Result: {result}")

def main(args=None):
    rclpy.init(args=args)
    print("rclpy initialized")
    node = MultiArmPlanner()
    print("node created")
    rclpy.spin(node)
    print("spinning")
    node.destroy_node()
    print("node destroyed")
    rclpy.shutdown()
