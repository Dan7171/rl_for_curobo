#!/usr/bin/env python3
import rclpy, sys, time
from rclpy.node           import Node
from sensor_msgs.msg      import JointState
from geometry_msgs.msg    import PoseStamped
from moveit_msgs.srv      import GetMotionPlan
from moveit_msgs.msg      import MotionPlanRequest, Constraints, PositionConstraint
from shape_msgs.msg       import SolidPrimitive
from builtin_interfaces.msg import Duration

# --- helper: build pose goal as Constraints ----------------------------------
def pose_goal(frame, link, xyz):
    pc = PositionConstraint()
    pc.header.frame_id = frame
    pc.link_name       = link

    # a 1-cm radius sphere around desired XYZ
    sphere = SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.01])
    pc.constraint_region.primitives.append(sphere)
    pc.constraint_region.primitive_poses.append(
        PoseStamped().pose  # default = identity
    )
    pc.target_point_offset.x, pc.target_point_offset.y, pc.target_point_offset.z = xyz
    c = Constraints()
    c.position_constraints.append(pc)
    return c

# ---------------------------------------------------------------------------
class ServicePlanner(Node):
    def __init__(self):
        super().__init__("service_planner")

        self.cli = self.create_client(GetMotionPlan, "/plan_kinematic_path")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("waiting for move_group service…")

        # Isaac-Sim command publisher
        self.cmd_pub = self.create_publisher(JointState, "/isaac_joint_commands", 10)

        self.plan_and_send()

    def plan_and_send(self):
        req = GetMotionPlan.Request()
        mpr = MotionPlanRequest()
        mpr.group_name = "panda_arm"         # change to your planning group
        mpr.num_planning_attempts = 1
        mpr.allowed_planning_time = 2.0

        # fill start-state = current (empty RobotState means “use move_group’s latest”)
        # fill goal
        mpr.goal_constraints.append( pose_goal("panda_link0", "panda_link8",
                                               xyz=[0.5, 0.0, 0.4]) )
        req.motion_plan_request = mpr

        self.get_logger().info("Calling service…")
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if not future.result():
            self.get_logger().error("Planning failed")
            return
        traj = future.result().motion_plan_response.trajectory.joint_trajectory

        # stream every point to Isaac Sim
        self.get_logger().info(f"Executing {len(traj.points)} points")
        t0 = time.time()
        for pt in traj.points:
            msg = JointState()
            msg.name     = traj.joint_names
            msg.position = pt.positions
            msg.velocity = pt.velocities
            self.cmd_pub.publish(msg)

            dt = pt.time_from_start.sec + pt.time_from_start.nanosec*1e-9
            time.sleep(max(0.0, dt - (time.time()-t0)))

def main():
    rclpy.init()
    ServicePlanner()
    rclpy.spin(rclpy.get_default_context())

if __name__ == "__main__":
    main()