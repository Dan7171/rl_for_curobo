#!/usr/bin/env python3
"""Simple listener printing first plan waypoint for each robot."""
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class PlanListener(Node):
    def __init__(self, max_robots: int = 10):
        super().__init__("plan_listener")
        for rid in range(max_robots):
            self.create_subscription(String, f"/robot_{rid}/plan", self._make_cb(rid), 10)
        self.get_logger().info("Plan listener started (waiting for /robot_<id>/plan)")

    def _make_cb(self, rid):
        def _cb(msg: String):
            try:
                data = json.loads(msg.data)
                first = data.get("task_space", {}).get("spheres", {}).get("p", [[0,0,0]])[0]
                self.get_logger().info(f"robot_{rid} first point: {first}")
            except Exception as e:
                self.get_logger().warning(f"Decode error for robot_{rid}: {e}")
        return _cb


def main(argv=None):
    rclpy.init(args=argv)
    node = PlanListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main() 