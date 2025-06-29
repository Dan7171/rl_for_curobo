#!/usr/bin/env python3
"""
Robot Controller Node
Provides convenient interfaces for controlling the Franka Panda robot
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
import math


class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        
        # Publishers for robot commands
        self.joint_positions_pub = self.create_publisher(
            JointState, '/robot/joint_positions', 10)
        
        self.joint_velocities_pub = self.create_publisher(
            JointState, '/robot/joint_velocities', 10)
            
        self.position_array_pub = self.create_publisher(
            Float64MultiArray, '/robot/position_array', 10)
        
        # Subscriber for robot status
        self.status_sub = self.create_subscription(
            String, '/robot/state', self.status_callback, 10)
        
        # Joint names for Franka Panda
        self.joint_names = [
            "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
            "panda_joint5", "panda_joint6", "panda_joint7"
        ]
        
        # Predefined poses
        self.poses = {
            'home': [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
            'ready': [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.785],
            'extended': [1.57, -0.5, 0.0, -1.5, 0.0, 1.0, 0.785],
            'folded': [0.0, -1.57, 0.0, -2.8, 0.0, 0.5, 0.785]
        }
        
        # Timer for demonstration movements
        self.demo_timer = None
        self.demo_step = 0
        
        self.get_logger().info("🚀 Robot Controller started")
        self.get_logger().info("📋 Available commands:")
        self.get_logger().info("  - move_to_pose(pose_name)")
        self.get_logger().info("  - move_joints(positions)")
        self.get_logger().info("  - start_demo()")
        
    def status_callback(self, msg):
        """Handle robot status updates"""
        # Just log status for now
        pass
        
    def move_to_pose(self, pose_name):
        """Move robot to a predefined pose"""
        if pose_name not in self.poses:
            self.get_logger().error(f"Unknown pose: {pose_name}")
            return False
            
        positions = self.poses[pose_name]
        return self.move_joints(positions)
        
    def move_joints(self, positions):
        """Move robot joints to specified positions"""
        if len(positions) != len(self.joint_names):
            self.get_logger().error(f"Invalid number of positions: {len(positions)}, expected {len(self.joint_names)}")
            return False
            
        # Send as JointState message
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = self.joint_names
        joint_msg.position = positions
        
        self.joint_positions_pub.publish(joint_msg)
        
        positions_str = [f'{pos:.3f}' for pos in positions]
        self.get_logger().info(f"📤 Moving to: {positions_str}")
        return True
        
    def move_joint_velocities(self, velocities):
        """Apply joint velocities"""
        if len(velocities) != len(self.joint_names):
            self.get_logger().error(f"Invalid number of velocities: {len(velocities)}, expected {len(self.joint_names)}")
            return False
            
        # Send as JointState message
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = self.joint_names
        joint_msg.velocity = velocities
        
        self.joint_velocities_pub.publish(joint_msg)
        
        velocities_str = [f'{vel:.3f}' for vel in velocities]
        self.get_logger().info(f"📤 Velocities: {velocities_str}")
        return True
        
    def start_demo(self):
        """Start demonstration sequence"""
        self.get_logger().info("🎬 Starting robot demonstration...")
        self.demo_step = 0
        self.demo_timer = self.create_timer(3.0, self.demo_callback)
        
    def demo_callback(self):
        """Demonstration sequence callback"""
        pose_sequence = ['home', 'ready', 'extended', 'folded', 'home']
        
        if self.demo_step < len(pose_sequence):
            pose_name = pose_sequence[self.demo_step]
            self.get_logger().info(f"🎬 Demo step {self.demo_step + 1}: Moving to {pose_name}")
            self.move_to_pose(pose_name)
            self.demo_step += 1
        else:
            self.get_logger().info("🎬 Demo complete!")
            if self.demo_timer is not None:
                self.demo_timer.cancel()
                self.demo_timer = None


def main(args=None):
    rclpy.init(args=args)
    
    try:
        controller = RobotController()
        
        # Example: Move to home position
        controller.move_to_pose('home')
        
        # Start demo after 5 seconds
        def start_demo():
            controller.start_demo()
        
        controller.create_timer(5.0, start_demo)
        
        rclpy.spin(controller)
        
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 