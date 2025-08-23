#!/usr/bin/env python3
"""
ROS2 Robot Command Sender
Receives ROS2 joint position messages and sends commands to Isaac Sim via file
"""

import json
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String


class RobotCommandSender(Node):
    def __init__(self):
        super().__init__('robot_command_sender')
        
        # Command file for Isaac Sim communication
        self.command_file = Path("/tmp/isaac_robot_commands.json")
        
        # ROS2 subscriber for joint positions
        self.joint_sub = self.create_subscription(
            JointState, '/robot/joint_positions', self.joint_callback, 10)
        
        # Publisher for status (optional)
        self.status_pub = self.create_publisher(String, '/robot/state', 10)
        
        self.get_logger().info("🚀 Robot Command Sender started")
        self.get_logger().info(f"📂 Sending commands to: {self.command_file}")
        
    def joint_callback(self, msg):
        """Handle joint position commands and send to Isaac Sim"""
        command = {
            'type': 'joint_position',
            'joint_names': list(msg.name),
            'positions': list(msg.position),
            'timestamp': time.time()
        }
        self.send_command([command])
        self.get_logger().info(f"📤 Joint positions: {command['joint_names']} = {command['positions']}")
        
    def send_command(self, commands):
        """Send commands to Isaac Sim via file"""
        try:
            existing_commands = []
            if self.command_file.exists():
                with open(self.command_file, 'r') as f:
                    existing_commands = json.load(f)
            all_commands = existing_commands + commands
            with open(self.command_file, 'w') as f:
                json.dump(all_commands, f)
        except Exception as e:
            self.get_logger().error(f"❌ Failed to send command: {e}")


def main(args=None):
    rclpy.init(args=args)
    
    try:
        sender = RobotCommandSender()
        rclpy.spin(sender)
    except KeyboardInterrupt:
        pass
    finally:
        sender.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 