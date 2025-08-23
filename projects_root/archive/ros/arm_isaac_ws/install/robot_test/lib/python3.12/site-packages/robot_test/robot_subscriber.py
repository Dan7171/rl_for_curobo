#!/usr/bin/env python3
"""
Robot Subscriber Node
Monitors robot joint states and provides analysis
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
import math


class RobotSubscriber(Node):
    def __init__(self):
        super().__init__('robot_subscriber')
        
        # Subscribers
        self.joint_states_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_states_callback, 10)
        
        self.robot_state_sub = self.create_subscription(
            String, '/robot/state', self.robot_state_callback, 10)
        
        # Publishers for analysis
        self.analysis_pub = self.create_publisher(String, '/robot/analysis', 10)
        
        # Robot state tracking
        self.last_joint_positions = None
        self.joint_names = [
            "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
            "panda_joint5", "panda_joint6", "panda_joint7"
        ]
        
        # Analysis timer
        self.analysis_timer = self.create_timer(1.0, self.publish_analysis)
        
        self.get_logger().info("🚀 Robot Subscriber started")
        self.get_logger().info("📊 Monitoring robot state and joint positions")
        
    def joint_states_callback(self, msg):
        """Handle joint states from Isaac Sim"""
        if len(msg.position) >= len(self.joint_names):
            self.last_joint_positions = list(msg.position[:len(self.joint_names)])
            
            positions_str = [f'{pos:.3f}' for pos in self.last_joint_positions]
            self.get_logger().info(f"📊 Joint positions: {positions_str}")
        
    def robot_state_callback(self, msg):
        """Handle robot state updates"""
        self.get_logger().info(f"🤖 Robot state: {msg.data}")
        
    def publish_analysis(self):
        """Publish robot analysis"""
        if self.last_joint_positions is None:
            return
            
        analysis = self.analyze_robot_pose()
        
        analysis_msg = String()
        analysis_msg.data = analysis
        self.analysis_pub.publish(analysis_msg)
        
    def analyze_robot_pose(self):
        """Analyze current robot pose"""
        if self.last_joint_positions is None:
            return "No joint data available"
            
        # Simple analysis
        analysis_parts = []
        
        # Check if joints are at limits
        joint_limits = [
            (-2.8973, 2.8973),  # Joint 1
            (-1.7628, 1.7628),  # Joint 2
            (-2.8973, 2.8973),  # Joint 3
            (-3.0718, -0.0698), # Joint 4
            (-2.8973, 2.8973),  # Joint 5
            (-0.0175, 3.7525),  # Joint 6
            (-2.8973, 2.8973)   # Joint 7
        ]
        
        near_limits = []
        for i, (pos, (min_limit, max_limit)) in enumerate(zip(self.last_joint_positions, joint_limits)):
            if abs(pos - min_limit) < 0.1 or abs(pos - max_limit) < 0.1:
                near_limits.append(f"Joint{i+1}")
                
        if near_limits:
            analysis_parts.append(f"Near limits: {', '.join(near_limits)}")
        
        # Check pose type
        pose_type = self.classify_pose()
        if pose_type:
            analysis_parts.append(f"Pose: {pose_type}")
            
        # Calculate end-effector approximate height (simple approximation)
        approx_height = self.estimate_end_effector_height()
        analysis_parts.append(f"EE height: {approx_height:.2f}m")
        
        return " | ".join(analysis_parts) if analysis_parts else "Normal operation"
        
    def classify_pose(self):
        """Classify the current pose"""
        if self.last_joint_positions is None:
            return None
            
        # Define some common poses (approximate)
        poses = {
            'home': [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
            'ready': [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.785],
            'extended': [1.57, -0.5, 0.0, -1.5, 0.0, 1.0, 0.785],
            'folded': [0.0, -1.57, 0.0, -2.8, 0.0, 0.5, 0.785]
        }
        
        current = self.last_joint_positions
        tolerance = 0.2
        
        for pose_name, pose_joints in poses.items():
            if all(abs(c - p) < tolerance for c, p in zip(current, pose_joints)):
                return pose_name
                
        return "custom"
        
    def estimate_end_effector_height(self):
        """Simple estimation of end-effector height"""
        if self.last_joint_positions is None:
            return 0.0
            
        # Very rough approximation based on joint angles
        # This is just for demonstration - real FK would be more complex
        base_height = 0.333  # Base height
        
        # Simple approximation using joint 2 and 4 (main vertical contributors)
        j2, j4 = self.last_joint_positions[1], self.last_joint_positions[3]
        
        # Rough calculation (not accurate, just for demo)
        vertical_contribution = 0.4 * math.sin(j2) + 0.3 * math.sin(j2 + j4)
        
        return base_height + vertical_contribution


def main(args=None):
    rclpy.init(args=args)
    
    try:
        subscriber = RobotSubscriber()
        rclpy.spin(subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 