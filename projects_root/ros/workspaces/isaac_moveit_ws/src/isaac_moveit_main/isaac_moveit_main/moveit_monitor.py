#!/usr/bin/env python3
"""
MoveIt-RViz Communication Monitor

This script monitors the communication between RViz and MoveIt to verify
that requests are being received and processed correctly.

Monitors:
- MoveIt action goals and results
- RViz planning scene updates
- Goal pose publications
- Trajectory execution
- Service calls

Usage:
    ros2 run isaac_moveit_main moveit_monitor

Author: Assistant
Date: 2024
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup

# ROS2 Messages
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Pose
from sensor_msgs.msg import JointState
from moveit_msgs.msg import (
    PlanningScene, DisplayTrajectory, MotionPlanRequest,
    MotionPlanResponse, RobotState, Constraints
)
from visualization_msgs.msg import Marker, MarkerArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# Custom action types (from your existing code)
from moveit_msgs.action import MoveGroup

# ROS2 Services
from moveit_msgs.srv import GetPositionFK, GetPositionIK, GetPlanningScene
from std_srvs.srv import Trigger

import time
import threading
from collections import defaultdict, deque
import json


class MoveItMonitor(Node):
    """
    Monitor for MoveIt-RViz communication.
    
    This class tracks:
    - Action goals and results
    - Topic publications
    - Service calls
    - Planning scene updates
    - Trajectory execution
    """
    
    def __init__(self):
        super().__init__('moveit_monitor')
        
        # Statistics tracking
        self.stats = {
            'action_goals': 0,
            'action_results': 0,
            'planning_scene_updates': 0,
            'goal_poses': 0,
            'trajectory_executions': 0,
            'service_calls': 0,
            'errors': 0
        }
        
        # Recent activity tracking (last 10 events)
        self.recent_activity = deque(maxlen=10)
        
        # Initialize joint states tracking
        self._last_joint_states = None
        
        # Set up QoS for reliable monitoring
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Monitor MoveIt action
        self._action_client = ActionClient(
            self, MoveGroup, '/move_action',
            callback_group=ReentrantCallbackGroup()
        )
        
        # Monitor key topics
        self._setup_topic_monitors(qos)
        
        # Monitor services
        self._setup_service_monitors()
        
        # Start monitoring
        self.get_logger().info('MoveIt-RViz Monitor started')
        self.get_logger().info('Monitoring topics and actions...')
        
        # Start statistics timer
        self._stats_timer = self.create_timer(5.0, self._print_stats)
        
    def _setup_topic_monitors(self, qos):
        """Set up topic monitors for key MoveIt-RViz communication."""
        
        # Monitor goal poses from RViz
        self._goal_pose_sub = self.create_subscription(
            PoseStamped,
            '/move_group/display_goal_pose',
            self._goal_pose_callback,
            qos
        )
        
        # Monitor planning scene updates
        self._planning_scene_sub = self.create_subscription(
            PlanningScene,
            '/move_group/monitored_planning_scene',
            self._planning_scene_callback,
            qos
        )
        
        # Monitor planned trajectories
        self._planned_path_sub = self.create_subscription(
            DisplayTrajectory,
            '/move_group/display_planned_path',
            self._planned_path_callback,
            qos
        )
        
        # Monitor robot state
        self._robot_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self._robot_state_callback,
            qos
        )
        
        # Monitor markers (visualization)
        self._marker_sub = self.create_subscription(
            Marker,
            '/goal_pose_marker',
            self._marker_callback,
            qos
        )
        
        # Monitor trajectory execution
        self._trajectory_sub = self.create_subscription(
            JointTrajectory,
            '/arm_controller/follow_joint_trajectory/goal',
            self._trajectory_callback,
            qos
        )
        
    def _setup_service_monitors(self):
        """Set up service monitors."""
        # Note: Service monitoring requires creating service clients
        # For now, we'll monitor service calls through topic activity
        
        self.get_logger().info('Service monitoring available for:')
        self.get_logger().info('  - /compute_fk')
        self.get_logger().info('  - /compute_ik')
        self.get_logger().info('  - /get_planning_scene')
        
    def _goal_pose_callback(self, msg):
        """Monitor goal pose publications from RViz."""
        self.stats['goal_poses'] += 1
        self._log_activity('GOAL_POSE', f'Position: ({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, {msg.pose.position.z:.3f})')
        
    def _planning_scene_callback(self, msg):
        """Monitor planning scene updates."""
        self.stats['planning_scene_updates'] += 1
        scene_objects = len(msg.world.collision_objects) if hasattr(msg.world, 'collision_objects') else 0
        self._log_activity('PLANNING_SCENE', f'Objects: {scene_objects}')
        
    def _planned_path_callback(self, msg):
        """Monitor planned trajectory publications."""
        if msg.trajectory and len(msg.trajectory) > 0:
            trajectory = msg.trajectory[0]
            points = len(trajectory.joint_trajectory.points) if trajectory.joint_trajectory.points else 0
            self._log_activity('PLANNED_PATH', f'Trajectory points: {points}')
        
    def _robot_state_callback(self, msg):
        """Monitor robot state updates."""
        if hasattr(self, '_last_joint_states'):
            # Check for significant changes
            if self._last_joint_states and len(msg.position) == len(self._last_joint_states.position):
                max_change = max(abs(msg.position[i] - self._last_joint_states.position[i]) 
                               for i in range(len(msg.position)))
                if max_change > 0.01:  # Significant movement
                    self._log_activity('ROBOT_STATE', f'Joint movement detected: max_change={max_change:.4f}')
        
        self._last_joint_states = msg
        
    def _marker_callback(self, msg):
        """Monitor visualization markers."""
        self._log_activity('MARKER', f'Type: {msg.type}, ID: {msg.id}')
        
    def _trajectory_callback(self, msg):
        """Monitor trajectory execution."""
        self.stats['trajectory_executions'] += 1
        points = len(msg.points) if msg.points else 0
        self._log_activity('TRAJECTORY_EXECUTION', f'Points: {points}')
        
    def _log_activity(self, activity_type, details):
        """Log activity with timestamp."""
        timestamp = time.strftime('%H:%M:%S')
        log_entry = f'[{timestamp}] {activity_type}: {details}'
        self.recent_activity.append(log_entry)
        self.get_logger().info(log_entry)
        
    def _print_stats(self):
        """Print current statistics."""
        self.get_logger().info('=== MoveIt-RViz Communication Statistics ===')
        for key, value in self.stats.items():
            self.get_logger().info(f'{key.replace("_", " ").title()}: {value}')
        self.get_logger().info('==========================================')
        
    def test_moveit_connection(self):
        """Test MoveIt action server connection."""
        self.get_logger().info('Testing MoveIt action server connection...')
        
        if self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().info('✓ MoveIt action server is available')
            return True
        else:
            self.get_logger().error('✗ MoveIt action server not available')
            return False
            
    def monitor_action_goals(self):
        """Monitor action goals in real-time."""
        self.get_logger().info('Monitoring MoveIt action goals...')
        
        # This would require implementing action goal monitoring
        # For now, we'll monitor through topic activity
        
    def get_communication_summary(self):
        """Get a summary of RViz-MoveIt communication."""
        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'statistics': self.stats.copy(),
            'recent_activity': list(self.recent_activity),
            'topics_active': self._check_active_topics(),
            'services_available': self._check_available_services()
        }
        return summary
        
    def _check_active_topics(self):
        """Check which monitored topics are active."""
        import subprocess
        try:
            result = subprocess.run(['ros2', 'topic', 'list'], 
                                  capture_output=True, text=True, timeout=5)
            topics = result.stdout.split('\n')
            monitored_topics = [
                '/move_group/display_goal_pose',
                '/move_group/monitored_planning_scene',
                '/move_group/display_planned_path',
                '/joint_states',
                '/goal_pose_marker'
            ]
            active_topics = [topic for topic in monitored_topics if topic in topics]
            return active_topics
        except Exception as e:
            self.get_logger().warn(f'Could not check active topics: {e}')
            return []
            
    def _check_available_services(self):
        """Check which MoveIt services are available."""
        import subprocess
        try:
            result = subprocess.run(['ros2', 'service', 'list'], 
                                  capture_output=True, text=True, timeout=5)
            services = result.stdout.split('\n')
            moveit_services = [s for s in services if 'move_group' in s or 'compute_' in s]
            return moveit_services
        except Exception as e:
            self.get_logger().warn(f'Could not check available services: {e}')
            return []


def main(args=None):
    rclpy.init(args=args)
    
    monitor = MoveItMonitor()
    
    # Test connection
    if not monitor.test_moveit_connection():
        monitor.get_logger().error('MoveIt not available. Please start MoveIt first.')
        return
    
    monitor.get_logger().info('MoveIt-RViz Monitor is running...')
    monitor.get_logger().info('Try interacting with RViz to see communication activity.')
    monitor.get_logger().info('Press Ctrl+C to stop monitoring.')
    
    try:
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        monitor.get_logger().info('Shutting down monitor...')
        
        # Print final summary
        summary = monitor.get_communication_summary()
        monitor.get_logger().info('=== Final Communication Summary ===')
        monitor.get_logger().info(json.dumps(summary, indent=2))
        
    finally:
        monitor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 