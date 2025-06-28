#!/usr/bin/env python3
"""
Test script for ROS2 Multi-Robot Communication

This script tests that ROS2 is properly set up and can handle multi-robot communication.
It creates simple publisher/subscriber nodes to verify the communication infrastructure.
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
import threading
import time
import json
from std_msgs.msg import String


class TestPublisher(Node):
    """Simple test publisher node"""
    
    def __init__(self, robot_id: int):
        super().__init__(f'test_publisher_{robot_id}')
        self.robot_id = robot_id
        
        # Callback group for concurrent execution
        self.callback_group = ReentrantCallbackGroup()
        
        # Publisher
        self.publisher = self.create_publisher(
            String, 
            f'/robot_{robot_id}/test_msg', 
            10,
            callback_group=self.callback_group
        )
        
        # Timer to publish messages
        self.timer = self.create_timer(
            1.0,  # Publish every second
            self.publish_message,
            callback_group=self.callback_group
        )
        
        self.message_count = 0
        self.get_logger().info(f'Test Publisher {robot_id} initialized')
    
    def publish_message(self):
        """Publish a test message"""
        self.message_count += 1
        
        msg_data = {
            'robot_id': self.robot_id,
            'message_count': self.message_count,
            'timestamp': time.time(),
            'status': 'active'
        }
        
        msg = String()
        msg.data = json.dumps(msg_data)
        
        self.publisher.publish(msg)
        self.get_logger().info(f'Robot {self.robot_id} published message #{self.message_count}')


class TestSubscriber(Node):
    """Simple test subscriber node"""
    
    def __init__(self, robot_id: int, subscribe_to: list):
        super().__init__(f'test_subscriber_{robot_id}')
        self.robot_id = robot_id
        self.subscribe_to = subscribe_to
        
        # Callback group for concurrent execution
        self.callback_group = ReentrantCallbackGroup()
        
        # Subscribers for other robots
        self.subscribers = {}
        self.received_messages = {}
        
        for other_robot_id in subscribe_to:
            subscriber = self.create_subscription(
                String,
                f'/robot_{other_robot_id}/test_msg',
                lambda msg, rid=other_robot_id: self.message_callback(msg, rid),
                10,
                callback_group=self.callback_group
            )
            self.subscribers[other_robot_id] = subscriber
            self.received_messages[other_robot_id] = 0
        
        # Timer to report status
        self.status_timer = self.create_timer(
            5.0,  # Report every 5 seconds
            self.report_status,
            callback_group=self.callback_group
        )
        
        self.get_logger().info(f'Test Subscriber {robot_id} initialized, subscribing to: {subscribe_to}')
    
    def message_callback(self, msg: String, robot_id: int):
        """Handle received messages"""
        try:
            data = json.loads(msg.data)
            self.received_messages[robot_id] += 1
            
            self.get_logger().info(
                f'Robot {self.robot_id} received message from Robot {robot_id}: '
                f'#{data["message_count"]} (total received: {self.received_messages[robot_id]})'
            )
        except Exception as e:
            self.get_logger().error(f'Error parsing message from robot {robot_id}: {e}')
    
    def report_status(self):
        """Report communication status"""
        total_received = sum(self.received_messages.values())
        self.get_logger().info(
            f'Robot {self.robot_id} status - Total messages received: {total_received} '
            f'(from {len(self.received_messages)} robots)'
        )


def test_ros2_communication():
    """Test ROS2 multi-robot communication"""
    print("🧪 Starting ROS2 Multi-Robot Communication Test...")
    
    # Initialize ROS2
    rclpy.init()
    
    try:
        # Create test configuration (3 robots)
        n_robots = 3
        
        # Define which robots each robot subscribes to
        subscription_map = {
            0: [1, 2],  # Robot 0 subscribes to robots 1 and 2
            1: [0, 2],  # Robot 1 subscribes to robots 0 and 2
            2: [0, 1]   # Robot 2 subscribes to robots 0 and 1
        }
        
        # Create publishers and subscribers
        publishers = []
        subscribers = []
        
        for i in range(n_robots):
            pub = TestPublisher(i)
            sub = TestSubscriber(i, subscription_map[i])
            publishers.append(pub)
            subscribers.append(sub)
        
        # Create multi-threaded executor
        executor = MultiThreadedExecutor()
        
        # Add all nodes to executor
        for pub in publishers:
            executor.add_node(pub)
        for sub in subscribers:
            executor.add_node(sub)
        
        print(f"✅ Created {n_robots} publisher and {n_robots} subscriber nodes")
        print("🚀 Starting communication test (will run for 30 seconds)...")
        print("   Watch the logs to see messages being exchanged between robots")
        
        # Run for 30 seconds
        def stop_after_timeout():
            time.sleep(30)
            print("\n⏰ Test duration completed!")
            executor.shutdown()
        
        timeout_thread = threading.Thread(target=stop_after_timeout)
        timeout_thread.daemon = True
        timeout_thread.start()
        
        # Spin the executor
        try:
            executor.spin()
        except KeyboardInterrupt:
            print("\n🛑 Test interrupted by user")
        
        print("✅ ROS2 Multi-Robot Communication Test completed successfully!")
        print("   If you saw messages being exchanged, ROS2 is working correctly.")
        
    except Exception as e:
        print(f"❌ Error during ROS2 test: {e}")
        return False
    
    finally:
        # Cleanup
        try:
            for pub in publishers:
                pub.destroy_node()
            for sub in subscribers:
                sub.destroy_node()
        except:
            pass
        
        rclpy.shutdown()
    
    return True


if __name__ == "__main__":
    print("🤖 ROS2 Multi-Robot Communication Test")
    print("=" * 50)
    
    # Check if ROS2 is properly set up
    try:
        import rclpy
        from std_msgs.msg import String
        print("✅ ROS2 Python packages imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import ROS2 packages: {e}")
        print("   Please make sure to source the setup script:")
        print("   source projects_root/examples/setup_ros2_env.sh")
        exit(1)
    
    # Run the test
    success = test_ros2_communication()
    
    if success:
        print("\n🎉 Test completed! ROS2 multi-robot communication is working.")
        print("   You can now run the full multi-robot MPC system.")
    else:
        print("\n❌ Test failed. Please check your ROS2 setup.") 