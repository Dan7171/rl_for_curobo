#!/usr/bin/env python3
"""
Test script to check controller configuration.
"""

import rclpy
from rclpy.node import Node
from controller_manager_msgs.srv import ListControllers
from std_srvs.srv import Trigger
import time


class ControllerTester(Node):
    def __init__(self):
        super().__init__('controller_tester')
        self.get_logger().info('Controller tester started')
        
    def list_controllers(self):
        """List all available controllers."""
        client = self.create_client(ListControllers, '/controller_manager/list_controllers')
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Controller manager service not available')
            return False
            
        request = ListControllers.Request()
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            response = future.result()
            self.get_logger().info(f'Found {len(response.controller)} controllers:')
            for controller in response.controller:
                self.get_logger().info(f'  - {controller.name}: {controller.state}')
            return True
        else:
            self.get_logger().error('Failed to get controller list')
            return False


def main():
    rclpy.init()
    tester = ControllerTester()
    
    # Wait a bit for services to be available
    time.sleep(2)
    
    # Test controller listing
    success = tester.list_controllers()
    
    tester.destroy_node()
    rclpy.shutdown()
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main()) 