#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from unitree_hg.msg import LowCmd
import socket
import os
import json
import threading

class LowCmdPublisher(Node):
    """Publish actions (target joint states) to ros topic"""
    def __init__(self):
        super().__init__('lowcmd_publisher')
        self.publisher = self.create_publisher(
            LowCmd,
            '/cmd/left', # our custom topic name
            30 # Hz (publishing frequency)
        )
        self.get_logger().info('LowCmd publisher initialized')
    
    def send_joint_commands(self, joint_positions):
        """
        Publish joint position commands to the robot.
        
        Args:
            joint_positions: List of 7 desired joint positions (left arm)
        """
        # if joint_positions is None or len(joint_positions) != 7:
        #     self.get_logger().error(f'Invalid joint positions: expected 7, got {len(joint_positions) if joint_positions else 0}')
        #     return False
        # if joint_positions is None or len(joint_positions) != 29:
        #     self.get_logger().error(f'Invalid joint positions: expected 29, got {len(joint_positions) if joint_positions else 0}')
        #     return False
        if joint_positions is None or len(joint_positions) != 35:
            self.get_logger().error(f'Invalid joint positions: expected 35, got {len(joint_positions) if joint_positions else 0}')
            return False
            
        # Create LowCmd message
        cmd_msg = LowCmd()
        
        # Set motor commands for each joint
        for i, position in enumerate(joint_positions):
            if i < len(cmd_msg.motor_cmd):
                cmd_msg.motor_cmd[i].q = float(position)
                # You may need to set other fields like kp, kd, tau_ff, dq depending on your robot's requirements
                # Example:
                # cmd_msg.motor_cmd[i].kp = 50.0
                # cmd_msg.motor_cmd[i].kd = 5.0
                # cmd_msg.motor_cmd[i].tau_ff = 0.0
                # cmd_msg.motor_cmd[i].dq = 0.0
        
        # Publish command
        self.publisher.publish(cmd_msg)
        self.get_logger().info(f'Published joint commands: {len(joint_positions)} joints')
        return True


class LowCmdServer:
    """Receives action requests from client and publishes them to ros topic for processing"""
    def __init__(self, socket_path='/tmp/lowcmd.sock'):
        self.socket_path = socket_path
        
        # Remove socket file if it exists
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)
        
        # Initialize ROS2
        rclpy.init()
        self.publisher = LowCmdPublisher()
        
        # Create Unix domain socket server
        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(self.socket_path)
        self.server_socket.listen(5)
        
        print(f"LowCmd server listening on {self.socket_path}")
        
        # Start ROS2 spin in separate thread
        self.spin_thread = threading.Thread(target=self._spin_ros, daemon=True)
        self.spin_thread.start()
        
        # Start accepting connections
        self.run()
    
    def _spin_ros(self):
        """Spin ROS2 in background thread"""
        while rclpy.ok():
            rclpy.spin_once(self.publisher, timeout_sec=0.01)
    
    def run(self):
        """Accept connections and handle requests"""
        try:
            while True:
                # Accept new connection
                client_socket, _ = self.server_socket.accept()
                
                # Handle request in separate thread
                threading.Thread(
                    target=self._handle_client,
                    args=(client_socket,),
                    daemon=True
                ).start()
        except KeyboardInterrupt:
            print("\nShutting down server...")
        finally:
            self.cleanup()
    
    def _handle_client(self, client_socket):
        """Handle individual client request"""
        try:
            # Receive command data
            data = client_socket.recv(8192)  # Larger buffer for joint commands
            
            if data:
                # Parse JSON request
                request = json.loads(data.decode('utf-8'))
                joint_positions = request.get('joint_positions')
                
                # Send command to robot
                success = self.publisher.send_joint_commands(joint_positions)
                
                # Send acknowledgment back to client
                response = json.dumps({
                    'success': success,
                    'message': 'Command sent' if success else 'Command failed'
                })
                client_socket.sendall(response.encode('utf-8'))
        except Exception as e:
            print(f"Error handling client: {e}")
            # Send error response
            try:
                error_response = json.dumps({
                    'success': False,
                    'message': str(e)
                })
                client_socket.sendall(error_response.encode('utf-8'))
            except:
                pass
        finally:
            client_socket.close()
    
    def cleanup(self):
        """Cleanup resources"""
        self.server_socket.close()
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)
        self.publisher.destroy_node()
        rclpy.shutdown()
        
if __name__ == '__main__':
    server = LowCmdServer()
