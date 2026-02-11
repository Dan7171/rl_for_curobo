
import socket
import json

def get_joint_states(socket_path='/tmp/lowstate.sock'):
    """
    Get current joint positions from the server.
    
    Returns:
        list: Joint positions, or None if error
    """
    try:
        # Create socket and connect to server
        client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client_socket.connect(socket_path)
        
        # Send request (can be any message)
        client_socket.sendall(b'get_state')
        
        # Receive response
        data = client_socket.recv(4096)
        response = json.loads(data.decode('utf-8'))
        
        # Close connection
        client_socket.close()
        
        return response['joint_positions']
    
    except Exception as e:
        print(f"Error getting joint states: {e}")
        return None

def send_joint_commands(joint_positions, socket_path='/tmp/lowcmd.sock'):
    """
    Send desired joint positions to the robot via the ROS2 server process.
    
    Args:
        joint_positions: List of 7 desired joint positions
        socket_path: Path to the Unix socket
    
    Returns:
        dict: Response with 'success' (bool) and 'message' (str)
    """
    try:
        # Create socket and connect to server
        client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client_socket.connect(socket_path)
        
        # Prepare request
        request = json.dumps({
            'joint_positions': joint_positions
        })
        
        # Send request
        client_socket.sendall(request.encode('utf-8'))
        
        # Receive response
        data = client_socket.recv(4096)
        response = json.loads(data.decode('utf-8'))
        
        # Close connection
        client_socket.close()
        
        return response
    
    except Exception as e:
        print(f"Error sending joint commands: {e}")
        return {
            'success': False,
            'message': str(e)
        }

if __name__ == '__main__':
    # Example usage - call this in your loop
    import time
    
    print("Requesting joint states from server...")
    
    
    for i in range(100):
        
        current_joint_positions = get_joint_states()
        # next_js = mpc.plan()
        next_joint_positions = current_joint_positions # temporary, replace with real action
        send_joint_commands(next_joint_positions)
        if current_joint_positions is not None:
            print(f"\n # n requests {i+1}:")
            print(f"  Joint count: {len(current_joint_positions)}")
            print(f"  Positions: {current_joint_positions}")
        else:
            print(f"Iteration {i+1}: Failed to get joint states")
        
        time.sleep(0.5)
