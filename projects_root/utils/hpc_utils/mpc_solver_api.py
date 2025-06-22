"""
Client API for the MPC solver server using ZeroMQ for fast communication.
"""
import pickle
import zmq
from typing import Any, Optional, Dict
import time

from curobo.types.state import JointState
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig


class RemoteAttribute:
    """Proxy object for remote attributes that supports nested access and method calls."""
    
    def __init__(self, api_client, attr_path: str):
        self._api_client = api_client
        self._attr_path = attr_path
    
    def __getattr__(self, name: str):
        # For simple attributes that should be fetched, get them directly
        if name in ['joint_names', 'shape', 'device', 'dtype']:
            # Get the actual value instead of creating another RemoteAttribute
            return self._api_client._get_attr_from_server(f"{self._attr_path}.{name}")
        
        # For complex attributes, create nested RemoteAttribute
        new_path = f"{self._attr_path}.{name}"
        return RemoteAttribute(self._api_client, new_path)
    
    def __call__(self, *args, **kwargs):
        # Support method calls like mpc.rollout_fn.compute_kinematics(...)
        return self._api_client._call_on_server(self._attr_path, *args, **kwargs)
    
    def __getitem__(self, key):
        # Support indexing like mpc.some_array[0]
        return self._api_client._call_on_server(f"{self._attr_path}.__getitem__", key)
    
    def clone(self):
        # Support tensor operations like .clone()
        return self._api_client._call_on_server(f"{self._attr_path}.clone")
    
    def unsqueeze(self, dim):
        # Support tensor operations like .unsqueeze()
        return self._api_client._call_on_server(f"{self._attr_path}.unsqueeze", dim)
    
    def copy_(self, other):
        # Support tensor operations like .copy_()
        return self._api_client._call_on_server(f"{self._attr_path}.copy_", other)


class MpcSolverApi:
    """
    High-performance client API for remote MPC solver using ZeroMQ + pickle.
    """

    def __init__(self, server_ip: str, server_port: int, config_params: Dict[str, Any]):
        """
        Connect to MPC solver server and initialize remote solver.

        Args:
            server_ip: IP address of the server
            server_port: Port of the server  
            config_params: Dictionary of configuration parameters for MPC solver
        """
        
        self.server_ip = server_ip
        self.server_port = server_port
        
        # Initialize ZeroMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)  # Request socket
        self.socket.connect(f"tcp://{server_ip}:{server_port}")
        
        # Set socket options for performance
        self.socket.setsockopt(zmq.LINGER, 0)  # Don't wait on close
        self.socket.setsockopt(zmq.RCVTIMEO, 30000)  # 30 second timeout
        self.socket.setsockopt(zmq.SNDTIMEO, 30000)  # 30 second timeout
        
        print(f"Connected to MPC server at {server_ip}:{server_port}")
        
        # Send the configuration parameters directly to server
        self._send_request("init_mpc", config_params)
        print("Remote MPC solver initialized")

    def _extract_config_params(self, config_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        This method is no longer needed since we now pass config_params directly.
        Keeping for backward compatibility but just returns the input.
        
        Args:
            config_params: Dictionary of configuration parameters
            
        Returns:
            Dictionary of serializable parameters
        """
        return config_params

    def __del__(self):
        """Clean up ZeroMQ resources."""
        if hasattr(self, 'socket'):
            self.socket.close()
        if hasattr(self, 'context'):
            self.context.term()

    @property
    def rollout_fn(self):
        """Access to rollout_fn with nested attribute support."""
        return RemoteAttribute(self, "mpc.rollout_fn")
    
    @property  
    def world_coll_checker(self):
        """Access to world collision checker."""
        return RemoteAttribute(self, "mpc.world_coll_checker")

    def step(
        self,
        current_state: JointState,
        shift_steps: int = 1,
        seed_traj: Optional[JointState] = None,
        max_attempts: int = 1,
    ):
        """
        Solve for the next action given the current state.

        Args:
            current_state: Current joint state of the robot.
            shift_steps: Number of steps to shift the trajectory.
            seed_traj: Initial trajectory to seed the optimization.
            max_attempts: Maximum number of attempts to solve the problem.

        Returns:
            WrapResult: Result of the optimization.
        """
        return self._call_on_server("mpc.step", current_state, shift_steps, seed_traj, max_attempts)

    def setup_solve_single(self, goal, num_seeds: int = 1):
        """Setup single solve goal."""
        return self._call_on_server("mpc.setup_solve_single", goal, num_seeds)
    
    def update_goal(self, goal_buffer):
        """Update goal buffer."""
        return self._call_on_server("mpc.update_goal", goal_buffer)
    
    def get_visual_rollouts(self):
        """Get visual rollouts for debugging."""
        return self._call_on_server("mpc.get_visual_rollouts")

    def _call_on_server(self, method_name: str, *args, **kwargs) -> Any:
        """
        Call a method on the remote server.
        
        Supports nested calls like "mpc.rollout_fn.compute_kinematics"
        
        Args:
            method_name: Name of the method/attribute to access
            *args: Arguments to pass
            **kwargs: Keyword arguments to pass
            
        Returns:
            Result from the remote method call
        """
        return self._send_request("call_method", method_name, args, kwargs)

    def _get_attr_from_server(self, attr_name: str) -> Any:
        """
        Get an attribute from the remote server.
        
        Args:
            attr_name: Name of the attribute (supports nested like "mpc.rollout_fn.joint_names")
            
        Returns:
            The attribute value
        """
        return self._send_request("get_attr", attr_name)

    def _send_request(self, request_type: str, *args) -> Any:
        """
        Send a request to the server and get response.
        
        Args:
            request_type: Type of request ("init_mpc", "call_method", "get_attr")
            *args: Arguments for the request
            
        Returns:
            Deserialized response from server
        """
        # Prepare request
        request = {
            "type": request_type,
            "args": args,
            "timestamp": time.time()
        }
        
        # Serialize and send
        try:
            serialized_request = pickle.dumps(request)
            self.socket.send(serialized_request)
            
            # Receive response
            serialized_response = self.socket.recv()
            response = pickle.loads(serialized_response)
            
            # Check for errors
            if response.get("error"):
                raise RuntimeError(f"Server error: {response['error']}")
                
            return response.get("result")
            
        except zmq.Again:
            raise TimeoutError("Request to MPC server timed out")
        except Exception as e:
            raise RuntimeError(f"Communication error with MPC server: {e}")

    @property
    def server_copy_debug(self) -> MpcSolver:
        """
        Get the server-side MPC solver object for debugging.
        WARNING: This returns a local copy and should only be used for debugging.
        """
        return self._send_request("get_debug_copy")
    
    