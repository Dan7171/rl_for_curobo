"""
Client API for the MPC solver server using ZeroMQ.
"""
import pickle
import zmq
from typing import Any, Optional, Dict
import time
import zlib

from curobo.types.state import JointState
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig


class LightweightResult:
    """Wrapper to make lightweight response compatible with WrapResult interface."""
    
    def __init__(self, data: dict):
        self._data = data
        
    def __getattr__(self, name: str):
        if name in self._data:
            return self._data[name]
        elif name == 'metrics' and self._data.get('metrics'):
            return LightweightMetrics(self._data['metrics'])
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


class LightweightMetrics:
    """Wrapper for lightweight metrics to maintain compatibility."""
    
    def __init__(self, data: dict):
        self._data = data
        
    def __getattr__(self, name: str):
        if name in self._data:
            return self._data[name]
        else:
            return None  # Return None for missing attributes instead of raising error
            
    def item(self):
        """Support for .item() calls on tensor-like attributes."""
        if hasattr(self._data.get('feasible'), 'item'):
            return self._data['feasible'].item()
        return self._data.get('feasible')


class RemoteAttribute:
    """Proxy object for remote attributes that supports nested access and method calls."""
    
    def __init__(self, api_client, attr_path: str):
        self._api_client = api_client
        self._attr_path = attr_path
    
    def __getattr__(self, name: str):
        # For simple attributes that should be fetched, get them directly
        if name in ['joint_names', 'shape', 'device', 'dtype']:
            return self._api_client._get_attr_from_server(f"{self._attr_path}.{name}")
        
        # For other attributes, return a new RemoteAttribute for chaining
        return RemoteAttribute(self._api_client, f"{self._attr_path}.{name}")
    
    def __call__(self, *args, **kwargs):
        """Make the attribute callable - this will call the method on the server."""
        return self._api_client._call_on_server(self._attr_path, *args, **kwargs)
    
    def __getitem__(self, key):
        """Support indexing operations."""
        return self._api_client._call_on_server(f"{self._attr_path}.__getitem__", key)
    
    def clone(self):
        """Clone the tensor on the server."""
        return self._api_client._call_on_server(f"{self._attr_path}.clone")
    
    def unsqueeze(self, dim):
        """Unsqueeze the tensor on the server."""
        return self._api_client._call_on_server(f"{self._attr_path}.unsqueeze", dim)
    
    def copy_(self, other):
        """Copy another tensor into this one on the server."""
        return self._api_client._call_on_server(f"{self._attr_path}.copy_", other)


class MpcSolverApi:
    """
    Client API for remote MPC solver using ZeroMQ.
    
    Performance Optimization:
    The lightweight_response option reduces network payload by 98.4% by returning only
    essential data (action tensor) instead of the full WrapResult with all rollout data,
    debug information, and metrics. This eliminates the PyTorch tensor deserialization
    bottleneck that causes repeated _load_from_bytes calls during pickle.loads().
    
    Typical payload sizes:
    - Full WrapResult: ~100KB (dozens of tensors)  
    - Lightweight response: ~1.7KB (only action + metadata)
    """

    def __init__(self, server_ip: str, server_port: int, config_params: Dict[str, Any], lightweight_response: bool = True):
        """
        Connect to MPC solver server and initialize remote solver.

        Args:
            server_ip: IP address of the server
            server_port: Port of the server  
            config_params: Dictionary of configuration parameters for MPC solver
            lightweight_response: If True, return only essential data (action tensor) instead of full WrapResult
        """
        
        self.server_ip = server_ip
        self.server_port = server_port
        self.lightweight_response = lightweight_response
        
        # Initialize ZeroMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        
        # Basic socket settings
        self.socket.setsockopt(zmq.LINGER, 0)  # Don't block on close
        self.socket.setsockopt(zmq.RCVTIMEO, 30000)  # 30 second timeout
        
        # Connect to server
        self.socket.connect(f"tcp://{server_ip}:{server_port}")
        
        # Serialization settings
        self.pickle_protocol = pickle.HIGHEST_PROTOCOL
        
        print(f"Connected to MPC server at {server_ip}:{server_port}")
        print(f"Lightweight response mode: {lightweight_response}")
        
        # Initialize the MPC solver
        self._send_request("init_mpc", config_params)
        print("Remote MPC solver initialized")

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
            WrapResult or dict: Result of the optimization. If lightweight_response=True,
            returns dict with only essential data, otherwise returns full WrapResult.
        """
        return self._call_on_server("mpc.step", current_state, shift_steps, seed_traj, max_attempts, self.lightweight_response)

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
            attr_name: Name of the attribute
            
        Returns:
            The attribute value
        """
        return self._send_request("get_attr", attr_name)

    def _send_request(self, request_type: str, *args) -> Any:
        """
        Send a request to the server and get response.
        
        Args:
            request_type: Type of request
            *args: Arguments for the request
            
        Returns:
            Deserialized response from server
        """
        import time
        
        total_start = time.time()
        
        request = {
            "type": request_type,
            "args": args,
        }
        
        try:
            # TIMING: Serialize request
            serialize_start = time.time()
            pickled_request = pickle.dumps(request, protocol=self.pickle_protocol)
            serialize_time = time.time() - serialize_start
            
            # TIMING: Compress request
            compress_start = time.time()
            if len(pickled_request) > 1024:
                compressed_request = zlib.compress(pickled_request, level=1)
                self.socket.send(b'C' + compressed_request)
                compress_time = time.time() - compress_start
                request_size = len(compressed_request)
                compressed = True
            else:
                self.socket.send(b'U' + pickled_request)
                compress_time = time.time() - compress_start
                request_size = len(pickled_request)
                compressed = False
            
            # TIMING: Network round-trip (send + receive)
            network_start = time.time()
            response_data = self.socket.recv()
            network_time = time.time() - network_start
            
            # TIMING: Decompress response
            decompress_start = time.time()
            if response_data[0:1] == b'C':
                serialized_response = zlib.decompress(response_data[1:])
                response_size = len(response_data) - 1
                response_compressed = True
            else:
                serialized_response = response_data[1:]
                response_size = len(response_data) - 1
                response_compressed = False
            decompress_time = time.time() - decompress_start
            
            # TIMING: Deserialize response
            deserialize_start = time.time()
            response = pickle.loads(serialized_response)
            deserialize_time = time.time() - deserialize_start
            
            # TIMING: Wrap result
            wrap_start = time.time()
            # Check for errors
            if response.get("error"):
                raise RuntimeError(f"Server error: {response['error']}")
                
            result = response.get("result")
            
            # Wrap lightweight responses for compatibility
            if (isinstance(result, dict) and 
                'action' in result and 'js_action' in result and 'solve_time' in result):
                result = LightweightResult(result)
            wrap_time = time.time() - wrap_start
            
            total_time = time.time() - total_start
            
            # DETAILED TIMING LOG (only for step requests to avoid spam)
            if request_type == "call_method" and len(args) > 1 and args[1] == "mpc.step":
                print(f"\n=== MPC STEP TIMING BREAKDOWN ===")
                print(f"Total time:        {total_time*1000:.2f}ms")
                print(f"  Serialize req:   {serialize_time*1000:.2f}ms")
                print(f"  Compress req:    {compress_time*1000:.2f}ms")
                print(f"  Network round:   {network_time*1000:.2f}ms")
                print(f"  Decompress resp: {decompress_time*1000:.2f}ms")
                print(f"  Deserialize resp:{deserialize_time*1000:.2f}ms")
                print(f"  Wrap result:     {wrap_time*1000:.2f}ms")
                print(f"Request size:      {request_size} bytes ({'compressed' if compressed else 'uncompressed'})")
                print(f"Response size:     {response_size} bytes ({'compressed' if response_compressed else 'uncompressed'})")
                print(f"===================================\n")
                
            return result
            
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
    
    