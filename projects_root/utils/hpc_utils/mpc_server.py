#!/usr/bin/env python3
"""
High-performance MPC Solver Server using ZeroMQ for fast communication.

This server runs the actual MpcSolver and responds to client requests.
"""

import pickle
import zmq
import traceback
import argparse
from typing import Any, Dict

# CuRobo imports
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig
from curobo.util.logger import setup_curobo_logger
from curobo.types.base import TensorDeviceType


class MpcSolverServer:
    """
    Server that hosts an MpcSolver and responds to client requests via ZeroMQ.
    """
    
    def __init__(self, port: int = 8888):
        """
        Initialize the server.
        
        Args:
            port: Port to listen on
        """
        self.port = port
        self.mpc_solver = None
        
        # Initialize ZeroMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)  # Reply socket
        self.socket.bind(f"tcp://*:{port}")
        
        print(f"MPC Solver Server listening on port {port}")
        
    def __del__(self):
        """Clean up ZeroMQ resources."""
        if hasattr(self, 'socket'):
            self.socket.close()
        if hasattr(self, 'context'):
            self.context.term()
    
    def run(self):
        """Main server loop."""
        print("Server running... Press Ctrl+C to stop")
        
        try:
            while True:
                # Receive request
                try:
                    serialized_request = self.socket.recv()
                    request = pickle.loads(serialized_request)
                    
                    # Process request
                    response = self._process_request(request)
                    
                    # Send response
                    serialized_response = pickle.dumps(response)
                    self.socket.send(serialized_response)
                    
                except Exception as e:
                    # Send error response
                    error_response = {
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    }
                    serialized_response = pickle.dumps(error_response)
                    self.socket.send(serialized_response)
                    print(f"Error processing request: {e}")
                    
        except KeyboardInterrupt:
            print("\nShutting down server...")
        finally:
            self.socket.close()
            self.context.term()
    
    def _process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a client request.
        
        Args:
            request: Request dictionary with 'type' and 'args'
            
        Returns:
            Response dictionary
        """
        request_type = request.get("type")
        args = request.get("args", ())
        
        if request_type == "init_mpc":
            return self._init_mpc(*args)
        elif request_type == "call_method":
            return self._call_method(*args)
        elif request_type == "get_attr":
            return self._get_attr(*args)
        elif request_type == "get_debug_copy":
            return self._get_debug_copy()
        else:
            raise ValueError(f"Unknown request type: {request_type}")
    
    def _init_mpc(self, config_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize the MPC solver from configuration parameters.
        
        Args:
            config_params: Dictionary of configuration parameters
        """
        print("Initializing MPC solver from parameters...")
        
        # Extract robot_cfg and world_model separately
        robot_cfg = config_params.get('robot_cfg')
        world_model = config_params.get('world_cfg')  # Client sends as 'world_cfg' but we use as 'world_model'
        
        if robot_cfg is None:
            raise ValueError("robot_cfg is required")
        if world_model is None:
            raise ValueError("world_cfg is required")
        
        # Ensure tensor_args is never None
        tensor_args = config_params.get('tensor_args')
        if tensor_args is None:
            tensor_args = TensorDeviceType()
        
        # Create MpcSolverConfig using the correct API from mpc.py
        mpc_config = MpcSolverConfig.load_from_robot_config(
            robot_cfg=robot_cfg,
            world_model=world_model,  # Use world_model parameter name
            use_cuda_graph=config_params.get('use_cuda_graph', True),
            use_cuda_graph_metrics=config_params.get('use_cuda_graph_metrics', True),
            use_cuda_graph_full_step=config_params.get('use_cuda_graph_full_step', False),
            self_collision_check=config_params.get('self_collision_check', True),
            collision_checker_type=config_params.get('collision_checker_type', None),
            collision_cache=config_params.get('collision_cache', None),
            use_mppi=config_params.get('use_mppi', True),
            use_lbfgs=config_params.get('use_lbfgs', False),
            use_es=config_params.get('use_es', None),
            store_rollouts=config_params.get('store_rollouts', True),
            step_dt=config_params.get('step_dt', None),
            tensor_args=tensor_args,  # Use the guaranteed non-None tensor_args
            compute_metrics=config_params.get('compute_metrics', True),
            sync_cuda_time=config_params.get('sync_cuda_time', True),
            particle_opt_iters=config_params.get('particle_opt_iters', None),
            collision_activation_distance=config_params.get('collision_activation_distance', None),
            n_collision_envs=config_params.get('n_collision_envs', None),
            project_pose_to_goal_frame=config_params.get('project_pose_to_goal_frame', True),
        )
        
        # Create the MPC solver
        self.mpc_solver = MpcSolver(mpc_config)
        print("MPC solver initialized successfully")
        return {"result": "initialized"}
    
    def _call_method(self, method_path: str, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """
        Call a method on the MPC solver.
        
        Args:
            method_path: Dot-separated path like "mpc.step" or "mpc.rollout_fn.compute_kinematics"
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            Result of the method call
        """
        if self.mpc_solver is None:
            raise RuntimeError("MPC solver not initialized")
        
        # Navigate to the method/attribute
        obj = self.mpc_solver
        parts = method_path.split('.')
        
        # Skip 'mpc' prefix if present
        if parts[0] == 'mpc':
            parts = parts[1:]
        
        # Navigate to the parent object
        for part in parts[:-1]:
            obj = getattr(obj, part)
        
        # Get the final method/attribute
        method_name = parts[-1]
        
        # Handle special method names
        if method_name.startswith('__') and method_name.endswith('__'):
            # Special methods like __getitem__
            method = getattr(obj, method_name)
            result = method(*args, **kwargs)
        else:
            # Regular methods and attributes
            attr = getattr(obj, method_name)
            if callable(attr):
                result = attr(*args, **kwargs)
            else:
                # If it's an attribute, return it (for properties)
                result = attr
        
        return {"result": result}
    
    def _get_attr(self, attr_path: str) -> Dict[str, Any]:
        """
        Get an attribute from the MPC solver.
        
        Args:
            attr_path: Dot-separated path like "mpc.rollout_fn.joint_names"
            
        Returns:
            Value of the attribute
        """
        if self.mpc_solver is None:
            raise RuntimeError("MPC solver not initialized")
        
        # Navigate to the attribute
        obj = self.mpc_solver
        parts = attr_path.split('.')
        
        # Skip 'mpc' prefix if present
        if parts[0] == 'mpc':
            parts = parts[1:]
        
        # Navigate to the attribute
        for part in parts:
            obj = getattr(obj, part)
        
        return {"result": obj}
    
    def _get_debug_copy(self) -> Dict[str, Any]:
        """Get a copy of the MPC solver for debugging."""
        if self.mpc_solver is None:
            raise RuntimeError("MPC solver not initialized")
        
        # WARNING: This creates a full copy - expensive!
        import copy
        return {"result": copy.deepcopy(self.mpc_solver)}


def main():
    """Main server entry point."""
    parser = argparse.ArgumentParser(description="MPC Solver Server")
    parser.add_argument("--port", type=int, default=8888, help="Port to listen on")
    parser.add_argument("--log_level", type=str, default="info", help="Logging level")
    args = parser.parse_args()
    
    # Setup logging
    setup_curobo_logger(args.log_level)
    
    # Create and run server
    server = MpcSolverServer(args.port)
    server.run()


if __name__ == "__main__":
    main() 