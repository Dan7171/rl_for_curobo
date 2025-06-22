# MPC Client-Server Architecture

This directory contains a high-performance client-server implementation for CuRobo's MPC solver, designed for distributed computing scenarios where you want to offload heavy MPC computations to a remote server.

## Overview

- **`mpc_server.py`** - Server that runs the actual `MpcSolver` and responds to client requests
- **`mpc_example_client.py`** - Client version of the original MPC example that uses `MpcSolverApi`
- **`../utils/hpc_utils/mpc_solver_api.py`** - Client API that communicates with the server

## Key Features

- **High Performance**: Uses ZeroMQ + pickle for fast Python-to-Python communication
- **Any Python Object**: Can pass tensors, custom classes, etc. seamlessly  
- **Transparent API**: Client code looks identical to direct MPC usage
- **Nested Attribute Support**: `mpc.rollout_fn.compute_kinematics()` works transparently
- **Error Handling**: Proper error propagation from server to client

## Architecture

```
┌─────────────────┐    ZeroMQ+Pickle    ┌─────────────────┐
│                 │ ◄─────────────────► │                 │
│  Client         │                     │  Server         │
│  (Simulation)   │   Fast Binary       │  (GPU Compute)  │
│                 │   Communication     │                 │
│  MpcSolverApi   │                     │  MpcSolver      │
└─────────────────┘                     └─────────────────┘
```

## Installation

Install ZeroMQ:
```bash
pip install pyzmq
```

## Usage

### 1. Start the Server

```bash
# On the compute server (with GPU)
cd projects_root/examples
python mpc_server.py --port 8888
```

### 2. Test Communication (Optional)

```bash
# Test the communication without full simulation
cd projects_root/examples
python test_mpc_communication.py --server_ip YOUR_SERVER_IP --server_port 8888
```

### 3. Run the Client

```bash
# On the client machine (simulation)
cd projects_root/examples  
python mpc_example_client.py --server_ip YOUR_SERVER_IP --server_port 8888
```

### 3. Code Comparison

**Original (local):**
```python
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig

mpc = MpcSolver(mpc_config)
result = mpc.step(current_state, max_attempts=2)
```

**Client-Server:**
```python
from mpc_solver_api import MpcSolverApi

mpc = MpcSolverApi("server_ip", 8888, mpc_config)  # Only difference!
result = mpc.step(current_state, max_attempts=2)   # Same API
```

## Performance Considerations

- **Network Latency**: Best for LAN/high-speed connections
- **Batch Processing**: More efficient for batch operations
- **Tensor Serialization**: PyTorch tensors are efficiently serialized with pickle
- **Memory**: Large tensors are copied over network - consider data size

## Advanced Usage

### Custom Server Configuration

```bash
python mpc_server.py --port 9999 --log_level debug
```

### Multiple Clients

The server handles one client at a time (REQ/REP pattern). For multiple clients, consider:
- Running multiple server instances on different ports
- Using ZeroMQ ROUTER/DEALER pattern (requires code modification)

### Error Handling

```python
try:
    result = mpc.step(current_state)
except TimeoutError:
    print("Server timeout")
except RuntimeError as e:
    print(f"Server error: {e}")
```

## Troubleshooting

1. **Connection refused**: Ensure server is running and port is correct
2. **Import errors**: Expected in linting - runtime imports work in proper environment
3. **Timeout**: Increase timeout in `MpcSolverApi` if needed
4. **Memory errors**: Large tensors may cause issues - monitor memory usage

## Future Improvements

- Async client for non-blocking calls
- Connection pooling for multiple clients
- Compression for large tensor transfers
- GPU-direct memory sharing for same-machine scenarios 