# MPC Client-Server Setup Guide

Follow these step-by-step instructions to run the MPC solver in a distributed client-server architecture.

## Prerequisites

- CuRobo environment activated on both client and server
- ZeroMQ installed (`pip install pyzmq`)
- Network connectivity between client and server (if different machines)

---

## Step 1: Select Your Server

Choose your server machine:
- **Local host**: Client and server on the same PC (no IP configuration needed)
- **Remote server**: Any machine with GPU and CuRobo installed (note the IP address)

---

## Step 2: Start the MPC Server

### On the server machine:

1. **Activate CuRobo environment:**
   ```bash
   conda activate %your_curobo_env%
   ```

2. **Navigate to server directory and run:**
   ```bash
   cd /home/dan/rl_for_curobo/projects_root/utils/hpc_utils
   python3 mpc_server.py --port 8888
   ```

3. **Expected output:**
   ```
   MPC Solver Server listening on port 8888
   Server running... Press Ctrl+C to stop
   ```

---

## Step 3: Run the Client Script

### On the client machine:

1. **Activate CuRobo environment:**
   ```bash
   conda activate %your_curobo_env%
   ```

2. **Navigate to examples directory and run:**
   ```bash
   cd /home/dan/rl_for_curobo/projects_root/examples
   python3 mpc_example_client.py --server_ip %server ip% --server_port 8888 --headless_mode native
   # note: if server is running locally: set %server ip% to localhost

   ```

### Architecture Notes

- **API Communication**: The client script uses `projects_root/utils/hpc_utils/mpc_solver_api.py` to communicate with the server
- **Transparent Integration**: `mpc_example_client.py` is identical to `mpc_example.py` except for remote MPC computation
- **Same Interface**: All MPC calls work exactly the same, but execute on the remote server

---

## 🎯 Success!

Your MPC computations are now running on the server while your simulation runs on the client!


