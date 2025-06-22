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

1. **Activate CuRobo environment:**
   ```bash
   conda activate %your_curobo_env%
   ```

2. **Navigate to server directory and run:**
   ```bash
   cd /home/dan/rl_for_curobo/projects_root/utils/hpc_utils
   python3 mpc_server.py --port 10051
   ```

3. **Expected output:**
   ```
   MPC Solver Server listening on port 10051
   Server running... Press Ctrl+C to stop
   ```

---

## Step 3: Run the Client Script

```bash
cd /home/dan/rl_for_curobo/projects_root/examples

# For localhost:
python3 mpc_example_client.py --server_ip localhost --server_port 10051

# For remote server (replace with your server IP):
python3 mpc_example_client.py --server_ip %server_ip% --server_port 10051

# Example for SLURM cluster:
python3 mpc_example_client.py --server_ip 132.72.65.138 --server_port 10051
```

---

## Performance Notes

- **Local**: ~0.005s per MPC step (baseline)
- **Remote**: ~0.035s per MPC step (network + serialization overhead)

**Note**: Remote connections are ~7x slower than local due to pickle serialization of PyTorch tensors and network latency. This is the fundamental trade-off for distributed computation.

---

## Troubleshooting

### Connection Issues:
1. **"No route to host"**: Check firewall settings or try different ports
2. **"Address already in use"**: Port is occupied, try `--port 10052` or similar
3. **Slow performance**: Check network latency with `ping`

---

## 🎯 Success!

Your MPC computations are now running on the server while your simulation runs on the client!


