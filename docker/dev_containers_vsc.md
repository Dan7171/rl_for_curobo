# Isaac Sim 4.5 + CuRobo Dev Container

This dev container configuration allows you to develop with Isaac Sim 4.5 and CuRobo directly in VSCode/Cursor, providing a consistent development environment.

## Prerequisites

1. **Docker** with GPU support (nvidia-docker2)
2. **VSCode** or **Cursor** with the Dev Containers extension
3. **NVIDIA GPU** with drivers installed
4. **X11 server** for GUI forwarding (Linux/WSL2)

## Setup

### 1. Prepare Host System

Before opening the dev container for the first time, run the setup script:

```bash
./.devcontainer/setup_host.sh
```

This script will:
- Create required cache directories for Isaac Sim
- Set up X11 forwarding permissions
- Display helpful usage instructions

### 2. Open in Dev Container

1. Open this project in VSCode/Cursor
2. You should see a popup asking to "Reopen in Container" - click it
3. Or manually: `Ctrl+Shift+P` → "Dev Containers: Reopen in Container"

The container will:
- Pull the Isaac Sim image if needed (`de257/curobo_isaac45v2:latest`)
- Install your `rl_for_curobo` package in development mode
- Set up the Python interpreter to use Isaac Sim's Python

## Usage

### Running Isaac Sim

```bash
# Start Isaac Sim GUI
/isaac-sim/isaac-sim.sh

# Run with a specific script
omni_python /workspace/rl_for_curobo/projects_root/examples/your_script.py
```

### Development Commands

```bash
# Install package in development mode (already done in postCreateCommand)
/isaac-sim/python.sh -m pip install -e /workspace/rl_for_curobo

# Run your scripts
cd /workspace/rl_for_curobo
omni_python projects_root/demos/dec_mpc_predictive/simple_env.py

# Run CuRobo examples
omni_python /pkgs/curobo/examples/isaac_sim/mpc_example.py
```

### Python Interpreter

The dev container is configured to use Isaac Sim's Python interpreter (`/isaac-sim/python.sh`) which includes:
- Isaac Sim Python bindings
- CuRobo
- PyTorch with CUDA support
- All required dependencies

## Configuration Details

### Container Specifications
- **Base Image**: `de257/curobo_isaac45v2:latest`
- **User**: root (required for Isaac Sim)
- **Workspace**: `/workspace/rl_for_curobo`
- **GPU Access**: Full GPU passthrough
- **Network**: Host networking

### Mounted Directories
- **Project**: `${workspaceFolder}` → `/workspace/rl_for_curobo`
- **X11**: `/tmp/.X11-unix` and `$HOME/.Xauthority`
- **Isaac Sim Caches**: `~/docker/isaac-sim/cache/*`

### Environment Variables
- `OMNI_KIT_ALLOW_ROOT=1`: Allow Isaac Sim to run as root
- `DISPLAY`: X11 display forwarding
- `NVIDIA_*`: GPU driver configuration
- ROS2 environment sourced

## Troubleshooting

### GUI Not Working
```bash
# On host, allow X11 connections
xhost +local:root

# Check DISPLAY variable in container
echo $DISPLAY
```

### Permission Issues
The container runs as root by default, which is required for Isaac Sim. File ownership may need adjustment:

```bash
# Fix ownership (run on host)
sudo chown -R $USER:$USER .
```

### GPU Not Available
```bash
# Check GPU access in container
nvidia-smi

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Cache Issues
If Isaac Sim loads slowly or has issues:

```bash
# Clear cache (run on host)
rm -rf ~/docker/isaac-sim/cache/*
./.devcontainer/setup_host.sh
```

## Comparison with Docker Script

This dev container provides the same functionality as `start_docker_isaac_sim45rootv2.sh` but integrated into VSCode/Cursor:

| Feature | Docker Script | Dev Container |
|---------|---------------|---------------|
| Isaac Sim GUI | ✅ | ✅ |
| GPU Access | ✅ | ✅ |
| Package Installation | ✅ | ✅ (automatic) |
| IDE Integration | ❌ | ✅ |
| Debugging | Limited | Full VSCode/Cursor |
| Extensions | Manual | Automatic |

## Development Tips

1. **Use the integrated terminal** for running scripts - it has the correct environment
2. **Python IntelliSense** is configured for Isaac Sim and CuRobo
3. **Jupyter notebooks** are supported with the installed extensions
4. **Git integration** works normally within the container
5. **File watching** excludes cache directories for performance 