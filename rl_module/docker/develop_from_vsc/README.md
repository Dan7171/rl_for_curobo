# VSCode Development Setup for CuRobo with Isaac Sim 4.0.0

## System Requirements

1. **Hardware & OS**
   - Linux system with NVIDIA GPU
   - Tested on Ubuntu 20.04/22.04

2. **Required Software**
   - Docker
   ```bash
   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   ```
   - NVIDIA Docker Runtime
   ```bash
   # Install NVIDIA Docker
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```
   - VSCode
   ```bash
   # Install VSCode
   sudo snap install code --classic
   ```

3. **Directory Setup**
   ```bash
   # Create required directories
   mkdir -p ~/docker/isaac-sim/{cache/{kit,ov,pip,glcache,computecache},logs,data,documents}
   ```

## Prerequisites

1. **Configuration Files**
   - Ensure you have both `.devcontainer/devcontainer.json` and `pyrightconfig.json` in your project root
   - If not, create and copy them:
     ```bash
     mkdir -p .devcontainer
     cp rl_module/docker/develop_from_vsc/devcontainer.json .devcontainer/
     cp rl_module/docker/develop_from_vsc/pyrightconfig.json ./
     ```

2. **Required VSCode Extensions (Host)**
   - Install "Dev Containers" extension (`ms-vscode-remote.remote-containers`)
   - Install "Docker" extension (`ms-azuretools.vscode-docker`)
   
   The following extensions will be automatically installed in the container:
   - Python (`ms-python.python`)
   - Pylance (`ms-python.vscode-pylance`)
   - Python Indent (`kevinrose.vsc-python-indent`)
   - Python Docstring Generator (`njpwerner.autodocstring`)
   - Python Import Sort (`ms-python.isort`)
   - IntelliCode for Python (`visualstudioexptteam.vscodeintellicode-python`)

## Running the Development Environment

1. Open VSCode in your project directory
2. Press `Ctrl + Shift + P`
3. Type and select "Dev Containers: Reopen in Container"

### Notes
- If prompted to rebuild the container, select "Rebuild". This doesn't rebuild the Docker image; it only creates a new container with the specified configuration and mounts.
- The container setup will:
  - Mount necessary directories
  - Configure display settings
  - Install Python extensions
  - Set up the Isaac Sim Python interpreter (`/isaac-sim/python.sh`)

### Switching Between Host and Container

To return to the host system:
1. Click the blue "Dev Container" button in the bottom-left corner
2. Select "close remote connection" from the menu above

To reconnect to the container later:
1. Use `Ctrl + Shift + P`
2. "Dev Containers: Reopen in Container"

## Troubleshooting
- If display issues occur, ensure X11 forwarding is properly set up on your host
- If Python interpreter isn't recognized, try reloading VSCode window (`Ctrl + Shift + P` -> "Developer: Reload Window")
- If you get permission errors, ensure your user has access to Docker:
  ```bash
  sudo usermod -aG docker $USER
  # Log out and back in for changes to take effect
  ```

## Additional Notes for Python Development
- If modules are not being recognized (appearing in white):
  1. Wait for the Python Language Server to complete indexing
  2. Try reloading the VSCode window (`Ctrl + Shift + P` -> "Developer: Reload Window")
  3. Ensure you're in the correct workspace folder
  4. Check the Python extension output for any errors (`Ctrl + Shift + P` -> "Python: Show Output") 