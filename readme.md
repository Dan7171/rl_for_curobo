# rl_for_curobo

Multi-arm MPC (Model Predictive Control) project for robotics. This repository contains a centralized and decentralized MPC framework for multi-arm robotic systems.

## Requirements

### System Requirements
- **OS**: Ubuntu >= 20.04
- **GPU**: NVIDIA GPU with 8+ GB VRAM (RTX 4060 or higher recommended)
- **RAM**: 32 GB
- **Other**: git, conda, git-lfs

> **Note**: We developed primarily with RTX 4060, which is below the minimum specs for some components, but works well for most use cases.

### Supported Isaac Sim Versions
- **Isaac Sim 4.5** (Python 3.10, ldd 2.34+)
- **Isaac Sim 5.0** (Python 3.11, ldd 2.35+)

Before proceeding, check the [Isaac Sim 4.5 installation docs](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/install_python.html) and [Isaac Sim 5.0 installation docs](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/install_python.html) to determine which version is compatible with your system (primarily depends on your ldd version).

## Installation

### Step 1: Create Conda Environment

Choose a name for your environment (replace `<env_name>` with your chosen name):

```bash
# For Isaac Sim 4.5 (Python 3.10)
conda create -n <env_name> python=3.10

# OR for Isaac Sim 5.0 (Python 3.11)
conda create -n <env_name> python=3.11

# Activate the environment
conda activate <env_name>
```

### Step 2: Install Isaac Sim

We recommend installing Isaac Sim using pip within the conda environment. Alternative installation methods (Docker, from source, etc.) are also supported, but ensure all packages are installed in the same Python environment.

#### Option A: Isaac Sim 4.5 (Python 3.10)

```bash
conda activate <env_name>
pip install isaacsim[all]==4.5.0 --extra-index-url https://pypi.nvidia.com
pip install isaacsim[extscache]==4.5.0 --extra-index-url https://pypi.nvidia.com
```

#### Option B: Isaac Sim 5.0 (Python 3.11)

```bash
conda activate <env_name>
pip install isaacsim[all,extscache]==5.0.0 --extra-index-url https://pypi.nvidia.com
```

> **Note**: For detailed installation instructions, refer to the [official Isaac Sim Python installation guide](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_python.html).

### Step 3: Clone Repository

```bash
git clone https://github.com/RoboWorkshop/rl_for_curobo.git
cd rl_for_curobo
```

> **Note**: The repository name `rl_for_curobo` is a legacy name. This will be changed to `mpc-multi-arm` or similar in future versions.

### Step 4: Install CuRobo

```bash
cd curobo
pip install -e . --no-build-isolation
```

If the installation fails, try:

```bash
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO=0.0.0+local pip install -e . --no-build-isolation
```

> **Note**: You don't need to clone the CuRobo repository separately - it's already included in this repository. Do not run `git clone https://github.com/NVlabs/curobo.git`.

For more information, see the [CuRobo installation documentation](https://curobo.org/get_started/1_install_instructions.html).

### Step 5: Install rl_for_curobo Module

```bash
cd ..  # Return to rl_for_curobo root directory
pip install .
```

You should see installation logs and a success message: `Successfully installed rl_for_curobo-0.1.0`

### Step 6: Setup Git LFS

```bash
git lfs install
git lfs pull  # Important: Pulls large files like robot meshes
```

### Step 7: Verify Installation

Run a hello world example to verify everything is working correctly:

```bash
# TODO: Add example command here
```

## Getting Started

Once installation is complete, you can start using the multi-arm MPC framework. See the `examples/` directory for usage examples.

## Additional Resources

- [CuRobo Documentation](https://curobo.org/)
- [Isaac Sim Documentation](https://docs.isaacsim.omniverse.nvidia.com/)

## License

See LICENSE file for details.
