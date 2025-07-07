# Python Path Setup for RL CuRobo

This document explains how to set up Python paths for the RL CuRobo project without using ugly `sys.path.insert()` hacks or pip installations.

## Problem Solved

Previously, you had to use ugly code like:
```python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
```

And you had pip installation issues. Now you have a clean solution!

## Solution

### Method 1: Automatic Setup (Recommended)

Just import the setup function at the top of your scripts:

```python
# At the top of your script
from pathlib import Path
import sys
import os

def setup_project_paths():
    """Setup Python paths for the project automatically."""
    project_root = Path(__file__).parent.parent.parent.absolute()
    
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    projects_root = project_root / "projects_root"
    if str(projects_root) not in sys.path:
        sys.path.insert(0, str(projects_root))
    
    # Handle container mode
    if 'REPO_PATH_CONTAINER' in os.environ:
        container_path = os.environ['REPO_PATH_CONTAINER']
        if container_path not in sys.path:
            sys.path.insert(0, container_path)
            container_projects_root = os.path.join(container_path, "projects_root")
            if container_projects_root not in sys.path:
                sys.path.insert(0, container_projects_root)

setup_project_paths()
```

### Method 2: Environment Setup

#### Host Mode
```bash
# Source the setup script
source setup_env.sh

# Or set PYTHONPATH manually
export PYTHONPATH="/path/to/rl_for_curobo:/path/to/rl_for_curobo/projects_root:$PYTHONPATH"
```

#### Container Mode
```bash
# Set the container path
export REPO_PATH_CONTAINER="/path/to/mounted/repo"

# Source the setup script
source setup_env.sh
```

### Method 3: Python Script Setup
```bash
# Run the setup script
python setup_python_path.py

# Or import it in your script
import setup_python_path
```

## Usage Examples

After setup, you can import modules cleanly:

```python
# These will work without any path hacks!
from projects_root.utils.issacsim import init_app, wait_for_playing
from projects_root.utils.usd_utils import load_usd_to_stage
from projects_root.autonomous_arm import ArmMpc
from curobo.geom.types import Sphere, WorldConfig
```

## Benefits

1. **No more ugly `sys.path.insert()` hacks**
2. **No pip installation needed**
3. **Works in both host and container modes**
4. **Clean, maintainable code**
5. **Automatic path detection**

## Container Mode

When running in a container, set the `REPO_PATH_CONTAINER` environment variable:

```bash
export REPO_PATH_CONTAINER="/workspace/rl_for_curobo"
```

The setup will automatically detect this and use the container paths instead of local paths.

## Troubleshooting

If imports still don't work:

1. Check that `PYTHONPATH` is set correctly:
   ```bash
   echo $PYTHONPATH
   ```

2. Verify the paths exist:
   ```bash
   ls -la /path/to/rl_for_curobo/projects_root
   ```

3. Test the setup:
   ```python
   python -c "import sys; print('\n'.join(sys.path))"
   ```

## Migration Guide

To update existing scripts:

1. **Remove** the ugly `sys.path.insert()` line
2. **Add** the `setup_project_paths()` function at the top
3. **Import** your modules normally

Before:
```python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from projects_root.utils.issacsim import init_app
```

After:
```python
setup_project_paths()  # Clean setup function
from projects_root.utils.issacsim import init_app  # Clean import
``` 