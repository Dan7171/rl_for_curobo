"""
Simple import for RL CuRobo project.
Usage: Just add this line to any script:
    exec(open('import_rl_curobo.py').read())
"""

import sys
import os
from pathlib import Path

# Get the directory where this script is located (project root)
project_root = Path(__file__).parent.absolute()

# Add project root to Python path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Handle container mode
if 'REPO_PATH_CONTAINER' in os.environ:
    container_path = os.environ['REPO_PATH_CONTAINER']
    if container_path not in sys.path:
        sys.path.insert(0, container_path) 