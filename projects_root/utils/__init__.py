"""
Utils module for RL CuRobo project.
"""

import sys
import os
from pathlib import Path

# Auto-setup paths when this module is imported
def _setup_paths():
    """Setup Python paths for the RL CuRobo project."""
    # Get the project root (3 levels up from this file)
    project_root = Path(__file__).parent.parent.parent.absolute()
    
    # Add project root to Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Handle container mode
    if 'REPO_PATH_CONTAINER' in os.environ:
        container_path = os.environ['REPO_PATH_CONTAINER']
        if container_path not in sys.path:
            sys.path.insert(0, container_path)

# Auto-setup when this module is imported
_setup_paths()


