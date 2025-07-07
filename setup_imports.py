"""
Simple import setup for RL CuRobo project.
Just add this line to any script:
    import setup_imports
"""

import sys
import os
from pathlib import Path

def setup_paths():
    """Setup Python paths for the RL CuRobo project."""
    # Get the project root (where this file is located)
    project_root = Path(__file__).parent.absolute()
    
    # Add project root to Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Handle container mode
    if 'REPO_PATH_CONTAINER' in os.environ:
        container_path = os.environ['REPO_PATH_CONTAINER']
        if container_path not in sys.path:
            sys.path.insert(0, container_path)

# Auto-setup when this module is imported
setup_paths() 