#!/usr/bin/env python3
"""
Setup script for RL CuRobo Python path.
This script can be used in both host and container environments.

Usage:
    # Source this script to set PYTHONPATH
    source setup_python_path.py
    
    # Or import it in Python
    import setup_python_path
    
    # Or run it directly
    python setup_python_path.py
"""

import os
import sys
from pathlib import Path

def setup_python_path():
    """Setup Python path for RL CuRobo project."""
    
    # Get the root directory (where this script is located)
    root_dir = Path(__file__).parent.absolute()
    
    # Get current PYTHONPATH
    pythonpath = os.environ.get('PYTHONPATH', '')
    pythonpath_parts = pythonpath.split(':') if pythonpath else []
    
    # Add root directory
    if str(root_dir) not in pythonpath_parts:
        pythonpath_parts.insert(0, str(root_dir))
    
    # Add projects_root directory
    projects_root_dir = root_dir / "projects_root"
    if str(projects_root_dir) not in pythonpath_parts:
        pythonpath_parts.insert(0, str(projects_root_dir))
    
    # Handle container mode
    if 'REPO_PATH_CONTAINER' in os.environ:
        container_path = os.environ['REPO_PATH_CONTAINER']
        if container_path not in pythonpath_parts:
            pythonpath_parts.insert(0, container_path)
        
        container_projects_root = os.path.join(container_path, "projects_root")
        if container_projects_root not in pythonpath_parts:
            pythonpath_parts.insert(0, container_projects_root)
    
    # Update PYTHONPATH
    new_pythonpath = ':'.join(pythonpath_parts)
    os.environ['PYTHONPATH'] = new_pythonpath
    
    # Also update sys.path for current Python session
    for path in pythonpath_parts:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    print(f"PYTHONPATH set to: {new_pythonpath}")
    return new_pythonpath

def export_pythonpath():
    """Export PYTHONPATH for shell sourcing."""
    new_path = setup_python_path()
    print(f"export PYTHONPATH='{new_path}'")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--export":
        export_pythonpath()
    else:
        setup_python_path() 