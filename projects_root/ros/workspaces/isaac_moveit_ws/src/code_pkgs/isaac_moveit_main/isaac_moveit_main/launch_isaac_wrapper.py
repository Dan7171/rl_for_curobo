#!/usr/bin/env python3
"""
Wrapper script to launch Isaac Sim with proper Python environment.
"""

import subprocess
import sys
import os


def main():
    """
    Main function to launch Isaac Sim with proper Python environment.
    """
    # Get the path to the actual launch_isaac.py script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    launch_script = os.path.join(script_dir, 'launch_isaac.py')
    
    # Use Isaac Sim Python to run the script
    isaac_python = '/isaac-sim/python.sh'
    
    if not os.path.exists(isaac_python):
        print("Error: Isaac Sim Python not found at /isaac-sim/python.sh")
        sys.exit(1)
    
    if not os.path.exists(launch_script):
        print(f"Error: Launch script not found at {launch_script}")
        sys.exit(1)
    
    # Run the script with Isaac Sim Python
    try:
        result = subprocess.run([isaac_python, launch_script], check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running Isaac Sim script: {e}")
        return e.returncode
    except FileNotFoundError:
        print("Error: Isaac Sim Python not found")
        return 1


if __name__ == '__main__':
    sys.exit(main()) 