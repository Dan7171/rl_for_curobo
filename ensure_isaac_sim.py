"""
Helper script to ensure Isaac Sim environment is properly configured.
Import this at the top of any script that needs Isaac Sim.
"""

import os
import subprocess
import sys

def ensure_isaac_sim_env():
    """Ensure Isaac Sim environment is activated."""
    isaac_sim_setup = os.path.expanduser("~/isaacsim500/isaacsim/_build/linux-x86_64/release/setup_conda_env.sh")
    
    if os.path.exists(isaac_sim_setup):
        # Check if Isaac Sim environment variables are already set
        if 'ISAAC_SIM' not in os.environ:
            print("Setting up Isaac Sim environment...")
            # Source the setup script
            result = subprocess.run(
                f"source {isaac_sim_setup} && env",
                shell=True,
                capture_output=True,
                text=True,
                executable="/bin/bash"
            )
            
            if result.returncode == 0:
                # Parse environment variables and set them
                for line in result.stdout.split('\n'):
                    if '=' in line and not line.startswith('_'):
                        key, value = line.split('=', 1)
                        os.environ[key] = value
                print("Isaac Sim environment configured!")
            else:
                print(f"Warning: Failed to setup Isaac Sim environment: {result.stderr}")
    else:
        print(f"Warning: Isaac Sim setup script not found at {isaac_sim_setup}")

# Auto-run when imported
if __name__ != "__main__":
    ensure_isaac_sim_env() 