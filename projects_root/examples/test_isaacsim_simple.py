"""
Simple script to test if Isaac Sim starts and stays running.
This is equivalent to running ./isaac-sim.sh from the command line.
"""

try:
    import isaacsim
except ImportError:
    print("Warning: isaacsim module not found. This is normal if not running in Isaac Sim environment.")

from omni.isaac.kit import SimulationApp

def main():
    # Start Isaac Sim
    simulation_app = SimulationApp({
        "headless": False,  # Set to True for headless mode
        "width": 1920,
        "height": 1080,
    })
    
    print("Isaac Sim started successfully!")
    print("The Isaac Sim window should now be open.")
    print("Close the window or press Ctrl+C to exit.")
    
    try:
        # Keep Isaac Sim running until the window is closed
        while simulation_app.is_running():
            simulation_app.update()
            
    except KeyboardInterrupt:
        print("\nReceived interrupt signal. Closing Isaac Sim...")
    
    finally:
        # Clean shutdown
        simulation_app.close()
        print("Isaac Sim closed.")

if __name__ == "__main__":
    main() 