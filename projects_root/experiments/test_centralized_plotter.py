#!/usr/bin/env python3
"""
Test script for the centralized live plotter.
This script simulates multiple agents sending cost data to verify the plotting system works.
"""

import sys
import os
import time
import threading
import torch
import numpy as np

# Add the curobo path
sys.path.append('/home/dan/rl_for_curobo/curobo/src')

def test_centralized_plotter():
    """Test the centralized plotter with simulated data."""
    
    print("=== Testing Centralized Live Plotter ===")
    
    try:
        # Import the plotter
        from curobo.rollout.arm_reacher import get_global_plotter
        
        # Initialize plotter
        plotter = get_global_plotter()
        plotter.enable_plotting(max_agents=3)
        print("✓ Centralized plotter enabled")
        
        # Start background plot update thread
        stop_event = threading.Event()
        
        def plot_update_worker():
            while not stop_event.is_set():
                try:
                    plotter.update_plots()
                except Exception as e:
                    print(f"Plot update error: {e}")
                time.sleep(0.1)  # Update every 100ms
        
        plot_thread = threading.Thread(target=plot_update_worker, daemon=True)
        plot_thread.start()
        print("✓ Plot update thread started")
        
        # Simulate 3 agents sending data
        print("Simulating 3 agents sending cost data...")
        
        for iteration in range(200):  # Run for 200 iterations
            for agent_id in range(3):
                # Create simulated cost data
                cost_dict = {
                    'total': torch.tensor(100 + 50 * np.sin(iteration * 0.1 + agent_id)),
                    'goal': torch.tensor(30 + 20 * np.sin(iteration * 0.05 + agent_id)),
                    'collision': torch.tensor(10 + 5 * np.random.random()),
                    'velocity': torch.tensor(5 + 2 * np.cos(iteration * 0.08 + agent_id)),
                }
                
                # Send data to plotter
                plotter.add_data(agent_id, cost_dict)
            
            # Print progress
            if iteration % 20 == 0:
                print(f"Iteration {iteration}/200")
            
            time.sleep(0.05)  # Simulate some computation time
        
        print("✓ Finished sending test data")
        print("The plot window should now show 3 agents with different cost trajectories.")
        print("Press Ctrl+C to stop and close the plotter...")
        
        # Keep running until user stops
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping test...")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        try:
            if 'stop_event' in locals():
                stop_event.set()
            if 'plot_thread' in locals():
                plot_thread.join(timeout=1.0)
            if 'plotter' in locals():
                plotter.shutdown()
            print("✓ Cleanup completed")
        except Exception as e:
            print(f"Cleanup error: {e}")

if __name__ == "__main__":
    test_centralized_plotter() 