#!/usr/bin/env python3
"""
Simple test to verify centralized plotter works without CUDA.
"""

import sys
import time
import threading
import torch
import numpy as np

# Add the curobo path
sys.path.append('/home/dan/rl_for_curobo/curobo/src')

def test_simple_plotting():
    """Test centralized plotter with simulated ArmReacher data."""
    
    print("=== Testing Simple Centralized Plotter ===")
    
    try:
        # Import the plotter
        from curobo.rollout.arm_reacher import get_global_plotter
        
        # Initialize plotter
        plotter = get_global_plotter()
        plotter.enable_plotting(max_agents=2)
        print("✓ Centralized plotter enabled with 2 agents")
        
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
        
        # Simulate cost data from ArmReacher instances
        print("Simulating cost data from 2 agents...")
        
        for iteration in range(100):  # Run for 100 iterations
            for agent_id in range(2):
                # Create realistic cost data (similar to what ArmReacher.cost_fn sends)
                cost_dict = {
                    'total': torch.tensor(50 + 30 * np.sin(iteration * 0.1 + agent_id * 1.5) + 5 * np.random.random()),
                    'goal': torch.tensor(20 + 15 * np.sin(iteration * 0.05 + agent_id) + 2 * np.random.random()),
                    'collision': torch.tensor(5 + 3 * np.random.random()),
                    'velocity': torch.tensor(10 + 5 * np.cos(iteration * 0.08 + agent_id) + np.random.random()),
                    'acceleration': torch.tensor(8 + 3 * np.sin(iteration * 0.12 + agent_id * 0.7) + np.random.random()),
                }
                
                # Send data to centralized plotter (this is what ArmReacher.cost_fn does)
                plotter.add_data(agent_id, cost_dict)
            
            # Print progress occasionally
            if iteration % 20 == 0:
                print(f"Iteration {iteration}/100 - Data sent to plotter")
            
            time.sleep(0.05)  # Simulate computation time
        
        print("✓ Finished sending test data")
        print("\n🎯 IMPORTANT: Check if a matplotlib window opened showing 2 agent subplots!")
        print("   - Each subplot should show multiple cost curves")
        print("   - The plots should be updating in real-time")
        print("   - Agent 0 should be in top-left, Agent 1 in top-right")
        print("\nPress Ctrl+C to stop...")
        
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
    test_simple_plotting() 