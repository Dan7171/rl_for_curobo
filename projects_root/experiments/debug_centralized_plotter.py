#!/usr/bin/env python3
"""
Debug version of centralized plotter test with verbose output.
"""

import sys
import time
import threading
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add the curobo path
sys.path.append('/home/dan/rl_for_curobo/curobo/src')

def debug_centralized_plotter():
    """Debug the centralized plotter step by step."""
    
    print("=== Debugging Centralized Plotter ===")
    
    try:
        # Import the plotter
        from curobo.rollout.arm_reacher import get_global_plotter
        print("✓ Import successful")
        
        # Check matplotlib state
        print(f"Matplotlib backend: {plt.get_backend()}")
        print(f"Interactive mode: {plt.isinteractive()}")
        
        # Initialize plotter
        plotter = get_global_plotter()
        print("✓ Plotter instance created")
        
        # Enable plotting
        plotter.enable_plotting(max_agents=1)
        print("✓ Plotter enabled")
        
        # Check if figure was created
        if plotter.fig is not None:
            print(f"✓ Figure created: {type(plotter.fig)}")
            print(f"  Figure number: {plotter.fig.number}")
            print(f"  Figure size: {plotter.fig.get_size_inches()}")
            print(f"  Axes count: {len(plotter.axes)}")
        else:
            print("❌ Figure is None!")
            return
        
        # Check if figure is in pyplot's figure list
        all_figs = plt.get_fignums()
        print(f"All matplotlib figures: {all_figs}")
        
        # Try to show the figure explicitly
        print("Attempting to show figure...")
        plt.figure(plotter.fig.number)
        plt.show(block=False)
        plt.draw()
        
        # Send some test data
        print("Sending test data...")
        for i in range(20):
            cost_dict = {
                'total': torch.tensor(50 + 10 * np.sin(i * 0.3)),
                'goal': torch.tensor(20 + 5 * np.cos(i * 0.2)),
            }
            plotter.add_data(0, cost_dict)
            
            # Force update
            plotter.update_plots()
            
            # Force redraw
            plt.figure(plotter.fig.number)
            plt.draw()
            plt.pause(0.1)
            
            if i % 5 == 0:
                print(f"  Sent data point {i}, queue size: {plotter.data_queue.qsize()}")
        
        print("\n🎯 DEBUG QUESTIONS:")
        print("1. Do you see a matplotlib window titled 'Multi-Agent Cost Monitoring'?")
        print("2. Is there a subplot labeled 'Agent 0 - Cost Monitoring'?")
        print("3. Are there two curves labeled 'total' and 'goal'?")
        print("\nKeeping window open for 10 seconds...")
        
        # Keep showing for 10 seconds
        for i in range(100):
            plt.pause(0.1)
        
        print("✓ Debug completed")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        try:
            if 'plotter' in locals():
                plotter.shutdown()
            plt.close('all')
        except:
            pass

if __name__ == "__main__":
    debug_centralized_plotter() 