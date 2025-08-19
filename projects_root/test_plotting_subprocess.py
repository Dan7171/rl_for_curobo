#!/usr/bin/env python3
"""
Test script to verify the plotting subprocess works correctly.
Sends sample cost data to demonstrate real-time plotting functionality.
"""

import time
import torch

def test_plotting_subprocess():
    """Test the plotting server with sample data."""
    print("Testing plotting subprocess...")
    
    # Import plotting server functions
    from projects_root.utils.plotting_server import start_plotting_server, send_plot_data, stop_plotting_server
    
    try:
        # Start the plotting server for 2 agents
        print("Starting plotting server...")
        start_plotting_server(max_agents=2)
        
        # Give server time to initialize
        time.sleep(2)
        
        # Send test data with varying cost values
        print("Sending test data...")
        for i in range(20):
            # Create sample cost data with changing values
            test_costs = {
                'total': torch.tensor([10.0 - i * 0.1]),
                'goal': torch.tensor([5.0 - i * 0.05]),  
                'collision': torch.tensor([2.0 + i * 0.1])
            }
            
            # Send data for both agents with different values
            send_plot_data(0, test_costs)
            send_plot_data(1, {k: v + 1.0 for k, v in test_costs.items()})
            
            time.sleep(0.5)  # Update every 500ms
        
        print("Test data complete. Check that plot window shows cost curves.")
        print("Press Ctrl+C to stop...")
        
        # Keep running until interrupted
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("Stopping test...")
    
    finally:
        print("Stopping plotting server...")
        stop_plotting_server()
        print("Test complete.")

if __name__ == "__main__":
    test_plotting_subprocess() 