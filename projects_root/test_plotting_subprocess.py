#!/usr/bin/env python3
"""
Simple test script to verify plotting subprocess works correctly.
"""

import time
import torch

def test_plotting_subprocess():
    print("Testing plotting subprocess...")
    
    # Start the plotting server
    from projects_root.utils.plotting_server import start_plotting_server, send_plot_data, stop_plotting_server
    
    try:
        # Start server
        print("Starting plotting server...")
        start_plotting_server(max_agents=2)
        
        # Give server time to initialize
        time.sleep(2)
        
        # Send some test data
        print("Sending test data...")
        for i in range(20):
            test_costs = {
                'total': torch.tensor([10.0 - i * 0.1]),
                'goal': torch.tensor([5.0 - i * 0.05]),
                'collision': torch.tensor([2.0 + i * 0.1])
            }
            
            # Send data for agent 0
            send_plot_data(0, test_costs)
            
            # Send data for agent 1 with different values
            test_costs_1 = {k: v + 1.0 for k, v in test_costs.items()}
            send_plot_data(1, test_costs_1)
            
            print(f"Sent test data iteration {i}")
            time.sleep(0.5)  # Half second between updates
        
        print("Test data sending complete. Window should show plots.")
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