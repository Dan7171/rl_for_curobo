import pandas as pd
import numpy as np
from plot_pos_error_statistics import plot_pos_error_statistics, plot_pos_error_statistics_save, plot_pos_error_statistics_subplots

# Create sample data similar to your DataFrame
def create_sample_data():
    """Create sample data similar to the user's DataFrame structure"""
    np.random.seed(42)  # For reproducible results
    
    # Create time steps
    tsteps = np.arange(0, 904)  # 0 to 903
    tsecs = np.linspace(0, 39.50625, len(tsteps))
    
    # Create agents (0-31)
    agents = np.arange(32)
    
    # Create data
    data = []
    for tstep, tsec in zip(tsteps, tsecs):
        for agent in agents:
            # Simulate decreasing position error over time with some noise
            base_error = 3.4 * np.exp(-tsec / 10) + 0.01  # Exponential decay
            noise = np.random.normal(0, 0.1)
            pos_error = max(0, base_error + noise)
            
            data.append({
                'tstep': tstep,
                'tsec': tsec,
                'agent': agent,
                'pos_error': pos_error
            })
    
    return pd.DataFrame(data)

# Example usage
if __name__ == "__main__":
    # Create sample data (replace this with your actual DataFrame)
    df = create_sample_data()
    
    print("Sample data shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    print("\nLast few rows:")
    print(df.tail())
    
    print("\nUnique time steps:", len(df['tsec'].unique()))
    print("Number of agents:", len(df['agent'].unique()))
    
    # Option 1: Single plot with all statistics
    print("\nCreating single plot with all statistics...")
    stats = plot_pos_error_statistics(df)
    
    # Option 2: Save the plot
    print("\nSaving plot...")
    stats_save = plot_pos_error_statistics_save(df, 'pos_error_statistics.png')
    
    # Option 3: Subplots
    print("\nCreating subplots...")
    stats_subplots = plot_pos_error_statistics_subplots(df)
    
    # Print some statistics
    print("\nSummary statistics:")
    print("Final maximum position error:", stats['max_pos_error'].iloc[-1])
    print("Final mean position error:", stats['mean_pos_error'].iloc[-1])
    print("Final variance:", stats['var_pos_error'].iloc[-1]) 