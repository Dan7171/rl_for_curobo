import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_pos_error_statistics(df):
    """
    Create a plot showing maximum, mean, and variance of pos_error over all agents for each tsec.
    
    Args:
        df: DataFrame with columns ['tstep', 'tsec', 'agent', 'pos_error']
    """
    
    # Group by tsec and calculate statistics
    stats = df.groupby('tsec').agg({
        'pos_error': ['max', 'mean', 'var']
    }).reset_index()
    
    # Flatten column names
    stats.columns = ['tsec', 'max_pos_error', 'mean_pos_error', 'var_pos_error']
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot each statistic
    plt.plot(stats['tsec'], stats['max_pos_error'], 'r-', linewidth=2, label='Maximum', alpha=0.8)
    plt.plot(stats['tsec'], stats['mean_pos_error'], 'b-', linewidth=2, label='Mean', alpha=0.8)
    plt.plot(stats['tsec'], stats['var_pos_error'], 'g-', linewidth=2, label='Variance', alpha=0.8)
    
    # Customize the plot
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Position Error', fontsize=12)
    plt.title('Position Error Statistics Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Add some styling
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    return stats

# Example usage with your data:
# Replace 'your_dataframe' with the actual name of your DataFrame
# stats = plot_pos_error_statistics(your_dataframe)

# If you want to save the plot instead of showing it:
def plot_pos_error_statistics_save(df, save_path='pos_error_statistics.png'):
    """
    Create and save a plot showing maximum, mean, and variance of pos_error over all agents for each tsec.
    
    Args:
        df: DataFrame with columns ['tstep', 'tsec', 'agent', 'pos_error']
        save_path: Path to save the plot image
    """
    
    # Group by tsec and calculate statistics
    stats = df.groupby('tsec').agg({
        'pos_error': ['max', 'mean', 'var']
    }).reset_index()
    
    # Flatten column names
    stats.columns = ['tsec', 'max_pos_error', 'mean_pos_error', 'var_pos_error']
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot each statistic
    plt.plot(stats['tsec'], stats['max_pos_error'], 'r-', linewidth=2, label='Maximum', alpha=0.8)
    plt.plot(stats['tsec'], stats['mean_pos_error'], 'b-', linewidth=2, label='Mean', alpha=0.8)
    plt.plot(stats['tsec'], stats['var_pos_error'], 'g-', linewidth=2, label='Variance', alpha=0.8)
    
    # Customize the plot
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Position Error', fontsize=12)
    plt.title('Position Error Statistics Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Add some styling
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return stats

# Quick usage example:
# If your DataFrame is called 'df':
# 
# # Option 1: Show the plot
# stats = plot_pos_error_statistics(df)
# 
# # Option 2: Save the plot
# stats = plot_pos_error_statistics_save(df, 'my_plot.png')
# 
# # Print some summary statistics
# print("Final maximum position error:", stats['max_pos_error'].iloc[-1])
# print("Final mean position error:", stats['mean_pos_error'].iloc[-1])
# print("Final variance:", stats['var_pos_error'].iloc[-1]) 