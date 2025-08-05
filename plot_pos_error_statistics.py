import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_pos_error_statistics_with_intervals(df):
    """
    Create a plot showing maximum and mean of pos_error as lines, and variance as vertical intervals.
    
    Args:
        df: DataFrame with columns ['tstep', 'tsec', 'agent', 'pos_error']
    """
    
    # Group by tsec and calculate statistics
    stats = df.groupby('tsec').agg({
        'pos_error': ['max', 'mean', 'var']
    }).reset_index()
    
    # Flatten column names
    stats.columns = ['tsec', 'max_pos_error', 'mean_pos_error', 'var_pos_error']
    
    # Calculate standard deviation for the intervals
    stats['std_pos_error'] = np.sqrt(stats['var_pos_error'])
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot maximum and mean as lines
    plt.plot(stats['tsec'], stats['max_pos_error'], 'r-', linewidth=2, label='Maximum', alpha=0.8)
    plt.plot(stats['tsec'], stats['mean_pos_error'], 'b-', linewidth=2, label='Mean', alpha=0.8)
    
    # Plot variance as vertical intervals (error bars)
    # We'll use the mean ± standard deviation to show the spread
    plt.errorbar(stats['tsec'], stats['mean_pos_error'], 
                yerr=stats['std_pos_error'], 
                fmt='none',  # Don't plot points, just error bars
                color='g', 
                alpha=0.6, 
                capsize=3, 
                capthick=1,
                label='±1 Std Dev')
    
    # Customize the plot
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Position Error', fontsize=12)
    plt.title('Position Error Statistics Over Time\n(Mean ± Std Dev shown as vertical intervals)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Add some styling
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    return stats

def plot_pos_error_statistics_with_intervals_save(df, save_path='pos_error_statistics_intervals.png'):
    """
    Create and save a plot showing maximum and mean as lines, variance as vertical intervals.
    
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
    
    # Calculate standard deviation for the intervals
    stats['std_pos_error'] = np.sqrt(stats['var_pos_error'])
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot maximum and mean as lines
    plt.plot(stats['tsec'], stats['max_pos_error'], 'r-', linewidth=2, label='Maximum', alpha=0.8)
    plt.plot(stats['tsec'], stats['mean_pos_error'], 'b-', linewidth=2, label='Mean', alpha=0.8)
    
    # Plot variance as vertical intervals (error bars)
    plt.errorbar(stats['tsec'], stats['mean_pos_error'], 
                yerr=stats['std_pos_error'], 
                fmt='none',  # Don't plot points, just error bars
                color='g', 
                alpha=0.6, 
                capsize=3, 
                capthick=1,
                label='±1 Std Dev')
    
    # Customize the plot
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Position Error', fontsize=12)
    plt.title('Position Error Statistics Over Time\n(Mean ± Std Dev shown as vertical intervals)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Add some styling
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return stats

# Alternative version that shows the full range (min to max) as intervals
def plot_pos_error_statistics_with_range_intervals(df):
    """
    Create a plot showing mean as a line, and the full range (min to max) as vertical intervals.
    
    Args:
        df: DataFrame with columns ['tstep', 'tsec', 'agent', 'pos_error']
    """
    
    # Group by tsec and calculate statistics
    stats = df.groupby('tsec').agg({
        'pos_error': ['min', 'max', 'mean']
    }).reset_index()
    
    # Flatten column names
    stats.columns = ['tsec', 'min_pos_error', 'max_pos_error', 'mean_pos_error']
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot mean as a line
    plt.plot(stats['tsec'], stats['mean_pos_error'], 'b-', linewidth=2, label='Mean', alpha=0.8)
    
    # Plot the full range as vertical intervals
    plt.errorbar(stats['tsec'], stats['mean_pos_error'], 
                yerr=[stats['mean_pos_error'] - stats['min_pos_error'], 
                      stats['max_pos_error'] - stats['mean_pos_error']], 
                fmt='none',  # Don't plot points, just error bars
                color='g', 
                alpha=0.6, 
                capsize=3, 
                capthick=1,
                label='Min to Max Range')
    
    # Customize the plot
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Position Error', fontsize=12)
    plt.title('Position Error Statistics Over Time\n(Mean with Min-Max Range shown as vertical intervals)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Add some styling
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    return stats

# Example usage:
# If your DataFrame is called 'df':
# stats = plot_pos_error_statistics_with_intervals(df)
# 
# # Or to save:
# stats = plot_pos_error_statistics_with_intervals_save(df, 'my_plot.png')
# 
# # Or for full range intervals:
# stats = plot_pos_error_statistics_with_range_intervals(df) 