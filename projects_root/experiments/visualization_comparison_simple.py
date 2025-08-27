import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('default')
sns.set_palette("husl")

def create_algorithm_comparison_plots(df, save_path=None):
    """
    Create comprehensive comparison plots between algorithms for tplanning_moaw and ctrl_freq_moaw
    
    Args:
        df: DataFrame with columns ['alg', 'task_type', 'task_level', 'n_arms', 'wsteps', 'tplanning_moaw', 'ctrl_freq_moaw']
        save_path: Optional path to save the plots
    """
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Simple algorithm comparison (x-axis: algorithms, y-axis: metrics)
    ax1 = plt.subplot(2, 3, 1)
    df.boxplot(column='tplanning_moaw', by='alg', ax=ax1)
    ax1.set_title('Planning Time by Algorithm', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Algorithm')
    ax1.set_ylabel('Planning Time (s)')
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(2, 3, 2)
    df.boxplot(column='ctrl_freq_moaw', by='alg', ax=ax2)
    ax2.set_title('Control Frequency by Algorithm', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Algorithm')
    ax2.set_ylabel('Control Frequency (Hz)')
    ax2.grid(True, alpha=0.3)
    
    # 2. Algorithm + Task Type comparison
    ax3 = plt.subplot(2, 3, 3)
    df.boxplot(column='tplanning_moaw', by=['alg', 'task_type'], ax=ax3)
    ax3.set_title('Planning Time by Algorithm & Task Type', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Algorithm + Task Type')
    ax3.set_ylabel('Planning Time (s)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(2, 3, 4)
    df.boxplot(column='ctrl_freq_moaw', by=['alg', 'task_type'], ax=ax4)
    ax4.set_title('Control Frequency by Algorithm & Task Type', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Algorithm + Task Type')
    ax4.set_ylabel('Control Frequency (Hz)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # 3. Algorithm + Task Level comparison
    ax5 = plt.subplot(2, 3, 5)
    df.boxplot(column='tplanning_moaw', by=['alg', 'task_level'], ax=ax5)
    ax5.set_title('Planning Time by Algorithm & Task Level', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Algorithm + Task Level')
    ax5.set_ylabel('Planning Time (s)')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3)
    
    ax6 = plt.subplot(2, 3, 6)
    df.boxplot(column='ctrl_freq_moaw', by=['alg', 'task_level'], ax=ax6)
    ax6.set_title('Control Frequency by Algorithm & Task Level', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Algorithm + Task Level')
    ax6.set_ylabel('Control Frequency (Hz)')
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plots saved to {save_path}")
    
    plt.show()

def create_seaborn_plots(df, save_path=None):
    """
    Create enhanced visualizations using seaborn
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Violin plot for planning time by algorithm
    sns.violinplot(data=df, x='alg', y='tplanning_moaw', ax=axes[0,0])
    axes[0,0].set_title('Planning Time Distribution by Algorithm', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Algorithm')
    axes[0,0].set_ylabel('Planning Time (s)')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Violin plot for control frequency by algorithm
    sns.violinplot(data=df, x='alg', y='ctrl_freq_moaw', ax=axes[0,1])
    axes[0,1].set_title('Control Frequency Distribution by Algorithm', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Algorithm')
    axes[0,1].set_ylabel('Control Frequency (Hz)')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Swarm plot for planning time by algorithm and task type
    sns.swarmplot(data=df, x='alg', y='tplanning_moaw', hue='task_type', ax=axes[1,0], size=3)
    axes[1,0].set_title('Planning Time by Algorithm & Task Type', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('Algorithm')
    axes[1,0].set_ylabel('Planning Time (s)')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Swarm plot for control frequency by algorithm and task level
    sns.swarmplot(data=df, x='alg', y='ctrl_freq_moaw', hue='task_level', ax=axes[1,1], size=3)
    axes[1,1].set_title('Control Frequency by Algorithm & Task Level', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('Algorithm')
    axes[1,1].set_ylabel('Control Frequency (Hz)')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path.replace('.png', '_seaborn.png'), dpi=300, bbox_inches='tight')
        print(f"Seaborn plots saved to {save_path.replace('.png', '_seaborn.png')}")
    
    plt.show()

def create_heatmap_plots(df, save_path=None):
    """
    Create heatmap visualizations showing algorithm vs task combinations
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Heatmap for planning time by algorithm and task type
    pivot_planning_task = df.pivot_table(values='tplanning_moaw', 
                                        index='alg', 
                                        columns='task_type', 
                                        aggfunc='mean')
    sns.heatmap(pivot_planning_task, annot=True, fmt='.4f', cmap='YlOrRd', ax=axes[0,0])
    axes[0,0].set_title('Heatmap: Average Planning Time by Algorithm & Task Type', fontsize=14, fontweight='bold')
    
    # 2. Heatmap for control frequency by algorithm and task type
    pivot_freq_task = df.pivot_table(values='ctrl_freq_moaw', 
                                    index='alg', 
                                    columns='task_type', 
                                    aggfunc='mean')
    sns.heatmap(pivot_freq_task, annot=True, fmt='.2f', cmap='Blues', ax=axes[0,1])
    axes[0,1].set_title('Heatmap: Average Control Frequency by Algorithm & Task Type', fontsize=14, fontweight='bold')
    
    # 3. Heatmap for planning time by algorithm and task level
    pivot_planning_level = df.pivot_table(values='tplanning_moaw', 
                                         index='alg', 
                                         columns='task_level', 
                                         aggfunc='mean')
    sns.heatmap(pivot_planning_level, annot=True, fmt='.4f', cmap='YlOrRd', ax=axes[1,0])
    axes[1,0].set_title('Heatmap: Average Planning Time by Algorithm & Task Level', fontsize=14, fontweight='bold')
    
    # 4. Heatmap for control frequency by algorithm and task level
    pivot_freq_level = df.pivot_table(values='ctrl_freq_moaw', 
                                     index='alg', 
                                     columns='task_level', 
                                     aggfunc='mean')
    sns.heatmap(pivot_freq_level, annot=True, fmt='.2f', cmap='Blues', ax=axes[1,1])
    axes[1,1].set_title('Heatmap: Average Control Frequency by Algorithm & Task Level', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path.replace('.png', '_heatmap.png'), dpi=300, bbox_inches='tight')
        print(f"Heatmap plots saved to {save_path.replace('.png', '_heatmap.png')}")
    
    plt.show()

def create_statistical_summary(df):
    """
    Create statistical summary tables
    """
    print("=" * 80)
    print("STATISTICAL SUMMARY")
    print("=" * 80)
    
    # Summary by algorithm
    print("\n1. SUMMARY BY ALGORITHM:")
    print("-" * 40)
    summary_alg = df.groupby('alg')[['tplanning_moaw', 'ctrl_freq_moaw']].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(6)
    print(summary_alg)
    
    # Summary by algorithm and task type
    print("\n2. SUMMARY BY ALGORITHM & TASK TYPE:")
    print("-" * 40)
    summary_alg_task = df.groupby(['alg', 'task_type'])[['tplanning_moaw', 'ctrl_freq_moaw']].agg([
        'count', 'mean', 'std'
    ]).round(6)
    print(summary_alg_task)
    
    # Summary by algorithm and task level
    print("\n3. SUMMARY BY ALGORITHM & TASK LEVEL:")
    print("-" * 40)
    summary_alg_level = df.groupby(['alg', 'task_level'])[['tplanning_moaw', 'ctrl_freq_moaw']].agg([
        'count', 'mean', 'std'
    ]).round(6)
    print(summary_alg_level)
    
    return summary_alg, summary_alg_task, summary_alg_level

def load_your_data():
    """
    Replace this function with your actual data loading logic
    """
    # This is where you would load your actual data
    # For example:
    # df = pd.read_csv('your_data.csv')
    # or
    # df = your_data_loading_function()
    
    # For demonstration, I'll create sample data
    sample_data = {
        'alg': ['SD', 'SC', 'SC', 'CC', 'SC', 'SD', 'CC', 'SC', 'SD', 'CC'] * 10,
        'task_type': ['bin', 'bin', 'bin', 'bin', 'bin', 'reach', 'reach', 'reach', 'follow', 'follow'] * 10,
        'task_level': [1, 5, 3, 3, 2, 1, 2, 4, 1, 3] * 10,
        'n_arms': [4, 4, 4, 4, 4, 2, 2, 2, 1, 1] * 10,
        'wsteps': [500, 500, 500, 500, 500, 300, 300, 300, 200, 200] * 10,
        'tplanning_moaw': np.random.normal(0.05, 0.01, 100),
        'ctrl_freq_moaw': np.random.normal(20, 2, 100)
    }
    
    df = pd.DataFrame(sample_data)
    df['tplanning_moaw'] = np.abs(df['tplanning_moaw'])
    df['ctrl_freq_moaw'] = np.abs(df['ctrl_freq_moaw'])
    
    return df

def main():
    """
    Main function to run all visualizations
    """
    print("Algorithm Comparison Visualization Tool")
    print("=" * 50)
    
    # Load your data here
    print("Loading data...")
    df = load_your_data()
    
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Unique algorithms: {df['alg'].unique()}")
    print(f"Unique task types: {df['task_type'].unique()}")
    print(f"Unique task levels: {df['task_level'].unique()}")
    print("\nFirst few rows:")
    print(df.head())
    
    # Create all visualizations
    print("\nCreating visualizations...")
    
    # 1. Basic comparison plots
    print("Creating basic comparison plots...")
    create_algorithm_comparison_plots(df, 'algorithm_comparison.png')
    
    # 2. Seaborn enhanced plots
    print("Creating seaborn enhanced plots...")
    create_seaborn_plots(df, 'algorithm_comparison.png')
    
    # 3. Heatmap plots
    print("Creating heatmap plots...")
    create_heatmap_plots(df, 'algorithm_comparison.png')
    
    # 4. Statistical summary
    print("Creating statistical summary...")
    create_statistical_summary(df)
    
    print("\nAll visualizations completed!")

if __name__ == "__main__":
    main() 