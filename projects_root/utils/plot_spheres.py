from typing_extensions import Optional
import torch
import matplotlib
# Set backend before importing pyplot
try:
    import tkinter
    matplotlib.use('TkAgg')  # Use TkAgg backend if tkinter is available
except ImportError:
    try:
        matplotlib.use('Qt5Agg')  # Try Qt5Agg as fallback
    except ImportError:
        matplotlib.use('Agg')  # Use Agg (non-interactive) as last resort
        print("⚠️  Using non-interactive matplotlib backend. Plots will be saved but not displayed.")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import numpy as np
import threading
import time
import signal
import sys
from collections import defaultdict
import matplotlib.animation as animation


class SphereVisualizer:
    
    def __init__(self, group_ids: list[int], 
                 figsize=(12, 8), 
                 update_interval=50,
                 show_trajectory_trails=True):
        """
        Initialize the sphere visualizer.
        
        Args:
            group_ids: List of group IDs to visualize
            figsize: Figure size for the plot
            update_interval: Update interval in milliseconds for animation
            show_trajectory_trails: Whether to show trajectory trails
        """
        self.group_ids = group_ids
        self.group_colors = {
            0: (1.0, 0.0, 0.0),    # Red
            1: (0.0, 1.0, 0.0),    # Green  
            2: (0.0, 0.0, 1.0),    # Blue
            3: (1.0, 1.0, 0.0),    # Yellow
            4: (0.0, 1.0, 1.0),    # Cyan
            5: (1.0, 0.0, 1.0),    # Magenta
            6: (0.5, 0.5, 0.5),    # Gray
            7: (1.0, 0.5, 0.0),    # Orange
            8: (0.5, 0.0, 1.0),    # Purple
            9: (0.0, 0.5, 0.5),    # Teal
        }
        self.default_radius = 0.01
        self.show_trajectory_trails = show_trajectory_trails
        
        # Data storage for each group
        self.sphere_data = defaultdict(dict)  # {group_id: {'positions': tensor, 'radii': tensor, 'timestamp': float}}
        self.trajectory_history = defaultdict(list)  # {group_id: [positions_over_time]}
        
        # Plotting setup
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.update_interval = update_interval
        
        # Plot elements storage
        self.sphere_plots = defaultdict(list)  # {group_id: [plot_objects]}
        self.trail_plots = defaultdict(list)   # {group_id: [trail_objects]}
        
        # Animation and threading
        self.animation = None
        self.is_running = False
        self.lock = threading.Lock()
        
        # Check if we can display plots
        self.can_display = matplotlib.get_backend() != 'Agg'
        
        # Initialize the plot
        self._setup_plot()
        if self.can_display:
            self._launch_live_plotting()
        else:
            print("📊 Running in non-interactive mode")
    
    def _setup_plot(self):
        """Setup the 3D plot with proper labels and limits."""
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('Sphere Trajectories Visualization')
        
        # Set initial limits (will be updated dynamically)
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_zlim(0, 2)
        
        # Add grid
        self.ax.grid(True, alpha=0.3)
        
        # Create legend
        legend_elements = []
        for group_id in self.group_ids:
            color = self.group_colors.get(group_id, (0.5, 0.5, 0.5))
            legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color, markersize=10, 
                                        label=f'Group {group_id}'))
        
        if legend_elements:
            self.ax.legend(handles=legend_elements, loc='upper right')
    
    def _launch_live_plotting(self):
        """Launch the live plotting with animation."""
        self.is_running = True
        
        # Show plot first
        plt.ion()
        plt.show(block=False)
        plt.draw()
        
        # Create animation after showing the plot
        self.animation = animation.FuncAnimation(
            self.fig, self._update_plot, interval=self.update_interval, 
            blit=False, cache_frame_data=False
        )
    
    def _update_plot(self, frame):
        """Update the plot with current sphere data."""
        with self.lock:
            # Clear previous plots
            for group_plots in self.sphere_plots.values():
                for plot_obj in group_plots:
                    if hasattr(plot_obj, 'remove'):
                        plot_obj.remove()
            
            for trail_plots in self.trail_plots.values():
                for plot_obj in trail_plots:
                    if hasattr(plot_obj, 'remove'):
                        plot_obj.remove()
            
            self.sphere_plots.clear()
            self.trail_plots.clear()
            
            # Plot current sphere data
            all_positions = []
            
            for group_id, data in self.sphere_data.items():
                if 'positions' not in data:
                    continue
                
                positions = data['positions']  # Shape: (b, h, n, 3)
                radii = data.get('radii', None)
                
                if positions.numel() == 0:
                    continue
                
                # Convert to numpy for plotting
                if isinstance(positions, torch.Tensor):
                    positions_np = positions.detach().cpu().numpy()
                else:
                    positions_np = positions
                
                if isinstance(radii, torch.Tensor) and radii is not None:
                    radii_np = radii.detach().cpu().numpy()
                else:
                    radii_np = None
                
                b, h, n, _ = positions_np.shape
                color = self.group_colors.get(group_id, (0.5, 0.5, 0.5))
                
                group_plots = []
                
                for traj_idx in range(b):
                    # Calculate alpha based on trajectory index (fade out later trajectories)
                    alpha = 1.0 - (traj_idx / b) * 0.7
                    
                    # Plot spheres for the latest time step
                    latest_positions = positions_np[traj_idx, -1, :, :]  # Shape: (n, 3)
                    
                    # Collect all positions for axis limits
                    all_positions.extend(latest_positions.tolist())
                    
                    for sphere_idx in range(n):
                        x, y, z = latest_positions[sphere_idx]
                        
                        # Determine sphere radius
                        if radii_np is not None and sphere_idx < len(radii_np):
                            radius = radii_np[sphere_idx]
                        else:
                            radius = self.default_radius
                        
                        # Scale radius for visualization (make it more visible)
                        vis_radius = max(radius * 100, 0.05)
                        
                        # Plot sphere as a scatter point
                        scatter = self.ax.scatter(x, y, z, 
                                                color=color, 
                                                s=vis_radius*1000,  # Size for scatter
                                                alpha=alpha,
                                                edgecolors='black',
                                                linewidth=0.5)
                        group_plots.append(scatter)
                    
                    # Plot trajectory trails if enabled
                    if self.show_trajectory_trails and h > 1:
                        # Use full horizon for trails (from start to current time)
                        trail_start = 0
                        
                        for sphere_idx in range(n):
                            trail_positions = positions_np[traj_idx, trail_start:, sphere_idx, :]
                            
                            if len(trail_positions) > 1:
                                trail = self.ax.plot(trail_positions[:, 0], 
                                                   trail_positions[:, 1], 
                                                   trail_positions[:, 2],
                                                   color=color, 
                                                   alpha=alpha*0.5, 
                                                   linewidth=1.0)
                                self.trail_plots[group_id].extend(trail)
                
                self.sphere_plots[group_id] = group_plots
            
            # Update axis limits based on all positions
            if all_positions:
                all_positions_array = np.array(all_positions)
                margin = 0.5
                
                x_min, x_max = all_positions_array[:, 0].min() - margin, all_positions_array[:, 0].max() + margin
                y_min, y_max = all_positions_array[:, 1].min() - margin, all_positions_array[:, 1].max() + margin
                z_min, z_max = all_positions_array[:, 2].min() - margin, all_positions_array[:, 2].max() + margin
                
                self.ax.set_xlim(x_min, x_max)
                self.ax.set_ylim(y_min, y_max)
                self.ax.set_zlim(z_min, z_max)
        
        return []
    
    def update(self, P: torch.Tensor, group_id: int, r: Optional[torch.Tensor] = None):
        """
        Update the sphere positions for a specific group.
        
        Args:
            P: Tensor of sphere positions with shape (b, h, n, 3)
               where:
               - b = batch size (number of trajectories)
               - h = horizon (number of time steps)
               - n = number of spheres
               - 3 = position coordinates (x, y, z)
            group_id: Integer identifier for the group
            r: Optional tensor of sphere radii with shape (n,)
        """
        if not isinstance(P, torch.Tensor):
            P = torch.tensor(P)
        
        # Validate input shape
        if P.dim() != 4 or P.shape[-1] != 3:
            raise ValueError(f"Expected P to have shape (b, h, n, 3), got {P.shape}")
        
        if r is not None and not isinstance(r, torch.Tensor):
            r = torch.tensor(r)
        
        with self.lock:
            # Store the data
            self.sphere_data[group_id] = {
                'positions': P.clone(),
                'radii': r.clone() if r is not None else None,
                'timestamp': time.time()
            }
            
            # Update trajectory history for trails
            if self.show_trajectory_trails:
                self.trajectory_history[group_id].append(P.clone())
                
                # Limit history length
                max_history = 50
                if len(self.trajectory_history[group_id]) > max_history:
                    self.trajectory_history[group_id] = self.trajectory_history[group_id][-max_history:]
        
        # Force immediate plot update if interactive
        if self.can_display and hasattr(self, 'fig'):
            plt.draw()
            plt.pause(0.001)
    
    def clear_group(self, group_id: int):
        """Clear data for a specific group."""
        with self.lock:
            if group_id in self.sphere_data:
                del self.sphere_data[group_id]
            if group_id in self.trajectory_history:
                del self.trajectory_history[group_id]
    
    def clear_all(self):
        """Clear all sphere data."""
        with self.lock:
            self.sphere_data.clear()
            self.trajectory_history.clear()
    
    def set_view(self, elev=20, azim=45):
        """Set the 3D view angle."""
        self.ax.view_init(elev=elev, azim=azim)
        if self.can_display:
            plt.draw()
    
    def close(self):
        """Close the visualization and clean up resources."""
        self.is_running = False
        try:
            if self.animation and hasattr(self.animation, 'event_source') and self.animation.event_source:
                self.animation.event_source.stop()
        except:
            pass
        try:
            plt.close(self.fig)
        except:
            pass
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.close()
        except:
            pass


# Convenience function for quick visualization
def visualize_spheres(positions_dict: dict, 
                     radii_dict: Optional[dict] = None,
                     figsize=(12, 8)):
    """
    Convenience function for quick sphere visualization.
    
    Args:
        positions_dict: Dictionary {group_id: positions_tensor}
        radii_dict: Optional dictionary {group_id: radii_tensor}
        figsize: Figure size
    
    Returns:
        SphereVisualizer instance
    """
    group_ids = list(positions_dict.keys())
    visualizer = SphereVisualizer(group_ids, figsize=figsize)
    
    for group_id, positions in positions_dict.items():
        radii = radii_dict.get(group_id) if radii_dict else None
        visualizer.update(positions, group_id, radii)
    
    return visualizer


def create_demo_trajectories():
    """Create interesting demo trajectories for visualization."""
    print("🎬 Creating demo trajectories...")
    
    # Parameters
    b, h, n = 4, 25, 6  # 4 trajectories, 25 time steps, 6 spheres each
    
    # Time parameter
    t = torch.linspace(0, 4*np.pi, h)
    
    # Group 0: Robot arm trajectories (Red)
    robot_trajectories = torch.zeros(b, h, n, 3)
    for traj_idx in range(b):
        for sphere_idx in range(n):
            # Create arm-like motion with different joints
            base_radius = 0.3 + sphere_idx * 0.15
            height_offset = 0.5 + traj_idx * 0.2
            phase_offset = traj_idx * np.pi/4 + sphere_idx * np.pi/8
            
            # Circular motion with varying height
            robot_trajectories[traj_idx, :, sphere_idx, 0] = base_radius * torch.cos(t + phase_offset)
            robot_trajectories[traj_idx, :, sphere_idx, 1] = base_radius * torch.sin(t + phase_offset) 
            robot_trajectories[traj_idx, :, sphere_idx, 2] = height_offset + 0.3 * torch.sin(2*t + phase_offset)
    
    # Group 1: Dynamic obstacles (Green)
    obstacle_trajectories = torch.zeros(b, h, n, 3)
    for traj_idx in range(b):
        for sphere_idx in range(n):
            # Create obstacle motion patterns
            speed = 0.8 + traj_idx * 0.2
            direction = traj_idx * np.pi/2
            
            # Linear motion with sinusoidal perturbations
            obstacle_trajectories[traj_idx, :, sphere_idx, 0] = -1.5 + speed * t/4 * np.cos(direction) + 0.2 * torch.sin(3*t + sphere_idx)
            obstacle_trajectories[traj_idx, :, sphere_idx, 1] = -1.0 + speed * t/4 * np.sin(direction) + 0.2 * torch.cos(2*t + sphere_idx)
            obstacle_trajectories[traj_idx, :, sphere_idx, 2] = 0.8 + 0.3 * torch.sin(t + sphere_idx * np.pi/3)
    
    # Group 2: Swarm behavior (Blue)
    swarm_trajectories = torch.zeros(b, h, n, 3)
    center_x, center_y, center_z = 1.0, 1.0, 1.2
    for traj_idx in range(b):
        for sphere_idx in range(n):
            # Swarm orbiting around a center point
            orbit_radius = 0.4 + sphere_idx * 0.1
            orbit_speed = 1.0 + traj_idx * 0.3
            vertical_freq = 2.0 + sphere_idx * 0.5
            
            angle = orbit_speed * t + sphere_idx * 2*np.pi/n + traj_idx * np.pi/4
            swarm_trajectories[traj_idx, :, sphere_idx, 0] = center_x + orbit_radius * torch.cos(angle)
            swarm_trajectories[traj_idx, :, sphere_idx, 1] = center_y + orbit_radius * torch.sin(angle)
            swarm_trajectories[traj_idx, :, sphere_idx, 2] = center_z + 0.2 * torch.sin(vertical_freq * t + sphere_idx)
    
    # Create radii for each group
    robot_radii = torch.linspace(0.05, 0.12, n)  # Varying sizes for robot links
    obstacle_radii = torch.ones(n) * 0.08  # Uniform obstacle sizes
    swarm_radii = torch.ones(n) * 0.06  # Small swarm particles
    
    return {
        'trajectories': {
            0: robot_trajectories,
            1: obstacle_trajectories, 
            2: swarm_trajectories
        },
        'radii': {
            0: robot_radii,
            1: obstacle_radii,
            2: swarm_radii
        }
    }


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    print("\n⏹️  Received interrupt signal, cleaning up...")
    plt.close('all')
    sys.exit(0)


def run_interactive_demo():
    """Run an interactive demo with live updates."""
    print("🚀 Starting Interactive Sphere Visualization Demo")
    print("=" * 60)
    
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # Check matplotlib backend
    backend = matplotlib.get_backend()
    print(f"📊 Using matplotlib backend: {backend}")
    
    # Create demo data
    demo_data = create_demo_trajectories()
    
    # Initialize visualizer
    group_ids = [0, 1, 2]
    visualizer = SphereVisualizer(
        group_ids=group_ids,
        figsize=(14, 10),
        update_interval=100,  # Update every 100ms
        show_trajectory_trails=True
    )
    
    print("📊 Visualizer initialized with:")
    print(f"   - Groups: {group_ids} (Red=Robot, Green=Obstacles, Blue=Swarm)")
    print(f"   - Trajectories per group: {demo_data['trajectories'][0].shape[0]}")
    print(f"   - Time steps: {demo_data['trajectories'][0].shape[1]}")
    print(f"   - Spheres per trajectory: {demo_data['trajectories'][0].shape[2]}")
    
    # Set a nice viewing angle
    visualizer.set_view(elev=25, azim=45)
    
    try:
        # Simulate live updates
        print("\n🎬 Starting live trajectory updates...")
        if visualizer.can_display:
            print("   (Press Ctrl+C to stop, or close the plot window)")
        else:
            print("   (Running in non-interactive mode)")
        
        for step in range(50):  # Run for 50 update cycles
            # Update each group with current trajectory data
            for group_id in group_ids:
                trajectories = demo_data['trajectories'][group_id]
                # radii = demo_data['radii'][group_id]
                radii = None
                # Simulate progressive trajectory updates
                current_step = min(step + 5, trajectories.shape[1] - 1)
                current_trajectories = trajectories[:, :current_step+1, :, :]
                
                visualizer.update(current_trajectories, group_id, radii)
            
            # Print progress
            if step % 10 == 0:
                print(f"   Step {step}/50 - Trajectories updated")
            
            # Wait before next update
            time.sleep(0.2)
        
        print("\n✅ Demo completed successfully!")
        
        if visualizer.can_display:
            # Keep the plot open for viewing
            print("📺 Plot window will stay open for 10 seconds...")
            print("   You can interact with the 3D plot (rotate, zoom, etc.)")
            time.sleep(10)
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    
    finally:
        # Clean up
        visualizer.close()
        print("🧹 Visualization closed")


def run_static_demo():
    """Run a static demo using the convenience function."""
    print("\n🖼️  Running Static Demo with Convenience Function")
    print("=" * 50)
    
    # Create simpler demo data
    b, h, n = 2, 15, 4
    
    # Create two different trajectory patterns
    positions_dict = {}
    radii_dict = {}
    
    # Group 0: Helical trajectories
    t = torch.linspace(0, 2*np.pi, h)
    helix_traj = torch.zeros(b, h, n, 3)
    for i in range(b):
        for j in range(n):
            radius = 0.5 + i * 0.3 + j * 0.1
            helix_traj[i, :, j, 0] = radius * torch.cos(t + j * np.pi/2)
            helix_traj[i, :, j, 1] = radius * torch.sin(t + j * np.pi/2)
            helix_traj[i, :, j, 2] = 0.5 + t/np.pi + i * 0.3
    
    positions_dict[0] = helix_traj
    radii_dict[0] = torch.ones(n) * 0.08
    
    # Group 1: Figure-8 trajectories  
    fig8_traj = torch.zeros(b, h, n, 3)
    for i in range(b):
        for j in range(n):
            scale = 0.6 + i * 0.2
            fig8_traj[i, :, j, 0] = scale * torch.sin(t + j * np.pi/4)
            fig8_traj[i, :, j, 1] = scale * torch.sin(2*t + j * np.pi/4)
            fig8_traj[i, :, j, 2] = 1.2 + i * 0.2 + 0.1 * torch.cos(3*t + j)
    
    positions_dict[1] = fig8_traj
    radii_dict[1] = torch.ones(n) * 0.06
    
    # Use convenience function
    viz = visualize_spheres(
        positions_dict=positions_dict,
        radii_dict=radii_dict,
        figsize=(12, 8)
    )
    
    print("✅ Static demo created!")
    if viz.can_display:
        print("📺 Keeping plot open for 5 seconds...")
        time.sleep(5)
    viz.close()


if __name__ == "__main__":
    print("🌟 Sphere Visualizer Demo")
    print("=" * 40)
    print("This demo showcases the SphereVisualizer capabilities:")
    print("• Multiple trajectory groups with different colors")
    print("• Real-time animation with trajectory trails") 
    print("• Dynamic sphere sizing and transparency")
    print("• Automatic axis scaling and 3D visualization")
    print()
    
    # Print system info
    print(f"🖥️  System info:")
    print(f"   - PyTorch: {torch.__version__}")
    print(f"   - Matplotlib: {matplotlib.__version__}")
    print(f"   - Backend: {matplotlib.get_backend()}")
    print()
    
    try:
        # Run interactive demo
        run_interactive_demo()
        
        # Run static demo
        run_static_demo()
        
        print("\n🎉 All demos completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Ensure matplotlib doesn't keep running
        plt.close('all')
        print("\n👋 Demo finished!")


