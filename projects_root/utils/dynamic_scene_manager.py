"""
Dynamic Scene Manager for Multi-Robot Systems

This module provides reusable components for managing dynamic targets,
robot visual markers, and central obstacles in multi-robot scenarios.
"""

import time
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
import threading
from omni.isaac.core.prims import GeometryPrim
from omni.isaac.core.utils.prims import create_prim
from pxr import Gf, UsdGeom, Sdf
from projects_root.utils.colors import npColors


class DynamicTargetManager:
    """Manages dynamic target positions that change over time"""
    
    def __init__(self, n_robots: int, center_position: List[float] = [0.0, 0.0, 0.5],
                 target_area_radius: float = 0.4, target_height_range: Tuple[float, float] = (0.3, 0.8),
                 update_interval: Tuple[float, float] = (5.0, 10.0)):
        """
        Initialize dynamic target manager
        
        Args:
            n_robots: Number of robots
            center_position: Center position around which targets are distributed
            target_area_radius: Radius of area where targets can be placed
            target_height_range: Min/max height for targets (z-coordinate)
            update_interval: Min/max seconds between target updates
        """
        self.n_robots = n_robots
        self.center_position = np.array(center_position)
        self.target_area_radius = target_area_radius
        self.target_height_range = target_height_range
        self.update_interval = update_interval
        
        # Initialize all targets at center position
        self.current_targets = [self.center_position.copy() for _ in range(n_robots)]
        self.target_colors = self._assign_target_colors()
        
        # Threading for async updates
        self._stop_updates = False
        self._update_thread = None
        self._callbacks = []
        
    def _assign_target_colors(self) -> List[np.ndarray]:
        """Assign colors to targets based on number of robots"""
        # Use available colors plus some custom ones
        cyan = np.array([0, 0.5, 0.5])
        magenta = np.array([0.5, 0, 0.5])
        colors = [npColors.red, npColors.green, npColors.blue, npColors.yellow, 
                 cyan, magenta, npColors.orange, npColors.purple]
        return [colors[i % len(colors)] for i in range(self.n_robots)]
    
    def generate_random_target_position(self, avoid_center_radius: float = 0.15) -> np.ndarray:
        """Generate a random target position around the center"""
        while True:
            # Generate random angle and radius
            angle = random.uniform(0, 2 * np.pi)
            # Use square root for uniform distribution in circle
            radius = random.uniform(avoid_center_radius, self.target_area_radius) * np.sqrt(random.random())
            
            # Calculate position
            x = self.center_position[0] + radius * np.cos(angle)
            y = self.center_position[1] + radius * np.sin(angle)
            z = random.uniform(*self.target_height_range)
            
            position = np.array([x, y, z])
            
            # Ensure minimum distance from center (to avoid bin/table)
            if np.linalg.norm(position[:2] - self.center_position[:2]) >= avoid_center_radius:
                return position
    
    def update_targets(self, num_targets_to_change: Optional[int] = None) -> Dict[int, np.ndarray]:
        """
        Update random targets to new positions
        
        Args:
            num_targets_to_change: Number of targets to change (random if None)
            
        Returns:
            Dictionary mapping robot_id to new target position
        """
        if num_targets_to_change is None:
            num_targets_to_change = random.randint(1, max(1, self.n_robots // 2))
        
        # Select random robots to update
        robots_to_update = random.sample(range(self.n_robots), 
                                       min(num_targets_to_change, self.n_robots))
        
        updated_targets = {}
        for robot_id in robots_to_update:
            new_position = self.generate_random_target_position()
            self.current_targets[robot_id] = new_position
            updated_targets[robot_id] = new_position
            
        # Notify callbacks
        for callback in self._callbacks:
            callback(updated_targets)
            
        return updated_targets
    
    def get_target_position(self, robot_id: int) -> np.ndarray:
        """Get current target position for a robot"""
        return self.current_targets[robot_id].copy()
    
    def get_all_targets(self) -> List[np.ndarray]:
        """Get all current target positions"""
        return [target.copy() for target in self.current_targets]
    
    def get_target_color(self, robot_id: int) -> np.ndarray:
        """Get target color for a robot"""
        return self.target_colors[robot_id]
    
    def add_update_callback(self, callback):
        """Add callback function to be called when targets update"""
        self._callbacks.append(callback)
    
    def start_automatic_updates(self):
        """Start automatic target updates in background thread"""
        if self._update_thread is None or not self._update_thread.is_alive():
            self._stop_updates = False
            self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self._update_thread.start()
    
    def stop_automatic_updates(self):
        """Stop automatic target updates"""
        self._stop_updates = True
        if self._update_thread:
            self._update_thread.join()
    
    def _update_loop(self):
        """Background thread loop for automatic updates"""
        while not self._stop_updates:
            # Wait for random interval
            wait_time = random.uniform(*self.update_interval)
            time.sleep(wait_time)
            
            if not self._stop_updates:
                updated_targets = self.update_targets()
                print(f"🎯 TARGET UPDATE at time {time.time():.1f}s")
                print(f"   Updated {len(updated_targets)} robots: {list(updated_targets.keys())}")
                for robot_id, pos in updated_targets.items():
                    print(f"   Robot {robot_id}: new target at [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                print(f"   Next update in {random.uniform(*self.update_interval):.1f}s")


class RobotColorMarker:
    """Manages visual color markers for robots"""
    
    def __init__(self, stage, robot_positions: List[np.ndarray], 
                 target_colors: List[np.ndarray], marker_type: str = "sphere", 
                 marker_size: float = 0.08, height_offset: float = 0.3):
        """
        Initialize robot color markers
        
        Args:
            stage: USD stage for creating visual elements
            robot_positions: List of robot base positions
            target_colors: List of colors corresponding to targets
            marker_type: "sphere" or "cube" for marker shape
            marker_size: Size of the marker (radius for sphere, side length for cube)
            height_offset: Height above the robot base to place markers
        """
        self.stage = stage
        self.robot_positions = robot_positions
        self.target_colors = target_colors
        self.marker_type = marker_type
        self.marker_size = marker_size
        self.height_offset = height_offset
        self.markers = []
        self.robots = None  # Will be set later to track end-effector positions
        
        self._create_markers()
    
    def _create_markers(self):
        """Create visual markers for each robot"""
        for i, (pos, color) in enumerate(zip(self.robot_positions, self.target_colors)):
            marker_path = f"/World/robot_marker_{i}"
            
            if self.marker_type == "sphere":
                marker_prim = create_prim(marker_path, "Sphere")
                # Set sphere radius - larger and more visible
                sphere = UsdGeom.Sphere(marker_prim)
                sphere.GetRadiusAttr().Set(self.marker_size)
            else:  # cube
                marker_prim = create_prim(marker_path, "Cube")
                # Set cube size - larger and more visible
                cube = UsdGeom.Cube(marker_prim)
                cube.GetSizeAttr().Set(self.marker_size * 2)  # Cube size is edge length
            
            # Set initial position (above robot base, will be updated to end-effector later)
            marker_pos = pos.copy()
            marker_pos[2] += self.height_offset  # Raise marker above base
            
            geom_prim = UsdGeom.Xformable(marker_prim)
            geom_prim.ClearXformOpOrder()
            translate_op = geom_prim.AddXformOp(UsdGeom.XformOp.TypeTranslate)
            translate_op.Set(Gf.Vec3d(marker_pos[0], marker_pos[1], marker_pos[2]))
            
            # Set color using PrimvarsAPI with enhanced brightness
            primvar_api = UsdGeom.PrimvarsAPI(marker_prim)
            color_primvar = primvar_api.CreatePrimvar("displayColor", 
                                                    Sdf.ValueTypeNames.Color3f,
                                                    UsdGeom.Tokens.constant)
            # Make colors brighter and more saturated
            bright_color = (color * 1.5).clip(0, 1)  # Brighten but keep in valid range
            color_primvar.Set([tuple(bright_color)])
            
            self.markers.append(marker_prim)
    
    def set_robots(self, robots):
        """Set robot instances to track their end-effector positions"""
        self.robots = robots
    
    def update_marker_positions(self):
        """Update marker positions to follow robot end-effectors"""
        if self.robots is None:
            return
            
        for i, robot in enumerate(self.robots):
            if i < len(self.markers):
                try:
                    # Get robot's current end-effector position
                    robot_js = robot.get_sim_joint_state()
                    robot_cu_js = robot.get_curobo_joint_state(robot_js)
                    kinematics = robot.solver.compute_kinematics(robot_cu_js)
                    ee_pos = kinematics.ee_pos_seq.squeeze().cpu().numpy()
                    
                    # Transform to world coordinates
                    ee_world_pos = ee_pos + robot.p_R
                    
                    # Position marker above end-effector
                    marker_pos = ee_world_pos.copy()
                    marker_pos[2] += self.height_offset  # Raise above end-effector
                    
                    # Update marker position
                    marker_prim = self.markers[i]
                    geom_prim = UsdGeom.Xformable(marker_prim)
                    translate_ops = geom_prim.GetOrderedXformOps()
                    if translate_ops:
                        translate_ops[0].Set(Gf.Vec3d(marker_pos[0], marker_pos[1], marker_pos[2]))
                        
                except Exception as e:
                    # If there's an error getting end-effector position, keep marker at robot base
                    pass
    
    def update_marker_color(self, robot_id: int, new_color: np.ndarray):
        """Update the color of a specific robot marker"""
        if 0 <= robot_id < len(self.markers):
            marker_prim = self.markers[robot_id]
            primvar_api = UsdGeom.PrimvarsAPI(marker_prim)
            color_primvar = primvar_api.GetPrimvar("displayColor")
            if color_primvar:
                # Brighten the new color too
                bright_color = (new_color * 1.5).clip(0, 1)
                color_primvar.Set([tuple(bright_color)])


class CentralObstacleManager:
    """Manages central bin/table obstacles"""
    
    def __init__(self, obstacle_config: Dict):
        """
        Initialize central obstacle manager
        
        Args:
            obstacle_config: Configuration dictionary for the central obstacle
        """
        self.config = obstacle_config
    
    @staticmethod
    def create_bin_config(center_pos: List[float] = [0.0, 0.0, 0.0],
                         bin_size: List[float] = [0.4, 0.4, 0.3],
                         wall_thickness: float = 0.02) -> List[Dict]:
        """
        Create configuration for a central bin with walls
        
        Args:
            center_pos: Center position of the bin
            bin_size: [width, depth, height] of the bin
            wall_thickness: Thickness of bin walls
            
        Returns:
            List of obstacle configurations for bin walls and floor
        """
        x, y, z = center_pos
        w, d, h = bin_size
        
        obstacles = [
            # Bin floor
            {
                "name": "bin_floor",
                "curobo_type": "cuboid",
                "pose": [x, y, z, 1, 0, 0, 0],
                "dims": [w, d, wall_thickness],
                "color": [0.6, 0.4, 0.2],
                "mass": 5000,
                "linear_velocity": [0, 0, 0],
                "angular_velocity": [0, 0, 0],
                "gravity_enabled": False,
                "sim_collision_enabled": False,
                "visual_material": None
            },
            # Wall 1 (front)
            {
                "name": "bin_wall_front",
                "curobo_type": "cuboid", 
                "pose": [x + w/2, y, z + h/2, 1, 0, 0, 0],
                "dims": [wall_thickness, d, h],
                "color": [0.6, 0.4, 0.2],
                "mass": 5000,
                "linear_velocity": [0, 0, 0],
                "angular_velocity": [0, 0, 0],
                "gravity_enabled": False,
                "sim_collision_enabled": False,
                "visual_material": None
            },
            # Wall 2 (back)
            {
                "name": "bin_wall_back",
                "curobo_type": "cuboid",
                "pose": [x - w/2, y, z + h/2, 1, 0, 0, 0],
                "dims": [wall_thickness, d, h],
                "color": [0.6, 0.4, 0.2],
                "mass": 5000,
                "linear_velocity": [0, 0, 0],
                "angular_velocity": [0, 0, 0],
                "gravity_enabled": False,
                "sim_collision_enabled": False,
                "visual_material": None
            },
            # Wall 3 (left)
            {
                "name": "bin_wall_left",
                "curobo_type": "cuboid",
                "pose": [x, y - d/2, z + h/2, 1, 0, 0, 0],
                "dims": [w, wall_thickness, h],
                "color": [0.6, 0.4, 0.2],
                "mass": 5000,
                "linear_velocity": [0, 0, 0],
                "angular_velocity": [0, 0, 0],
                "gravity_enabled": False,
                "sim_collision_enabled": False,
                "visual_material": None
            },
            # Wall 4 (right)
            {
                "name": "bin_wall_right",
                "curobo_type": "cuboid",
                "pose": [x, y + d/2, z + h/2, 1, 0, 0, 0],
                "dims": [w, wall_thickness, h],
                "color": [0.6, 0.4, 0.2],
                "mass": 5000,
                "linear_velocity": [0, 0, 0],
                "angular_velocity": [0, 0, 0],
                "gravity_enabled": False,
                "sim_collision_enabled": False,
                "visual_material": None
            }
        ]
        
        return obstacles
    
    @staticmethod
    def create_table_config(center_pos: List[float] = [0.0, 0.0, 0.0],
                           table_size: List[float] = [0.6, 0.6, 0.05]) -> List[Dict]:
        """
        Create configuration for a central table
        
        Args:
            center_pos: Center position of the table
            table_size: [width, depth, height] of the table
            
        Returns:
            List with single table obstacle configuration
        """
        x, y, z = center_pos
        w, d, h = table_size
        
        return [{
            "name": "central_table",
            "curobo_type": "cuboid",
            "pose": [x, y, z, 1, 0, 0, 0],
            "dims": [w, d, h],
            "color": [0.7, 0.7, 0.7],
            "mass": 5000,
            "linear_velocity": [0, 0, 0],
            "angular_velocity": [0, 0, 0],
            "gravity_enabled": False,
            "sim_collision_enabled": False,
            "visual_material": None
        }]


class DynamicSceneManager:
    """Main manager that coordinates all dynamic scene components"""
    
    def __init__(self, n_robots: int, robot_positions: List[np.ndarray], stage,
                 scene_type: str = "bin", **kwargs):
        """
        Initialize complete dynamic scene manager
        
        Args:
            n_robots: Number of robots
            robot_positions: List of robot base positions
            stage: USD stage for visual elements
            scene_type: "bin" or "table" for central obstacle type
            **kwargs: Additional parameters for components
        """
        self.n_robots = n_robots
        self.robot_positions = robot_positions
        self.stage = stage
        self.scene_type = scene_type
        
        # Filter kwargs for target manager (only include valid DynamicTargetManager parameters)
        target_manager_valid_params = ['center_position', 'target_area_radius', 'target_height_range', 'update_interval']
        target_manager_kwargs = {k: v for k, v in kwargs.items() if k in target_manager_valid_params}
        
        # Initialize target manager
        self.target_manager = DynamicTargetManager(n_robots, **target_manager_kwargs)
        
        # Initialize robot markers with enhanced visibility
        self.marker_manager = RobotColorMarker(
            stage, robot_positions, self.target_manager.target_colors,
            marker_type=kwargs.get('marker_type', 'sphere'),
            marker_size=kwargs.get('marker_size', 0.08),
            height_offset=kwargs.get('marker_height_offset', 0.3)
        )
        
        # Setup central obstacles configuration
        center_pos = kwargs.get('center_position', [0.0, 0.0, 0.0])
        if scene_type == "bin":
            bin_size = kwargs.get('bin_size', [0.4, 0.4, 0.3])
            wall_thickness = kwargs.get('wall_thickness', 0.02)
            self.obstacle_configs = CentralObstacleManager.create_bin_config(
                center_pos, bin_size, wall_thickness
            )
        else:  # table
            table_size = kwargs.get('table_size', [0.6, 0.6, 0.05])
            self.obstacle_configs = CentralObstacleManager.create_table_config(
                center_pos, table_size
            )
    
    def start_scene(self):
        """Start the dynamic scene with automatic target updates"""
        self.target_manager.start_automatic_updates()
        print("Dynamic scene started - targets will update automatically")
    
    def stop_scene(self):
        """Stop the dynamic scene"""
        self.target_manager.stop_automatic_updates()
        print("Dynamic scene stopped")
    
    def get_target_for_robot(self, robot_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current target position and color for a specific robot
        
        Returns:
            Tuple of (target_position, target_color)
        """
        return (self.target_manager.get_target_position(robot_id),
                self.target_manager.get_target_color(robot_id))
    
    def manual_target_update(self, num_targets: Optional[int] = None):
        """Manually trigger target update"""
        return self.target_manager.update_targets(num_targets)
    
    def add_target_update_callback(self, callback):
        """Add callback for when targets are updated"""
        self.target_manager.add_update_callback(callback)
    
    def set_robots(self, robots):
        """Set robot instances for marker tracking"""
        self.marker_manager.set_robots(robots)
    
    def update_robot_markers(self):
        """Update robot marker positions to follow end-effectors"""
        self.marker_manager.update_marker_positions() 