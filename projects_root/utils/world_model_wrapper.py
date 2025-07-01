#!/usr/bin/env python3
#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

"""
World Model Wrapper for efficient obstacle pose updates.

This wrapper avoids the inefficient recreation of the entire collision world
every time an obstacle moves. Instead, it initializes the world model once
and provides efficient updates to individual obstacle poses.
"""

# Standard Library
from typing import List, Optional, Union

# Third Party
import numpy as np
import torch

# CuRobo
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.util.usd_helper import UsdHelper
from curobo.util.logger import log_info, log_warn, log_error

# Project utilities
from projects_root.projects.dynamic_obs.dynamic_obs_predictor.frame_utils import FrameUtils


class WorldModelWrapper:
    """
    Efficient world model wrapper that avoids recreation of collision world.
    
    This wrapper initializes the world model once using obstacles from stage,
    then provides efficient updates to individual obstacle poses without
    recreating the entire collision world.
    """
    
    def __init__(
        self, 
        world_config: WorldConfig,
        base_frame: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        world_base_frame: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    ):
        """
        Initialize the world model wrapper.
        
        Args:
            world_config: Initial world configuration with static obstacles
            base_frame: Base frame pose [x, y, z, qw, qx, qy, qz] - typically robot base frame
            world_base_frame: World base frame pose [x, y, z, qw, qx, qy, qz] - typically identity
        """
        self.world_config = world_config
        self.base_frame = np.array(base_frame)
        self.world_base_frame = np.array(world_base_frame)
        self.tensor_args = TensorDeviceType()
        
        # Will be set after first initialization
        self.collision_world = None
        self.collision_checker = None
        self.obstacle_names = set()
        self._initialized = False
        
        log_info("WorldModelWrapper initialized with base frame: {}".format(self.base_frame))
    
    def initialize_from_stage(
        self,
        usd_helper: UsdHelper,
        only_paths: List[str] = ["/World"],
        reference_prim_path: str = "/World",
        ignore_substring: Optional[List[str]] = None
    ) -> WorldConfig:
        """
        Initialize the collision world from USD stage obstacles.
        
        This should be called once to create the initial collision world.
        After this, use update() to efficiently update obstacle poses.
        
        Args:
            usd_helper: USD helper instance for reading stage obstacles
            only_paths: List of paths to search for obstacles
            reference_prim_path: Reference prim path for coordinate transforms
            ignore_substring: List of substrings to ignore in obstacle names
            
        Returns:
            WorldConfig: The initialized collision world configuration
        """
        if ignore_substring is None:
            ignore_substring = []
            
        # Get obstacles from stage and create collision world
        stage_obstacles = usd_helper.get_obstacles_from_stage(
            only_paths=only_paths,
            reference_prim_path=reference_prim_path,
            ignore_substring=ignore_substring
        )
        
        # Combine with initial world config
        collision_world = stage_obstacles.get_collision_check_world()
        
        # Add initial world config obstacles
        if self.world_config.cuboid:
            for cuboid in self.world_config.cuboid:
                collision_world.add_obstacle(cuboid)
        if self.world_config.mesh:
            for mesh in self.world_config.mesh:
                collision_world.add_obstacle(mesh)
        if self.world_config.sphere:
            for sphere in self.world_config.sphere:
                collision_world.add_obstacle(sphere)
        if self.world_config.capsule:
            for capsule in self.world_config.capsule:
                collision_world.add_obstacle(capsule)
        if self.world_config.cylinder:
            for cylinder in self.world_config.cylinder:
                collision_world.add_obstacle(cylinder)
                
        # Store the collision world and extract obstacle names
        self.collision_world = collision_world
        self._extract_obstacle_names()
        self._initialized = True
        
        log_info(f"WorldModelWrapper initialized with {len(self.obstacle_names)} obstacles")
        return collision_world
    
    def set_collision_checker(self, collision_checker):
        """
        Set the collision checker reference.
        
        This should be called after the collision checker is created from the world config.
        
        Args:
            collision_checker: The collision checker instance (e.g., WorldMeshCollision)
        """
        self.collision_checker = collision_checker
        log_info("Collision checker set in WorldModelWrapper")
    
    def update(
        self,
        usd_helper: UsdHelper,
        only_paths: List[str] = ["/World"],
        reference_prim_path: str = "/World", 
        ignore_substring: Optional[List[str]] = None
    ):
        """
        Efficiently update obstacle poses without recreating the collision world.
        
        This method reads current obstacle poses from the stage and updates only
        the poses in the existing collision checker, avoiding expensive recreation.
        
        Args:
            usd_helper: USD helper instance for reading current obstacle poses
            only_paths: List of paths to search for obstacles
            reference_prim_path: Reference prim path for coordinate transforms
            ignore_substring: List of substrings to ignore in obstacle names
        """
        if not self._initialized:
            log_error("WorldModelWrapper not initialized. Call initialize_from_stage() first.")
            return
            
        if self.collision_checker is None:
            log_error("Collision checker not set. Call set_collision_checker() first.")
            return
            
        if ignore_substring is None:
            ignore_substring = []
            
        # Get current obstacles from stage
        current_obstacles = usd_helper.get_obstacles_from_stage(
            only_paths=only_paths,
            reference_prim_path=reference_prim_path,
            ignore_substring=ignore_substring
        )
        
        # Update poses for each obstacle
        if current_obstacles.objects:
            for obstacle in current_obstacles.objects:
                obstacle_name = obstacle.name
                
                if obstacle_name in self.obstacle_names:
                    # Transform obstacle pose from world frame to base frame
                    world_pose = np.array(obstacle.pose)  # [x, y, z, qw, qx, qy, qz]
                    base_frame_pose = self._transform_pose_world_to_base(world_pose)
                    
                    # Create CuRobo Pose object
                    curobo_pose = Pose(
                        position=self.tensor_args.to_device(base_frame_pose[:3]),
                        quaternion=self.tensor_args.to_device(base_frame_pose[3:])
                    )
                    
                    # Update pose in collision checker
                    try:
                        self.collision_checker.update_obstacle_pose(
                            name=obstacle_name,
                            w_obj_pose=curobo_pose,
                            env_idx=0
                        )
                    except Exception as e:
                        log_warn(f"Failed to update obstacle {obstacle_name}: {e}")
                else:
                    log_warn(f"Obstacle {obstacle_name} not found in initialized world model")
    
    def update_base_frame(self, new_base_frame: np.ndarray):
        """
        Update the base frame pose.
        
        Args:
            new_base_frame: New base frame pose [x, y, z, qw, qx, qy, qz]
        """
        self.base_frame = np.array(new_base_frame)
        log_info(f"Base frame updated to: {self.base_frame}")
    
    def get_collision_world(self) -> Optional[WorldConfig]:
        """
        Get the collision world configuration.
        
        Returns:
            WorldConfig: The collision world or None if not initialized
        """
        return self.collision_world
    
    def get_obstacle_names(self) -> set:
        """
        Get the set of obstacle names in the world model.
        
        Returns:
            set: Set of obstacle names
        """
        return self.obstacle_names.copy()
    
    def is_initialized(self) -> bool:
        """
        Check if the wrapper is initialized.
        
        Returns:
            bool: True if initialized, False otherwise
        """
        return self._initialized
    
    def _extract_obstacle_names(self):
        """Extract obstacle names from the collision world."""
        self.obstacle_names = set()
        
        if self.collision_world:
            if hasattr(self.collision_world, 'objects') and self.collision_world.objects:
                for obj in self.collision_world.objects:
                    if hasattr(obj, 'name'):
                        self.obstacle_names.add(obj.name)
            
            # Also check specific obstacle types
            for attr_name in ['cuboid', 'mesh', 'sphere', 'capsule', 'cylinder']:
                if hasattr(self.collision_world, attr_name):
                    obstacles = getattr(self.collision_world, attr_name)
                    if obstacles:
                        for obj in obstacles:
                            if hasattr(obj, 'name'):
                                self.obstacle_names.add(obj.name)
    
    def _transform_pose_world_to_base(self, world_pose: np.ndarray) -> np.ndarray:
        """
        Transform pose from world frame to base frame.
        
        Args:
            world_pose: Pose in world frame [x, y, z, qw, qx, qy, qz]
            
        Returns:
            np.ndarray: Pose in base frame [x, y, z, qw, qx, qy, qz]
        """
        # Extract position and orientation
        world_position = world_pose[:3]
        world_orientation = world_pose[3:]  # [qw, qx, qy, qz]
        
        # Transform from world frame to base frame using FrameUtils
        base_position, base_orientation = FrameUtils.world_to_F(
            self.base_frame[:3],    # base frame position in world
            self.base_frame[3:],    # base frame orientation in world  
            world_position,         # obstacle position in world
            world_orientation       # obstacle orientation in world
        )
        
        return np.concatenate([base_position, base_orientation]) 