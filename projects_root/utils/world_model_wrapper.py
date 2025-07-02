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
from typing import List, Optional, Tuple, Union, Any, Dict

# Third Party
import numpy as np
import torch
# CuRobo
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose

# -- UsdHelper is part of the Omniverse/USD stack and may be unavailable in
# testing environments without those heavy dependencies.  We therefore attempt
# the import and fall back to a very small stub so the module can still be
# imported when running lightweight unit-tests.
try:
    from curobo.util.usd_helper import UsdHelper  # type: ignore
except Exception:  # pragma: no cover
    class UsdHelper:  # type: ignore
        """Stub replacement used when usd-core/pxr is not installed."""

        def __getattr__(self, _name: str) -> Any:  # noqa: D401,E501
            raise ImportError(
                "UsdHelper unavailable – install usd-core or run inside Isaac Sim."
            )

# Project utilities
from projects_root.projects.dynamic_obs.dynamic_obs_predictor.frame_utils import FrameUtils

# Logger (kept separate so it still works even if UsdHelper stubbed)
from curobo.util.logger import log_info, log_warn, log_error


class WorldModelWrapper:
    """
    *Efficient*  wrapper for curobo world models that avoids recreation of collision world.
    
    This wrapper initializes the world model once using obstacles from stage,
    then provides efficient updates to individual obstacle poses without
    recreating the entire collision world.
    """
    
    def __init__(
        self, 
        world_config: WorldConfig, 
        X_robot_W: np.ndarray,
        robot_prim_path_stage: str,
        X_world: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        world_prim_path_stage: str = "/World", 
        verbosity: int = 2,
        pose_change_threshold: float = 1e-6,
    ):
        """
        Initialize the world model wrapper.
        
        Args:
            world_config: The associated Robot's Initial (collision) world configuration with potentially already existed obstacles (world collision model configuration). Curobo's object, has nothing to do with the simualtor (isaac sim).
            X_robot_W: Base frame pose [x, y, z, qw, qx, qy, qz] of the robot which is associated with this collision model, expressed in world
            robot_prim_path_stage: str: the prim path of the prim that is associated with the robot. You should normally not change it (unless you know what you are doing).
            X_world: World base frame pose [x, y, z, qw, qx, qy, qz]. Should be normally position = 0,0,0 (xyz) and orientation (quat)= 1,0,0,0 (wxyz). Change it only in the rare case when the world frame is not at the origin.
            world_prim_path_stage: str = "/World" - the prim path of the prim that is associated with the world frame. You should normally not change it (unless you know what you are doing).
            verbosity: Verbosity level for logging
            pose_change_threshold: Threshold (in meters) for considering pose changes, before the obstacle is considered moved and the collision model is updated.
        """
        self.world_config = world_config
        self.base_frame = np.array(X_robot_W) 
        self.world_base_frame = np.array(X_world) 
        self.robot_prim_path_stage = robot_prim_path_stage
        self.world_prim_path_stage = world_prim_path_stage
        self.tensor_args = TensorDeviceType()
        self.verbosity = int(verbosity)
        self.pose_change_threshold = float(pose_change_threshold)
        
        # Will be set after first initialization
        self.collision_world = None
        self.collision_checker = None
        self.obstacle_names = set()
        # Track last known world-frame pose of each obstacle for change detection
        self._last_world_poses: Dict[str, np.ndarray] = {}
        self._initialized = False
        # internal step counter for periodic summaries
        self._step_counter = 0
        
        if self.verbosity:
            log_info("WorldModelWrapper initialized with base frame: {}".format(self.base_frame))
    
    def initialize_from_stage(
        self,
        usd_helper: UsdHelper,
        only_paths: List[str] = ["/World"],
        ignore_substring: Optional[List[str]] = None
    ) -> WorldConfig:
        """
        Initialize the collision world from current USD stage obstacles.
        
        This should be called once to create the initial collision world.
        After this, use update() to efficiently update obstacle poses.
        
        Args:
            usd_helper: USD helper instance for reading stage obstacles
            only_paths: List of paths to search for obstacles
            ignore_substring: List of substrings to ignore in obstacle names
            
        Returns:
            WorldConfig: The initialized collision world configuration
        """
        if ignore_substring is None:
            ignore_substring = []
            
        # Get obstacles from stage and create collision world
        stage_obstacles = usd_helper.get_obstacles_from_stage(
            only_paths=only_paths,
            reference_prim_path=self.robot_prim_path_stage,
            ignore_substring=ignore_substring
        )
        # debug:
        # print("stage_obstacles.objects:")
        # for obstacle in stage_obstacles.objects:
        #     print(obstacle.name)
        # print("--------------------------------")
            
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

        # Initialize last pose cache
        if stage_obstacles.objects:
            for _obs in stage_obstacles.objects:
                self._last_world_poses[_obs.name] = np.array(_obs.pose)

        if self.verbosity >= 1:
            self._print_world_summary(header=True, list_names=(self.verbosity >= 2))

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
        ignore_substring: Optional[List[str]] = None,
        new_base_frame: Optional[np.ndarray] = None,
        
    ):
        """
        Efficiently update obstacle poses without recreating the collision world.
        
        This method reads current obstacle poses from the stage (except the ones that are in the ignore_substring list) and updates only
        the poses in the existing collision checker, AVOIDING EXPENSIVE RECREATION (that's the key change comapred to the original curobo's examples, like in mpc_example.py or motion_gen_reacher.py where they recreate the collision world every time the robot moves).
        
        Args:
            usd_helper: USD helper instance for reading current obstacle poses
            only_paths: List of paths to search for obstacles
            ignore_substring: List of substrings to ignore in obstacle names. 
            Should contain all the prim paths that you dont want to update their poses. 
            That's ideal for the case when you have a static objects, which you dont want to update their poses.
            For example, if you have a ground/plane/table, add their prim paths to this list.
            new_base_frame: Set to a np.array (x,y,z,qw,qx,qy,qz) only when using movable base robot, and only when you want to upadate the world model base frame (to bas as the new base frame of the robot, instead of the old base frame).
            In this case, the base frame of the robot (the robot which is associated with this collision model),
            is not fixed and can change. In this case, the base frame of the robot is updated to the new pose and the collision model is updated accordingly. 
            For example, if you have a movable base robot, set this to true. But if you have fixed base robot like robotic arm, set this to false.
        """
        if not self._initialized:
            log_error("WorldModelWrapper not initialized. Call initialize_from_stage() first.")
            return
            
        if self.collision_checker is None:
            log_error("Collision checker not set. Call set_collision_checker() first.")
            return
        
        if new_base_frame is not None:
            self._update_base_frame(new_base_frame)

        if ignore_substring is None:
            ignore_substring = []
            
        # Get current obstacles from stage
        current_obstacles = usd_helper.get_obstacles_from_stage(
            only_paths=only_paths,
            reference_prim_path=self.world_prim_path_stage,
            ignore_substring=ignore_substring
        )
        
        # Update poses for each obstacle
        if current_obstacles.objects:
            for obstacle in current_obstacles.objects:
                obstacle_name = obstacle.name
                obs_type = obstacle.__class__.__name__ if hasattr(obstacle, "__class__") else "Unknown"
                
                if obstacle_name in self.obstacle_names:
                    # Current obstacle world pose
                    X_obs_W = np.array(obstacle.pose)  # [x, y, z, qw, qx, qy, qz]

                    # Check if pose actually changed since last update
                    last_pose = self._last_world_poses.get(obstacle_name)
                    if last_pose is not None and np.allclose(
                        last_pose, X_obs_W, atol=self.pose_change_threshold
                    ):
                        # Skip unchanged obstacle to avoid spurious "MOVED" logs
                        continue
                    # Update cache
                    self._last_world_poses[obstacle_name] = X_obs_W.copy()
                    
                    # Transform obstacle pose from world frame to base frame
                    base_frame_pose = self._transform_pose_world_to_base(X_obs_W)
                    
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

                        if self.verbosity >= 1:
                            self._vprint(
                                f"{obstacle_name} MOVED (type: {obs_type})\n"
                                f"  X_W (pose w.r to world frame): {self._pose_str(X_obs_W)}\n"
                                f"  X_R (pose w.r to collision world frame): {self._pose_str(base_frame_pose)}"
                            )
                    except Exception as e:
                        log_warn(f"Failed to update obstacle {obstacle_name}: {e}")
                else:
                    # Not yet part of collision world – will be added by add_new_obstacles_from_stage()
                    if self.verbosity >= 1:
                        self._vprint(
                            f"New obstacle detected (will be added): {obstacle_name} (type: {obs_type})"
                        )

        # Optionally try to detect and add new obstacles that might have been introduced
        self.add_new_obstacles_from_stage(
            usd_helper,
            only_paths=only_paths,
            reference_prim_path=self.world_prim_path_stage,
            ignore_substring=ignore_substring,
            silent=True,
        )
    
    def _update_base_frame(self, new_base_frame: np.ndarray):
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
        Transform pose from world frame to the world model (the robot's) base frame.
        
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

    # ------------------------------------------------------------------
    # New functionality
    # ------------------------------------------------------------------
    def add_new_obstacles_from_stage(
        self,
        usd_helper: UsdHelper,
        reference_prim_path: str,
        only_paths: List[str] = ["/World"],
        ignore_substring: Optional[List[str]] = None,
        silent: bool = False,
    ) -> None:
        """Detect and add *new* obstacles that have appeared in the stage after the
        initialisation. This avoids rebuilding the whole collision world – the
        new obstacles are appended to the existing world model and then the
        collision checker is re-loaded so it can account for them.

        Args:
            usd_helper: UsdHelper instance to query stage.
            only_paths: Stage paths to search.
            ignore_substring: List of substrings to ignore.
            silent: If True, suppresses log output when no new obstacles found.
        """

        if not self._initialized:
            log_error("WorldModelWrapper not initialised. Cannot add new obstacles.")
            return

        if ignore_substring is None:
            ignore_substring = []

        # Query current obstacles from stage
        stage_obstacles = usd_helper.get_obstacles_from_stage(
            only_paths=only_paths,
            reference_prim_path=reference_prim_path,
            ignore_substring=ignore_substring,
        )

        newly_added: List[str] = []
        if stage_obstacles.objects:
            for obs in stage_obstacles.objects:
                if obs.name not in self.obstacle_names:
                    try:
                        # 1)  append to internal collision_world
                        if self.collision_world is not None and hasattr(self.collision_world, "add_obstacle"):
                            self.collision_world.add_obstacle(obs)
                        # 2) keep track locally
                        self.obstacle_names.add(obs.name)
                        newly_added.append(f"{obs.name} (type: {obs.__class__.__name__})")
                        # Cache its pose for future change detection
                        self._last_world_poses[obs.name] = np.array(obs.pose)

                        # Print pose info for each newly added obstacle
                        if self.verbosity >= 1:
                            w_pose = np.array(obs.pose)
                            b_pose = self._transform_pose_world_to_base(w_pose)
                            self._vprint(
                                f"ADDED {obs.name} (type: {obs.__class__.__name__})\n"
                                f"  X_W (pose w.r to world frame): : {self._pose_str(w_pose)}\n"
                                f"  X_R (pose w.r to collision world frame): {self._pose_str(b_pose)}"
                            )
                    except Exception as e:
                        log_warn(f"Failed adding new obstacle {obs.name}: {e}")

        # Reload collision checker if anything new was added so it can pick the changes.
        if newly_added and self.collision_checker is not None:
            try:
                self.collision_checker.load_collision_model(self.collision_world)
            except Exception as e:
                log_warn(f"Could not reload collision checker after adding new obstacles: {e}")

        if newly_added and self.verbosity >= 1:
            self._vprint("Added new obstacle(s): " + ", ".join(newly_added))

        # periodic summary depending on verbosity level
        if self.verbosity >= 2:
            self._step_counter += 1
            interval = 10 if self.verbosity < 4 else 1
            if self._step_counter % interval == 0:
                list_names = self.verbosity >= 3
                self._print_world_summary(header=False, list_names=list_names)

    # --------------------------------------------------------------
    # Helper for verbose summaries
    # --------------------------------------------------------------
    def _print_world_summary(self, header: bool = True, list_names: bool = False):
        """Prints number of obstacles grouped by type (and optionally names)."""
        if self.verbosity == 0:
            return

        type_counts = {}
        name_by_type = {}
        for name in self.obstacle_names:
            # Derive type via name lookup inside collision_world
            obj_type = "Unknown"
            if self.collision_world:
                for attr in ["cuboid", "mesh", "sphere", "capsule", "cylinder"]:
                    objs = getattr(self.collision_world, attr, [])
                    for o in objs or []:
                        if getattr(o, "name", None) == name:
                            obj_type = o.__class__.__name__
                            break
                    if obj_type != "Unknown":
                        break
            type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
            name_by_type.setdefault(obj_type, []).append(name)

        if header:
            self._vprint("===== Collision-world Summary =====")
        summary = ", ".join([f"{t}: {c}" for t, c in type_counts.items()])
        self._vprint("Obstacle counts → " + summary)
        if list_names:
            for t, names in name_by_type.items():
                self._vprint(f"  {t}: {', '.join(sorted(names))}")
        if header:
            self._vprint("====================================")

    # --------------------------------------------------------------
    # Internal helper for printing respecting verbosity
    # --------------------------------------------------------------
    def _vprint(self, msg: str):
        if self.verbosity > 0:
            print(msg)

    # helper to format pose arrays nicely
    @staticmethod
    def _pose_str(arr: np.ndarray) -> str:
        return np.round(arr.astype(float), 2).tolist().__str__()
