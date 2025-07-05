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
import time
# CuRobo
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.geom.sphere_fit import SphereFitType


# Project utilities
from projects_root.projects.dynamic_obs.dynamic_obs_predictor.frame_utils import FrameUtils

from curobo.util.logger import log_info, log_warn, log_error


class WorldModelWrapper:
    """
    *Efficient*  wrapper for curobo world models that avoids recreation of collision world.
    
    This wrapper initializes the world model once using obstacles from real world/simulator,
    then provides efficient updates to individual obstacle poses without
    recreating the entire collision world.
    """
    
    def __init__(
        self, 
        world_config: WorldConfig, 
        X_robot_W: np.ndarray,
        X_world: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        verbosity: int = 2,
        pose_change_threshold: float = 1e-6,
    ):
        """
        Initialize the world model wrapper.
        
        Args:
            world_config: The associated Robot's Initial (collision) world configuration with potentially already existed obstacles (world collision model configuration). Curobo's object, has nothing to do with the simualtor (isaac sim).
            X_robot_W: Base frame pose [x, y, z, qw, qx, qy, qz] of the robot which is associated with this collision model, expressed in world
            X_world: World base frame pose [x, y, z, qw, qx, qy, qz]. Should be normally position = 0,0,0 (xyz) and orientation (quat)= 1,0,0,0 (wxyz). Change it only in the rare case when the world frame is not at the origin.
            verbosity: Verbosity level for logging
            pose_change_threshold: Threshold (in meters) for considering pose changes, before the obstacle is considered moved and the collision model is updated.
        """
        self.world_config = world_config
        self.base_frame = np.array(X_robot_W) 
        self.world_base_frame = np.array(X_world) 
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
    
    def initialize_from_cu_world(
        self,
        cu_world_R: WorldConfig,

    ) -> WorldConfig:
        """
        Initialize the collision world from current sim/real world obstacles.
        
        This should be called once to create the initial collision world.
        After this, use update() to efficiently update obstacle poses.
        
        Args:
            cu_world_R: WorldConfig object that contains the current obstacle poses fromt he environment (real world/simulator), with poses expressed in robot frame.
        Returns:
            WorldConfig: The initialized collision world configuration
        """
            

        collision_world = cu_world_R.get_collision_check_world() 
        

        # TODO:
        # CHECK THIS:
        # what it probably does: getting the old obstacles from the initial world config
        # and adding them to the (newer?) collision world
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
        if cu_world_R.objects:
            for _obs in cu_world_R.objects:
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
        print("debug setting collision checker!!!!!!!!")
        self.collision_checker = collision_checker

        # ----------------------------------------------------------
        # Critical: ensure both *references* point to the *same* object
        # ----------------------------------------------------------
        # Most CuRobo collision-checker implementations keep their own
        # `world_model` copy that **is** modified by
        # `update_obstacle_pose(..., update_cpu_reference=True)`.
        # The visualiser relies on `get_collision_world()` therefore we
        # re-bind our `self.collision_world` to that very instance so every
        # change becomes visible without any extra copying or syncing.
        try:
            print("debug!!! collision_checker attributes:")
            # print(dir(self.collision_checker))
            # print(id(self.collision_checker))
            if getattr(self.collision_checker, "world_model", None) is not None:
                assert id(self.collision_checker.world_model)==id(self.collision_world), "Collision checker world model and self.collision_world must be the same object"
                self.collision_world = self.collision_checker.world_model
                
        except Exception:
            # Fallback – leave previous reference in place
            raise Exception("Failed to set collision checker reference")

        log_info("Collision checker set in WorldModelWrapper (world_model linked)")
    
    def update(
        self,
        cu_world_W: WorldConfig,
        new_base_frame: Optional[np.ndarray] = None,
        
    ):
        """
        
        THE DESIRED STATE: (NOT YET THE CASE):
        Given a curobo model of the world (could be built from real world/ or from simulator (using the usd_helper tool)), 
        update the obstacle poses in the collision world.         
        This method reads current obstacle poses from the real world/simulator (except the ones that are in the ignore_substring list) and updates only
        the poses in the existing collision checker, AVOIDING EXPENSIVE RECREATION (that's the key change comapred to the original curobo's examples, like in mpc_example.py or motion_gen_reacher.py where they recreate the collision world every time the robot moves).
        
        TODO: COULD BE MORE EFFICIENT JUST TO PASS THE NEW POSES (GET THE DESIRED STATE).
        IN THE DESIRED STATE WE AER NOT PASSING a WorldConfig but just the new poses.


        Args:
            cu_world_W: WorldConfig object that contains the current obstacle poses fromt he world (also real world)/simulator, expressed in world frame.
            new_base_frame: Set to a np.array (x,y,z,qw,qx,qy,qz) only when using movable base robot, and only when you want to upadate the world model base frame (to bas as the new base frame of the robot, instead of the old base frame).
            In this case, the base frame of the robot (the robot which is associated with this collision model),
            is not fixed and can change. In this case, the base frame of the robot is updated to the new pose and the collision model is updated accordingly. 
            For example, if you have a movable base robot, set this to true. But if you have fixed base robot like robotic arm, set this to false.
        """
        if not self._initialized:
            log_error("WorldModelWrapper not initialized. Call initialize_from_cu_world() first.")
            return
            
        if self.collision_checker is None:
            log_error("Collision checker not set. Call set_collision_checker() first.")
            return
        
        if new_base_frame is not None:
            self._update_base_frame(new_base_frame)

        # Update poses for each obstacle
        if cu_world_W.objects:
            for obstacle in cu_world_W.objects:
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
                            env_idx=0,
                            update_cpu_reference=True,
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
                    # Not yet part of collision world – will be added by add_new_obstacles_from_cu_world()
                    if self.verbosity >= 1:
                        self._vprint(
                            f"New obstacle detected (will be added): {obstacle_name} (type: {obs_type})"
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
    def add_new_obstacles_from_cu_world(
        self,
        cu_world_R: WorldConfig,
        silent: bool = False,
    ) -> None:
        """ After getting the obstacles from the real world/simulator, add them to the collision world.
        More efficient than the original curobo's examples, like in mpc_example.py or motion_gen_reacher.py where they 
        recreate the collision world every time every time instead of updating the existing collision world (at least in the examples).


        Args:
            cu_world_R: WorldConfig object that contains the current obstacle poses in the environment (real world/simulator), expressed in robot frame.
            silent: If True, suppresses log output when no new obstacles found.
        """

        if not self._initialized:
            log_error("WorldModelWrapper not initialised. Cannot add new obstacles.")
            return


        newly_added: List[str] = []
        if cu_world_R.objects:
            for obs in cu_world_R.objects:
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

    # ------------------------------------------------------------------
    # Public API – lightweight per-frame pose update (simulator-agnostic)
    # ------------------------------------------------------------------
    def update_from_pose_dict(self, pose_dict: Dict[str, List[float]]):
        """Update obstacle poses using a dict *{name: [x,y,z,qw,qx,qy,qz]}*.

        This avoids the heavy `UsdHelper.get_obstacles_from_stage()` call and
        lets higher-level code supply just the current poses of already known
        obstacles.

        *New* obstacles are ignored here – caller may detect them separately
        and invoke `add_new_obstacles_from_cu_world()` when necessary.
        """

        if not self._initialized or self.collision_checker is None:
            return

        for name, pose_list in pose_dict.items():
            if name not in self.obstacle_names:
                # Unknown object – handled elsewhere
                continue

            pose_arr = np.asarray(pose_list, dtype=float)

            last_pose = self._last_world_poses.get(name)
            if last_pose is not None and np.allclose(
                last_pose, pose_arr, atol=self.pose_change_threshold
            ):
                continue  # unchanged

            self._last_world_poses[name] = pose_arr.copy()

            # Convert to robot base frame
            base_pose = self._transform_pose_world_to_base(pose_arr)

            cu_pose = Pose(
                position=self.tensor_args.to_device(base_pose[:3]),
                quaternion=self.tensor_args.to_device(base_pose[3:]),
            )

            try:
                self.collision_checker.update_obstacle_pose(
                    name=name,
                    w_obj_pose=cu_pose,
                    env_idx=0,
                    update_cpu_reference=True,
                )
                if self.verbosity >= 3:
                    self._vprint(f"{name} MOVED (fast-dict)")
                    self._vprint(
                                f"  X_W (pose w.r to world frame): : {self._pose_str(pose_arr)}\n"
                                f"  X_R (pose w.r to collision world frame): {self._pose_str(base_pose)}"
                            )
            except Exception as e:
                log_warn(f"update_from_pose_dict failed for {name}: {e}")

    # ------------------------------------------------------------------
    # Static helper – generate bounding-sphere approximations for visualization
    # ------------------------------------------------------------------

    @staticmethod
    def make_geom_approx_to_spheres(
        collision_world: WorldConfig,
        base_pose_W: Union[List[float], np.ndarray, Pose],
        n_spheres: int = 30,
        fit_type: SphereFitType = SphereFitType.SAMPLE_SURFACE,
        radius_scale: float = 0.01,
    ) -> List[Tuple[np.ndarray, float]]:
        """Return a list of `(pos_W, radius)` tuples approximating collision objects.

        Args:
            collision_world: The current CuRobo collision world instance.
            base_pose_W: Pose (world ⇐ robot-base) – accepts Pose or 7-element array/list.
            n_spheres: Number of spheres to sample **per obstacle**.
            fit_type: SphereFitType used by `get_bounding_spheres` (default SAMPLE_SURFACE).
            radius_scale: Sphere radius as a fraction of the smallest OBB extent.

        Returns:
            List of tuples `(np.ndarray(3,), float)` – world-frame sphere centres and radii.
        """

        if collision_world is None:
            return []

        obs_list = list(collision_world.objects or [])
        if not obs_list:
            for attr in ["cuboid", "mesh", "sphere", "capsule", "cylinder"]:
                obs_attr = getattr(collision_world, attr, None)
                if obs_attr:
                    obs_list.extend(obs_attr)

        if not obs_list:
            return []

        # Normalise base_pose_W to Pose
        if not isinstance(base_pose_W, Pose):
            if isinstance(base_pose_W, (list, tuple, np.ndarray)) and len(base_pose_W) == 7:
                base_pose_W = Pose.from_list(list(base_pose_W))
            else:
                raise ValueError("base_pose_W must be Pose or 7-element list/array")

        sphere_list: List[Tuple[np.ndarray, float]] = []

        for obs in obs_list:
            try:
                # Determine automatic radius
                obb = obs.get_cuboid()
                if obb is not None and isinstance(getattr(obb, "dims", None), (list, tuple)):
                    auto_radius = radius_scale * float(min(obb.dims))
                else:
                    auto_radius = radius_scale  # metres

                sph = obs.get_bounding_spheres(
                    n_spheres=n_spheres,
                    surface_sphere_radius=auto_radius,
                    fit_type=fit_type,
                    pre_transform_pose=base_pose_W,
                )

                for s in sph:
                    p_attr = getattr(s, "pose", None)
                    pos_w = np.array(p_attr[:3], dtype=float) if p_attr else np.zeros(3)
                    sphere_list.append((pos_w, float(s.radius)))
            except Exception:
                # Skip obstacle on failure
                continue

        return sphere_list
