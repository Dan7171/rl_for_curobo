"""
Isaac Sim launcher with flexible action graph for any robot configuration.
This script starts Isaac Sim with ROS2 integration for MoveIt2 planning.
"""

import sys
import os
import argparse
import time
import numpy as np
from pathlib import Path

# Add the current directory to the path so we can import config_loader
sys.path.append(str(Path(__file__).parent))
from config_loader import ConfigLoader, RobotConfig

# Isaac Sim imports
try:
    from isaacsim import SimulationApp
except ImportError:
    try:
        from omni.isaac.kit import SimulationApp
    except ImportError:
        print("Error: Isaac Sim not found. Please make sure Isaac Sim is installed and in the Python path.")
        sys.exit(1)

# Isaac Sim configuration
CONFIG = {"renderer": "RayTracedLighting", "headless": False}

class IsaacSimLauncher:
    """
    Flexible Isaac Sim launcher that can work with any robot configuration.
    """
    
    def __init__(self, robot_config: RobotConfig):
        """
        Initialize the Isaac Sim launcher.
        
        Args:
            robot_config: Robot configuration object
        """
        self.robot_config = robot_config
        self.simulation_app = None
        self.simulation_context = None
        
        # Isaac Sim version detection
        self.isaac_sim_ge_4_5_version = True
        try:
            from isaacsim.core.version import get_version
        except ImportError:
            try:
                from omni.isaac.version import get_version
                self.isaac_sim_ge_4_5_version = False
            except ImportError:
                print("Warning: Could not determine Isaac Sim version")
                
        # Check if legacy Isaac Sim (2023.1.1 or older)
        try:
            self.is_legacy_isaacsim = len(get_version()[2]) == 4
        except:
            self.is_legacy_isaacsim = False
            
        # Graph and prim paths
        self.graph_path = "/ActionGraph"
        self.background_stage_path = "/background"
        self.background_usd_path = "/Isaac/Environments/Simple_Room/simple_room.usd"
        
        # Target visualization storage
        self.target_prims = []
        
    def start_simulation_app(self):
        """Start the Isaac Sim simulation app."""
        print("Starting Isaac Sim...")
        self.simulation_app = SimulationApp(CONFIG)
        
        # Import modules that need to be loaded after app creation
        self._import_isaac_modules()
        
        # Enable ROS2 bridge
        self._enable_ros2_bridge()
        
        # Create simulation context
        self.simulation_context = self.SimulationContext(stage_units_in_meters=1.0)
        
        print("Isaac Sim started successfully")
        
    def _import_isaac_modules(self):
        """Import Isaac Sim modules after app creation."""
        # Import based on version
        if self.isaac_sim_ge_4_5_version:
            from isaacsim.core.api import SimulationContext
            from isaacsim.core.utils.prims import set_targets
            from isaacsim.core.utils import extensions, prims, rotations, stage, viewports
            from isaacsim.storage.native import nucleus
            # "omni.graph.window.action" = { version = "1.40.0" }
        else:
            from isaacsim.core import SimulationContext
            from isaacsim.core.utils.prims import set_targets
            from isaacsim.core.utils import extensions, prims, rotations, stage, viewports
            from isaacsim.core.utils import nucleus
            
        # Store modules as instance variables
        self.SimulationContext = SimulationContext
        self.set_targets = set_targets
        self.extensions = extensions
        self.prims = prims
        self.rotations = rotations
        self.stage = stage
        self.viewports = viewports
        self.nucleus = nucleus
        
        # Additional imports
        from pxr import Gf, UsdGeom
        import omni.graph.core as og
        import omni
        
        self.Gf = Gf
        self.UsdGeom = UsdGeom
        self.og = og
        self.omni = omni
        
    def _enable_ros2_bridge(self):
        """Enable ROS2 bridge extension."""
        if self.isaac_sim_ge_4_5_version:
            self.extensions.enable_extension("isaacsim.ros2.bridge")
        else:
            self.extensions.enable_extension("omni.isaac.ros2_bridge")
            
    def setup_scene(self):
        """Set up the Isaac Sim scene with robot and environment."""
        print("Setting up scene...")
        
        # Get assets root path
        assets_root_path = self.nucleus.get_assets_root_path()
        if assets_root_path is None:
            print("Error: Could not find Isaac Sim assets folder")
            return False
            
        # Set camera view
        self.viewports.set_camera_view(
            eye=np.array([1.2, 1.2, 0.8]), 
            target=np.array([0, 0, 0.5])
        )
        
        # Load environment
        self.stage.add_reference_to_stage(
            assets_root_path + self.background_usd_path, 
            self.background_stage_path
        )
        
        # Load robot
        self._load_robot(assets_root_path)
        
        # Create target visualizations
        self._create_target_visualizations()
        
        print("Scene setup complete")
        return True
        
    def _load_robot(self, assets_root_path: str):
        """Load the robot into the scene."""
        robot_stage_path = self.robot_config.get_isaac_stage_path()
        robot_usd_path = self.robot_config.get_isaac_usd_path()
        robot_position = self.robot_config.get_robot_position()
        robot_orientation = self.robot_config.get_robot_orientation()

        # If no USD is provided in the config, try to import the URDF on-the-fly
        if (not robot_usd_path) or robot_usd_path.strip() == "":
            urdf_path = self.robot_config.get_urdf_path()
            if not urdf_path:
                raise RuntimeError("Neither 'usd_path' nor 'urdf_path' supplied in robot configuration")

            print(f"No USD specified – importing URDF into stage: {urdf_path}")
            robot_usd_path = self._import_urdf(urdf_path, robot_stage_path)
            
            # After import we keep the in-stage prim so downstream code uses it

        # Convert orientation to Gf.Rotation if needed
        if len(robot_orientation) == 4 and robot_orientation[3] != 0:
            # [x, y, z, angle_degrees] format
            rotation = self.rotations.gf_rotation_to_np_array(
                self.Gf.Rotation(
                    self.Gf.Vec3d(robot_orientation[0], robot_orientation[1], robot_orientation[2]), 
                    robot_orientation[3]
                )
            )
        else:
            # Default orientation
            rotation = self.rotations.gf_rotation_to_np_array(
                self.Gf.Rotation(self.Gf.Vec3d(0, 0, 1), 0)
            )
           
        # Create robot prim only if it does **not** already exist (the omni.kit
        # URDF importer spawns the articulation automatically).
        if self._prim_exists(robot_stage_path):
            print(f"Robot prim '{robot_stage_path}' already present – skipping explicit create_prim().")
        else:
            full_usd_path = assets_root_path + robot_usd_path if robot_usd_path.startswith("/") else robot_usd_path

            print(f"Loading robot from: {full_usd_path}")
            print(f"Robot stage path: {robot_stage_path}")
            print(f"Robot position: {robot_position}")

            self.prims.create_prim(
                robot_stage_path,
                "Xform",
                position=np.array(robot_position),
                orientation=rotation,
                usd_path=full_usd_path,
            )

    def _import_urdf(self, urdf_path: str, robot_stage_path: str) -> str:
        """Import a URDF file into the current stage and return the resulting USD path.

        This helper tries a few different APIs depending on Isaac Sim version.
        It first prefers the official omni.kit.commands interface added in
        Isaac Sim 4.0+, which both converts the URDF to USD *and* spawns the
        articulation in-stage.  If that fails, we fall back to the older
        direct-module approaches and finally (worst-case) just reference the
        URDF so the rest of the pipeline can continue.
        """

        # ------------------------------------------------------------------
        # Preferred path: omni.kit.commands URDF importer (Isaac Sim ≥ 4.x)
        # ------------------------------------------------------------------
        try:
            import omni.kit.commands as kit_cmds  # pylint: disable=import-error

            # Build a default ImportConfig via the command interface so we do
            # **not** depend on the internal Python package path – those move
            # around between releases.
            _, import_cfg = kit_cmds.execute("URDFCreateImportConfig")

            # Tailor a couple of obvious parameters from the YAML file if they
            # are present.  Failing that we rely on sensible defaults.
            try:
                fix_base = self.robot_config.get("fix_base", False)  # type: ignore
                import_cfg.fix_base = bool(fix_base)
            except Exception:
                pass  # optional – keep default

            # Parse & import – this will also *spawn* the articulation under
            # the returned prim path (articulation root).
            print("Importing URDF via omni.kit.commands …")
            _, spawned_path = kit_cmds.execute(
                "URDFParseAndImportFile",
                urdf_path=urdf_path,
                import_config=import_cfg,
                get_articulation_root=True,
            )

            print(f"URDF successfully imported. Articulation root: {spawned_path}")
            
            # Handle path mismatch: the importer creates a prim based on the robot name
            # in the URDF, but we might want a different path. Rename if needed.
            if spawned_path.split('/')[0] != robot_stage_path:
                print(f"Renaming imported prim from '{spawned_path}' to '{robot_stage_path}'")
                try:
                    # Use USD rename command to move the prim
                    kit_cmds.execute("MovePrim", path_from=spawned_path, path_to=robot_stage_path)
                    print(f"Successfully renamed prim to '{robot_stage_path}'")
                except Exception as rename_exc:
                    print(f"Warning: Failed to rename prim ({rename_exc}). Using original path '{spawned_path}'")
                    # Update the robot_stage_path to match what was actually created
                    robot_stage_path = spawned_path
            
            # Nothing to return here because the prim is already in the stage.
            # Down-stream code will notice the prim and skip duplicate creation.
            return ""

        except Exception as primary_exc:
            print(f"URDF import via omni.kit.commands failed ({primary_exc}). Trying legacy paths …")

        # ------------------------------------------------------------------
        # Legacy path: direct Python module (pre-4.0 releases)
        # ------------------------------------------------------------------
        try:
            from omni.importer import urdf as urdf_importer  # type: ignore

            print("Using omni.importer.urdf legacy importer …")
            usd_path = urdf_importer._urdf.import_from_file(  # pylint: disable=protected-access
                urdf_path, destination_path=robot_stage_path
            )

            print(f"URDF converted to USD at: {usd_path}")
            return usd_path

        except Exception as legacy_exc:
            # ------------------------------------------------------------------
            # Last-resort fallback – reference the URDF so the stage at least
            # contains *something* the user can debug with.
            # ------------------------------------------------------------------
            print(
                "Warning: URDF import failed via all known paths. "
                f"Falling back to simple reference. (Errors: {primary_exc}; {legacy_exc})"
            )

            try:
                self.stage.add_reference_to_stage(urdf_path, robot_stage_path)
            except Exception as e2:
                print(f"  Failed to add reference: {e2}")

            return urdf_path

    # ----------------------------------------------------------------------

    def _prim_exists(self, prim_path: str) -> bool:
        """Utility: check whether a prim already exists in the current stage."""
        return self.stage.get_current_stage().GetPrimAtPath(prim_path).IsValid()
            
    def _create_target_visualizations(self):
        """Create visual targets for each end effector."""
        end_effectors = self.robot_config.get_end_effectors()
        target_size = self.robot_config.get_target_visualization_size()
        
        for i, ee_config in enumerate(end_effectors):
            target_name = f"target_{ee_config['name']}"
            target_path = f"/{target_name}"
            
            # Get default target pose
            default_pose = ee_config.get("default_target_pose", {})
            position = default_pose.get("position", [0.5, 0.0, 0.5])
            
            # Create target visualization using an in-scene cube primitive.
            # We fall back to a simple Cube so the example works even when the
            # DexCube asset isn’t present in the Nucleus server.
            self.prims.create_prim(
                target_path,
                "Cube",
                position=np.array(position),
                scale=np.array([target_size, target_size, target_size]),
            )
            
            # Store target prim for later updates
            self.target_prims.append({
                "name": target_name,
                "path": target_path,
                "end_effector": ee_config['name'],
                "position": position
            })
            
            print(f"Created target visualization: {target_name} at {position}")
            
    def create_action_graph(self):
        """Create the ROS2 action graph for the robot."""
        print("Creating action graph...")
        
        try:
            # Get ROS domain ID
            ros_domain_id = self._get_ros_domain_id()
            
            # Get robot configuration
            robot_stage_path = self.robot_config.get_isaac_stage_path()
            topics = self.robot_config.get_ros2_topics()
            
            # Create action graph based on Isaac Sim version
            if self.isaac_sim_ge_4_5_version:
                self._create_action_graph_4_5(ros_domain_id, robot_stage_path, topics)
            else:
                self._create_action_graph_legacy(ros_domain_id, robot_stage_path, topics)
                
            print("Action graph created successfully")
            return True
            
        except Exception as e:
            print(f"Error creating action graph: {e}")
            return False
            
    def _get_ros_domain_id(self) -> int:
        """Get ROS domain ID from environment."""
        try:
            ros_domain_id = int(os.environ.get("ROS_DOMAIN_ID", "0"))
            print(f"Using ROS_DOMAIN_ID: {ros_domain_id}")
            return ros_domain_id
        except ValueError:
            print("Invalid ROS_DOMAIN_ID. Using default value 0")
            return 0
            
    def _create_action_graph_4_5(self, ros_domain_id: int, robot_stage_path: str, topics: dict):
        """Create action graph for Isaac Sim 4.5+."""
        og_keys_set_values = [
            ("Context.inputs:domain_id", ros_domain_id),
            ("ArticulationController.inputs:robotPath", robot_stage_path),
            ("PublishJointState.inputs:topicName", topics.get("isaac_joint_states", "isaac_joint_states")),
            ("SubscribeJointState.inputs:topicName", topics.get("isaac_joint_commands", "isaac_joint_commands")),
        ]
        
        self.og.Controller.edit(
            {"graph_path": self.graph_path, "evaluator_name": "execution"},
            {
                self.og.Controller.Keys.CREATE_NODES: [
                    ("OnImpulseEvent", "omni.graph.action.OnImpulseEvent"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                    ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                    ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                    ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
                    ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ],
                self.og.Controller.Keys.CONNECT: [
                    ("OnImpulseEvent.outputs:execOut", "PublishJointState.inputs:execIn"),
                    ("OnImpulseEvent.outputs:execOut", "SubscribeJointState.inputs:execIn"),
                    ("OnImpulseEvent.outputs:execOut", "PublishClock.inputs:execIn"),
                    ("OnImpulseEvent.outputs:execOut", "ArticulationController.inputs:execIn"),
                    ("Context.outputs:context", "PublishJointState.inputs:context"),
                    ("Context.outputs:context", "SubscribeJointState.inputs:context"),
                    ("Context.outputs:context", "PublishClock.inputs:context"),
                    ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
                    ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                    ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                    ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
                    ("SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
                    ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
                ],
                self.og.Controller.Keys.SET_VALUES: og_keys_set_values,
            },
        )
        
    def _create_action_graph_legacy(self, ros_domain_id: int, robot_stage_path: str, topics: dict):
        """Create action graph for Isaac Sim versions prior to 4.5."""
        og_keys_set_values = [
            ("Context.inputs:domain_id", ros_domain_id),
            ("ArticulationController.inputs:robotPath", robot_stage_path),
            ("PublishJointState.inputs:topicName", topics.get("isaac_joint_states", "isaac_joint_states")),
            ("SubscribeJointState.inputs:topicName", topics.get("isaac_joint_commands", "isaac_joint_commands")),
        ]
        
        # Add usePath for legacy versions
        if self.is_legacy_isaacsim:
            og_keys_set_values.insert(1, ("ArticulationController.inputs:usePath", True))
            
        self.og.Controller.edit(
            {"graph_path": self.graph_path, "evaluator_name": "execution"},
            {
                self.og.Controller.Keys.CREATE_NODES: [
                    ("OnImpulseEvent", "omni.graph.action.OnImpulseEvent"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("Context", "omni.isaac.ros2_bridge.ROS2Context"),
                    ("PublishJointState", "omni.isaac.ros2_bridge.ROS2PublishJointState"),
                    ("SubscribeJointState", "omni.isaac.ros2_bridge.ROS2SubscribeJointState"),
                    ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
                    ("PublishClock", "omni.isaac.ros2_bridge.ROS2PublishClock"),
                ],
                self.og.Controller.Keys.CONNECT: [
                    ("OnImpulseEvent.outputs:execOut", "PublishJointState.inputs:execIn"),
                    ("OnImpulseEvent.outputs:execOut", "SubscribeJointState.inputs:execIn"),
                    ("OnImpulseEvent.outputs:execOut", "PublishClock.inputs:execIn"),
                    ("OnImpulseEvent.outputs:execOut", "ArticulationController.inputs:execIn"),
                    ("Context.outputs:context", "PublishJointState.inputs:context"),
                    ("Context.outputs:context", "SubscribeJointState.inputs:context"),
                    ("Context.outputs:context", "PublishClock.inputs:context"),
                    ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
                    ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                    ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                    ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
                    ("SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
                    ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
                ],
                self.og.Controller.Keys.SET_VALUES: og_keys_set_values,
            },
        )
        
    def finalize_setup(self):
        """Finalize the setup and prepare for simulation."""
        print("Finalizing setup...")
        
        # Set target prim for PublishJointState node
        robot_stage_path = self.robot_config.get_isaac_stage_path()
        
        if self.isaac_sim_ge_4_5_version:
            self.set_targets(
                prim=self.stage.get_current_stage().GetPrimAtPath(f"{self.graph_path}/PublishJointState"),
                attribute="inputs:targetPrim",
                target_prim_paths=[robot_stage_path],
            )
        else:
            # For legacy versions
            try:
                from isaacsim.core.nodes.scripts.utils import set_target_prims
                set_target_prims(
                    primPath=f"{self.graph_path}/PublishJointState",
                    targetPrimPaths=[robot_stage_path]
                )
            except ImportError:
                print("Warning: Could not set target prims for legacy version")
                
        # Update simulation app
        self.simulation_app.update()
        self.simulation_app.update()
        
        # Initialize physics
        self.simulation_context.initialize_physics()
        
        # Start simulation
        self.simulation_context.play()
        self.simulation_app.update()
        
        print("Setup complete. Simulation is running.")
        
    def update_target_position(self, target_name: str, new_position: list):
        """Update the position of a target visualization."""
        for target in self.target_prims:
            if target["name"] == target_name:
                target["position"] = new_position
                
                # Update the prim position
                prim = self.stage.get_current_stage().GetPrimAtPath(target["path"])
                if prim:
                    xform = self.UsdGeom.Xformable(prim)
                    translate_op = xform.AddTranslateOp()
                    translate_op.Set(self.Gf.Vec3d(new_position[0], new_position[1], new_position[2]))
                    
                print(f"Updated target {target_name} to position {new_position}")
                break
                
    def get_target_positions(self) -> dict:
        """Get current positions of all targets."""
        return {target["name"]: target["position"] for target in self.target_prims}
        
    def run_simulation(self):
        """Run the simulation loop."""
        print("Starting simulation loop...")
        print("Press Ctrl+C to stop")
        
        try:
            while self.simulation_app.is_running():
                # Run simulation step
                self.simulation_context.step(render=True)
                
                # Trigger ROS2 nodes
                self.og.Controller.set(
                    self.og.Controller.attribute(f"{self.graph_path}/OnImpulseEvent.state:enableImpulse"), 
                    True
                )
                
        except KeyboardInterrupt:
            print("\nStopping simulation...")
            
    def shutdown(self):
        """Shutdown the simulation."""
        print("Shutting down...")
        if self.simulation_context:
            self.simulation_context.stop()
        if self.simulation_app:
            self.simulation_app.close()
        print("Shutdown complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Isaac Sim launcher for MoveIt2 integration")
    parser.add_argument("--robot", "-r", default="default_robot", 
                       help="Robot configuration name")
    parser.add_argument("--config", "-c", default=None,
                       help="Path to robot configuration file")
    parser.add_argument("--headless", action="store_true",
                       help="Run Isaac Sim in headless mode")
    
    args = parser.parse_args()
    
    # Update config for headless mode
    if args.headless:
        CONFIG["headless"] = True
        
    try:
        # Load robot configuration
        print(f"Loading robot configuration: {args.robot}")
        config_loader = ConfigLoader(args.config)
        robot_config = config_loader.get_robot_config(args.robot)
        
        # Validate configuration
        if not config_loader.validate_config(args.robot):
            print("Error: Invalid robot configuration")
            return 1
            
        # Create launcher
        launcher = IsaacSimLauncher(robot_config)
        
        # Start simulation
        launcher.start_simulation_app()
        
        # Setup scene
        if not launcher.setup_scene():
            print("Error: Failed to setup scene")
            return 1
            
        # Create action graph
        if not launcher.create_action_graph():
            print("Error: Failed to create action graph")
            return 1
            
        # Finalize setup
        launcher.finalize_setup()
        
        # Run simulation
        launcher.run_simulation()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    finally:
        if 'launcher' in locals():
            launcher.shutdown()
            
    return 0


if __name__ == "__main__":
    sys.exit(main()) 