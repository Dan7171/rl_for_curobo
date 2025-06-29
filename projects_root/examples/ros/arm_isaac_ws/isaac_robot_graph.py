#!/usr/bin/env python3
"""
Isaac Sim Robot with Joint Control - Replicated from Camera Example
Follows the exact working pattern of isaac_camera_working_movable.py
"""
import sys
import signal
import time
import traceback

# -------------------------------------------------
#  Create SimulationApp *first* and immediately load
#  the core extensions so that OmniGraph registers
#  the physics-related nodes during the very first
#  tick.  Only afterwards do we import the modules
#  that depend on those extensions.
# -------------------------------------------------
from isaacsim.simulation_app import SimulationApp

CONFIG = {
    "renderer": "RaytracedLighting",
    "headless": False,
    "width": "800",
    "height": "600",
}

# Launch Omniverse Kit
simulation_app = SimulationApp(CONFIG)

# Enable required extensions *before* importing the
# OmniGraph core package so that their node types are
# registered during the first `update()`.
from isaacsim.core.utils import extensions

print("Enabling isaacsim.ros2.bridge extension…")
extensions.enable_extension("isaacsim.ros2.bridge")
# enable omni graph editor
extensions.enable_extension("omni.graph.window.core") # editor extension
extensions.enable_extension("omni.graph.window.action") # editor extension
extensions.enable_extension("omni.graph.window.generic") # editor extension

# Force one OmniGraph tick – this registers all node
# descriptors provided by the extensions we just
# enabled (isaacsim.ros2.bridge, core nodes are already loaded by default).
simulation_app.update()

# Now it is safe to import the rest of the Isaac/Omni
# Python modules.
import carb
import omni
import omni.graph.core as og
import usdrt.Sdf
from isaacsim.core.api import SimulationContext
from isaacsim.core.utils import stage, prims
from isaacsim.robot.manipulators.examples.franka import Franka
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics


class RobotControlNode:
    def __init__(self):
        self.simulation_context = None
        self.shutdown_requested = False
        self.robot = None
        self.ros_robot_graph = None
        self.graph_created = False

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_requested = True

    def initialize_isaac_sim(self):
        """Initialize Isaac Sim, load room, and add a physics scene."""
        print("Initializing Isaac Sim simulation context...")
        self.simulation_context = SimulationContext(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
        
        current_stage = omni.usd.get_context().get_stage()

        # Load a basic room scene from Isaac assets
        room_usd_path = "https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Environments/Simple_Room/simple_room.usd"
        prims.create_prim("/World/Room", "Xform", usd_path=room_usd_path)
        print(f"✓ Successfully loaded room: {room_usd_path}")

        # Add lighting
        distant_light = UsdLux.DistantLight.Define(current_stage, "/World/DistantLight")
        distant_light.CreateIntensityAttr(3000.0)
        print("✓ Room lighting setup complete")

        # Add Physics Scene
        scene = UsdPhysics.Scene.Define(current_stage, "/World/physicsScene")
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)
        print("✓ Physics scene added")
        
        # Physics is NOT initialized here. It will be initialized after the robot is loaded.
        simulation_app.update()
        
        print("✓ Isaac Sim initialized successfully")

    def create_robot_and_ros_graph(self):
        """Creates the robot and the ROS graph. This should only be called once the simulation is running."""
        if self.graph_created:
            return

        print("Attempting to create robot and ROS graph...")
        
        # Create Franka robot at a specified location
        self.robot = Franka(prim_path="/World/Robot", name="franka_robot")
        print("✓ Robot prim created at /World/Robot")
        
        # Start the physics simulation right before creating the graph
        print("Playing simulation to bring physics online…")
        self.simulation_context.play()
        
        # Define paths
        ROBOT_PRIM_PATH = "/World/Robot"
        ROS_GRAPH_PATH = "/ROS_Robot"
        
        print("Creating ROS2 action graph...")
        keys = og.Controller.Keys
        (self.ros_robot_graph, _, _, _) = og.Controller.edit(
            {
                "graph_path": ROS_GRAPH_PATH,
                "evaluator_name": "push"
            },
            {
                keys.CREATE_NODES: [
                    ("OnTick", "omni.graph.action.OnTick"),
                    ("articulationController", "isaacsim.core.nodes.IsaacArticulationController"),
                    ("rosContext", "isaacsim.ros2.bridge.ROS2Context"),
                    ("publishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                    ("subscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                    # New: publish /clock so external ROS 2 nodes can use sim-time
                    ("publishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ],
                keys.CONNECT: [
                    ("OnTick.outputs:tick", "articulationController.inputs:execIn"),
                    ("OnTick.outputs:tick", "publishJointState.inputs:execIn"),
                    ("OnTick.outputs:tick", "subscribeJointState.inputs:execIn"),
                    # Exec connection for clock publisher
                    ("OnTick.outputs:tick", "publishClock.inputs:execIn"),
                    ("rosContext.outputs:context", "publishJointState.inputs:context"),
                    ("rosContext.outputs:context", "subscribeJointState.inputs:context"),
                    ("rosContext.outputs:context", "publishClock.inputs:context"),
                    ("subscribeJointState.outputs:positionCommand", "articulationController.inputs:positionCommand"),
                    ("subscribeJointState.outputs:velocityCommand", "articulationController.inputs:velocityCommand"),
                    ("subscribeJointState.outputs:effortCommand", "articulationController.inputs:effortCommand"),
                    # Allow partial joint commands by passing explicit indices list
                    ("subscribeJointState.outputs:jointNames", "articulationController.inputs:jointNames"),
                    # Provide simulation time directly from OnTick
                    ("OnTick.outputs:time", "publishClock.inputs:timeStamp"),
                    ("OnTick.outputs:time", "publishJointState.inputs:timeStamp"),
                ],
                keys.SET_VALUES: [
                    ("articulationController.inputs:robotPath", ROBOT_PRIM_PATH),
                    ("publishJointState.inputs:targetPrim", [usdrt.Sdf.Path(ROBOT_PRIM_PATH)]),
                    ("publishJointState.inputs:topicName", "joint_states"),
                    ("publishJointState.inputs:nodeNamespace", "robot"),
                    ("subscribeJointState.inputs:topicName", "joint_command"),
                    ("subscribeJointState.inputs:nodeNamespace", "robot"),
                    # Configure clock publisher
                    ("publishClock.inputs:topicName", "clock"),
                ],
            },
        )

        og.Controller.evaluate_sync(self.ros_robot_graph)
        simulation_app.update()

        self.graph_created = True
        print("✓ Robot and ROS2 Action Graph created successfully")
        print("🚀 Resuming simulation...")

    def run_simulation(self):
        """Run the main simulation loop, creating the graph after a delay."""
        print("🚀 Starting simulation loop...")
        
        frame_count = 0
        while simulation_app.is_running() and not self.shutdown_requested:
            self.simulation_context.step(render=True)
            
            # Delay graph creation until simulation is running
            if not self.graph_created and frame_count > 20:
                self.create_robot_and_ros_graph()

            frame_count += 1

            if self.simulation_context.is_playing() is False and self.graph_created:
                break
        
        if self.simulation_context.is_playing():
            self.simulation_context.stop()

def main():
    print("=== Isaac Sim Robot Control - Camera Pattern Replica ===")
    
    robot_node = RobotControlNode()
    
    try:
        robot_node.initialize_isaac_sim()
        # NOTE: Graph creation is now deferred to the simulation loop
        robot_node.run_simulation()
    except Exception as e:
        print("❌ Fatal error in main loop")
        traceback.print_exc()
    finally:
        print("🧹 Cleaning up and shutting down...")
        simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
        # Ensure cleanup is called on catastrophic failure
        simulation_app.close()
    sys.exit(0) 