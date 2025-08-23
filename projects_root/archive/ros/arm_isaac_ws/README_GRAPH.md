# OmniGraph Action Graph – Franka Panda ROS 2 Bridge

This file explains the purpose of every node that appears in the `/ROS_Robot` action graph constructed by `isaac_robot_graph.py` and shows how the nodes are wired together.

## 📦 Node Cheat-Sheet

* **On Tick** (`omni.graph.action.OnTick`)
  * Emits a Tick (exec) signal every physics frame.
  * Produces the current simulation **time** in seconds – we reuse it for the `/clock` header.

* **ROS2 Context** (`isaacsim.ros2.bridge.ROS2Context`)
  * Creates/owns the rclcpp context that all other ROS 2 nodes share.
  * Outputs a **context** object passed downstream.

* **ROS2 Subscribe Joint State** (`isaacsim.ros2.bridge.ROS2SubscribeJointState`)
  * Listens on `/robot/joint_command` (namespace `robot`).
  * Decodes `sensor_msgs/JointState` and exposes **position / velocity / effort** pin arrays plus **jointNames**.

* **Isaac Articulation Controller** (`isaacsim.core.nodes.IsaacArticulationController`)
  * Low-level driver that writes the commands into the Franka articulation.
  * Needs a **robotPath** parameter (`/World/Robot`) and accepts position / velocity / effort / jointNames.

* **ROS2 Publish Joint State** (`isaacsim.ros2.bridge.ROS2PublishJointState`)
  * Samples the articulation every frame and publishes `/robot/joint_states`.
  * Header `stamp` comes from the simulation **time** pin so downstream ROS 2 nodes run in sim-time.

* **ROS2 Publish Clock** (`isaacsim.ros2.bridge.ROS2PublishClock`)
  * Publishes a `rosgraph_msgs/Clock` message on `/clock` each frame, driven by the same simulation time value.

## 🔗 Connectivity Diagram

```
┌──────────────┐         exec          ┌──────────────────────┐
│   On Tick    │ ────────────────────▶ │ Isaac Articulation   │
│              │                      │     Controller       │
│  time ───────┼──────────────────────┐│                      │
└────┬─────────┘                      ││                      │
     │  exec                          ││ position/velocity/… │
     │                                │└──────────────────────┘
     │ exec                           │          ▲
     ▼                                │          │ jointNames
┌──────────────┐        context       │          │
│ ROS2 Context │──────────────────────┼──────────┘
└────┬─────────┘                      │
     │ context                       │
     │                               │
     ▼ exec                          │
┌───────────────────────────┐        │
│ ROS2 Subscribe JointState │────────┘
└───────────────────────────┘

# Feedback / telemetry path

On Tick exec ───────▶ ROS2 Publish JointState
      time ────────▶            │
ROS2 Context context ──────────▶│
                                ▼
                      /robot/joint_states

# Simulation time broadcast

On Tick exec ───────▶ ROS2 Publish Clock
      time ─────────▶│
ROS2 Context context ──────────▶│
                                ▼
                            /clock
```

Legend:
* Solid arrows = **exec** (trigger) connections.
* Dashed arrows = data pins (`time`, `context`, array commands…).

This architecture keeps the control loop fully inside OmniGraph while exposing standard ROS 2 topics for external controllers.
