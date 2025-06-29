import omni.graph.core as og
import threading

_GRAPH_LOCK = threading.Lock()

def build_plan_publish_graph(robot_ids, col_pred_with=None, graph_path="/ROS_MultiRobot"):
    """Create/overwrite an OmniGraph that **publishes** and optionally **subscribes** to robot plans.

    This helper is now *idempotent*: calling it multiple times with the same
    arguments will **not** raise Duplicate-Graph or Duplicate-Node errors.  If
    the graph (or a particular node) already exists we simply re-use it and
    only create the missing pieces.

    Parameters
    ----------
    robot_ids : Iterable[int]
        List of integer robot IDs (0-based) that will get their own /robot_<id>/plan topic.
    col_pred_with : Iterable[Iterable[int]], optional
        List of lists of robot IDs (0-based) that each robot subscribes to.
        If None, each robot subscribes to everyone else (except itself).
    graph_path : str, optional
        USD path at which the graph should be created.  Default is "/ROS_MultiRobot".

    Returns
    -------
    og.GraphHandle
        Handle to the created graph (can be used with og.Controller functions).
    """
    keys = og.Controller.Keys

    # ------------------------------------------------------------------
    # 1) Ensure the *container* graph exists only once
    # ------------------------------------------------------------------
    with _GRAPH_LOCK:
        if og.get_graph_by_path(graph_path) is None:
            og.Controller.edit({"graph_path": graph_path, "evaluator_name": "push"}, {})

    if col_pred_with is None:
        # default: each robot subscribes to everyone else (except itself)
        col_pred_with = [[j for j in robot_ids if j != i] for i in range(len(robot_ids))]

    # ------------------------------------------------------------------
    # 2) For every robot create the missing nodes only (skip if they exist)
    # ------------------------------------------------------------------
    for idx, rid in enumerate(robot_ids):
        on_tick    = f"OnTick_{rid}"
        ros_ctx    = f"rosContext_{rid}"
        pub_string = f"pubPlan_{rid}"

        def _node_exists(name: str) -> bool:
            return og.get_node_by_path(f"{graph_path}/{name}") is not None

        # Build list of nodes that are *not* yet present
        nodes_to_create = []
        if not _node_exists(on_tick):
            nodes_to_create.append((on_tick, "omni.graph.action.OnTick"))
        if not _node_exists(ros_ctx):
            nodes_to_create.append((ros_ctx, "isaacsim.ros2.bridge.ROS2Context"))
        if not _node_exists(pub_string):
            nodes_to_create.append((pub_string, "isaacsim.ros2.bridge.ROS2PublishString"))

        og.Controller.edit(
            {"graph_path": graph_path},
            {
                keys.CREATE_NODES: nodes_to_create,
            },
        )

        # Set values (idempotent – will just overwrite to same value)
        og.Controller.edit(
            {"graph_path": graph_path},
            {
                keys.SET_VALUES: [
                    (f"{pub_string}.inputs:topicName", f"/robot_{rid}/plan"),
                    (f"{pub_string}.inputs:nodeNamespace", "robot"),
                ],
                keys.CONNECT: [
                    (f"{on_tick}.outputs:tick", f"{pub_string}.inputs:execIn"),
                    (f"{ros_ctx}.outputs:context", f"{pub_string}.inputs:context"),
                ],
            },
        )

        # ----------  Subscriptions for this robot ---------- #
        for src_id in col_pred_with[idx]:
            node_name = f"subPlan_{src_id}_for_{rid}"

            if not _node_exists(node_name):
                og.Controller.edit(
                    {"graph_path": graph_path},
                    {
                        keys.CREATE_NODES: [
                            (node_name, "isaacsim.ros2.bridge.ROS2SubscribeString"),
                        ],
                    },
                )

            # Idempotent value/connection set
            og.Controller.edit(
                {"graph_path": graph_path},
                {
                    keys.SET_VALUES: [
                        (f"{node_name}.inputs:topicName", f"/robot_{src_id}/plan"),
                        (f"{node_name}.inputs:nodeNamespace", "robot"),
                    ],
                    keys.CONNECT: [
                        (f"{on_tick}.outputs:tick", f"{node_name}.inputs:execIn"),
                        (f"{ros_ctx}.outputs:context", f"{node_name}.inputs:context"),
                    ],
                },
            )

    # Evaluate once so the graph appears immediately in UI (safe even if graph
    # existed before).
    og.Controller.evaluate_sync(graph_path)
    return og.get_graph_by_path(graph_path)


def push_plan(robot_id: int, plan_json: str, graph_path="/ROS_MultiRobot"):
    """Write a JSON string to the StringData pin for the given robot."""
    pin_path = f"{graph_path}/pubPlan_{robot_id}.inputs:stringData"
    og.Controller.set(pin_path, plan_json)


def fetch_plan(src_id: int, dst_id: int, graph_path="/ROS_MultiRobot") -> str:
    """Return latest JSON string received by dst from src (or "" if none)."""
    path = f"{graph_path}/subPlan_{src_id}_for_{dst_id}.outputs:stringData"
    try:
        return og.Controller.get(path) or ""
    except Exception:
        return "" 