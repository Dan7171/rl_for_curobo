from projects_root.experiments.core_api.motion_planner.motion_planner import MotionPlanner


class MpcPlanner(MotionPlanner):
    def __init__(self):
        super().__init__(motion_gen, plan_config)
        
        