# before importing this file, make sure simulation app is already running
from projects_root.experiments.core_api.environment.environment import Environment

class RealEnvironment(Environment):
    def __init__(self, pre_step_callbacks:list[Callable], post_step_callbacks:list[Callable]):
        super().__init__(pre_step_callbacks, post_step_callbacks)

    def prepare(self, robot_cfgs:list[str]):
        pass
    
    def load_robots(self, robot_cfgs:list[str]):
        pass