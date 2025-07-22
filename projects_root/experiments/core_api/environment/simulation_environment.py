from projects_root.experiments.core_api.environment.environment import Environment
from abc import abstractmethod
from typing import Callable

class SimulationEnvironment(Environment):
    def __init__(self, pre_step_callback:Callable, post_step_callback:Callable):
        super().__init__(pre_step_callback, post_step_callback)
        
    def setup(self,**kwargs:dict):
        """
        robot_cfgs: list of robot config paths
        """
        pass
    
    @abstractmethod
    def step(self,**kwargs:dict):
        pass
    
    @abstractmethod
    def load_robots(self, robot_cfgs:list[str]):
        for robot_cfg in robot_cfgs:
            self.load_robot(robot_cfg)
    
    @abstractmethod
    def load_background(self, **kwargs:dict):
        pass