from abc import ABC, abstractmethod
from projects_root.experiments.core_api.environment.environment import Environment
from projects_root.experiments.core_api.environment.simulation_environment import SimulationEnvironment
from projects_root.experiments.core_api.environment.real_environment import RealEnvironment

class Challenge:
    def __init__(self, env:Environment):
        self.env = env
        pass
    @abstractmethod
    def pre_step(self,**kwargs:dict):
        pass
    @abstractmethod
    def post_step(self,**kwargs:dict):
        pass
    
 