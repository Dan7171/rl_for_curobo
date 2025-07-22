from typing import Callable
from abc import abstractmethod
class Environment:
    def __init__(self,  **kwargs:dict):
        self._kwargs = kwargs
    

    @abstractmethod
    def reset(self):
        """
        prepare env for experiment
        """
        pass
    
    @abstractmethod
    def step(self, pre_step_result:dict={}):
        """
        how environment is moving
        note: define only for simulation environments, real environments can just return None
        """
        pass


        