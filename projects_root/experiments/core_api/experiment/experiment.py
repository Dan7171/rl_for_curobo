
from typing import Callable, List
from projects_root.experiments.core_api.challenge.challenge import Challenge
from projects_root.experiments.core_api.environment.environment import Environment
from abc import abstractmethod
import projects_root.examples.helper as examples_helper

class RobotCfg:
    def __init__(self,robot_cfg:str,ee_goals:list[str]):
        self.robot_cfg = robot_cfg
        self.ee_goals = ee_goals

class Experiment:
    def __init__(self,
                 environment: Environment,
                 challenge: Challenge,
                 timeout:float,
                 ):
        
        self.environment = environment # simulator/real world - where the experiment is run
        self.challenge = challenge # challenge to be run in the experiment
        self.timeout = timeout # in seconds - timeout for the experiment
        self.metrics = {}
    
    

    @abstractmethod
    def _load_actors(self):
        pass
    
    @abstractmethod
    def execute(self,timeout:float,**kwargs:dict):
        pass
    
    @abstractmethod
    def collect_results(self,**kwargs:dict):
        pass
    
    @abstractmethod
    def generate_results(self,**kwargs:dict):
        pass
    
    @abstractmethod
    def cleanup(self):
        pass
    
    @abstractmethod
    def clean_up(self):
        """
        each experiment is responsible for cleanup when it is done
        """
    
    def run(self):
        self.execute(self.timeout)
        results = self.generate_results()
        self.cleanup()
        return results

    
    
 
