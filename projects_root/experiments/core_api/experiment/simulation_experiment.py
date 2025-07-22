import time
from abc import abstractmethod
from projects_root.experiments.core_api.environment.simulation_environment import SimulationEnvironment
from projects_root.experiments.core_api.challenge.simulation_challenge import SimulationChallenge
from projects_root.experiments.core_api.experiment.experiment import Experiment

class SimulationExperiment(Experiment):
    def __init__(self, sim_env: SimulationEnvironment, challenge: SimulationChallenge, timeout: float):
        super().__init__(sim_env, challenge, timeout)

    @abstractmethod
    def execute(self,timeout:float,**kwargs:dict):
        pass
    
    def collect_results(self,**kwargs:dict):
        pass
    
    def generate_results(self,**kwargs:dict):
        pass