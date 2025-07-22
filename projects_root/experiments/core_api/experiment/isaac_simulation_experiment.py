

from collections.abc import Callable
import time
from projects_root.experiments.core_api.environment.isaac_simulation_environment import IsaacSimulationEnvironment
from projects_root.experiments.core_api.challenge.challenge import Challenge
from projects_root.experiments.core_api.experiment.experiment import Experiment

class IsaacSimulationExperiment(Experiment):
    def __init__(self, 
                environment: IsaacSimulationEnvironment, 
                challenge: Challenge, 
                timeout: float,
                pre_step_callback:Callable,
                post_step_callback:Callable,
                **kwargs:dict
        ):
        super().__init__(environment, challenge, timeout)
        self._pre_step_callback = pre_step_callback
        self._post_step_callback = post_step_callback
    

            
    def execute(self,timeout:float,**kwargs:dict):
        start_time = time.time()
        while time.time() - start_time < timeout:
            
            
            
            
            pre_step_result:dict = self._pre_step_callback(**kwargs)
            self.environment.step(pre_step_result)
            
            self.challenge.post_step()
            self.challenge.get_metrics()
            self.challenge.get_stats()
            time.sleep(0.03)

