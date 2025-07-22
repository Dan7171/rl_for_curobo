from projects_root.experiments.core_api.environment.real_environment import RealEnvironment
from projects_root.experiments.core_api.experiment.experiment import Experiment

class RealExperiment(Experiment):
    def __init__(self, robot_cfgs: list[str], real_env: RealEnvironment, challenge: Challenge, timeout: float):
        super().__init__(robot_cfgs, real_env, challenge, timeout)
        
        
        