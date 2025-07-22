from projects_root.experiments.core_api.challenge.challenge import Challenge
from projects_root.experiments.core_api.environment.isaac_simulation_environment import IsaacSimulationEnvironment
class BinLoadingChallenge(Challenge):
    def __init__(self, env:IsaacSimulationEnvironment):
        super().__init__(env)
        world = self.env._world
        self.bin_prim = world.get_prim("/World/bin")
        self.bin_pose = self.bin_prim.GetAttribute("xformOp:transform").Get()
        self.bin_size = self.bin_prim.GetAttribute("size").Get()
        self.bin_center = self.bin_pose[:3, 3]
        self.bin_orientation = self.bin_pose[:3, :3]
        
    def pre_step(self, **kwargs:dict):
        
    
    def post_step(self, **kwargs:dict):
        pass
    
    def get_metrics(self, **kwargs:dict):
        
        
        
        
        
        