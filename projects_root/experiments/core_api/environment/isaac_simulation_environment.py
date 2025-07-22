# before importing this file, make sure simulation app is already running

from typing import Callable, Union
from projects_root.experiments.core_api.environment.simulation_environment import SimulationEnvironment
from curobo.util.usd_helper import UsdHelper
from projects_root.utils.usd_utils import load_usd_to_stage
from projects_root.utils.issacsim import init_app, wait_for_playing, activate_gpu_dynamics,make_world
from omni.isaac.core import World
import omni.usd

class IsaacSimulationEnvironment(SimulationEnvironment):
    def __init__(self, 
                pre_step_callback:Callable, 
                post_step_callback:Callable,
                stage_file:str=''
                ):
        super().__init__(pre_step_callback, post_step_callback)
        
        # reset stage
        self._stage = omni.usd.get_context().new_stage() # clear all obstacles
        self._stage.DefinePrim("/curobo", "Xform")  # Transform for CuRobo-specific objects        
        # initialize usd helper and load stage file if provided
        self._usd_helper = UsdHelper()  
        self._usd_helper.load_stage(self._stage)
        self._stage_file = stage_file
        if self._stage_file.endswith('.usd') or self._stage_file.endswith('.usda'):
            self._usd_helper.load_stage_from_file(self._stage_file) # set self.stage to the stage (self=usd_help)
        
        # initialize world        
        self._world = make_world(ground_plane=True, set_default_prim=True, to_Xform=True)        
        activate_gpu_dynamics(self._world)
        self._world.set_simulation_dt(0.03, 0.03)
        
    def step(self):
        return self._world.step()
    
    
   
   
# def reset(self):
#     # reset stage
#     self._stage = omni.usd.get_context().new_stage() # clear all obstacles
#     self._stage.DefinePrim("/curobo", "Xform")  # Transform for CuRobo-specific objects        
    
#     # initialize usd helper and load stage file if provided
#     self._usd_helper = UsdHelper()  
#     self._usd_helper.load_stage(self._stage) 
#     if self._stage_file.endswith('.usd') or self._stage_file.endswith('.usda'):
#         self._usd_helper.load_stage_from_file(self._stage_file) # set self.stage to the stage (self=usd_help)
    
#     # initialize world        
#     self._world = make_world(ground_plane=True, set_default_prim=True, to_Xform=True)        
#     activate_gpu_dynamics(self._world)
#     self._world.set_simulation_dt(0.03, 0.03)


