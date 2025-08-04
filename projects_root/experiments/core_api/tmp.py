meta_cfg_path= 'projects_root/experiments/benchmarks/cfgs/meta_cfg_arms.yml' #'projects_root/experiments/benchmarks/cfgs/meta_cfg_arms.yml'

from curobo.util_file import load_yaml, join_path, get_world_configs_path
load_yaml(join_path(get_world_configs_path(), "collision_base.yml"))
meta_cfg = load_yaml(meta_cfg_path)

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({**meta_cfg["env"]["simulation"]["init_app_settings"]})
from omni.isaac.core import World
my_world = World(stage_units_in_meters=1.0)
while simulation_app.is_running():        
    my_world.step(render=True)
simulation_app.close()
    