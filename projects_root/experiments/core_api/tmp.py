# meta_cfg_path= 'projects_root/experiments/benchmarks/cfgs/meta_cfg_arms.yml' #'projects_root/experiments/benchmarks/cfgs/meta_cfg_arms.yml'

# from curobo.util_file import load_yaml, join_path, get_world_configs_path
# load_yaml(join_path(get_world_configs_path(), "collision_base.yml"))
# meta_cfg = load_yaml(meta_cfg_path)

# from omni.isaac.kit import SimulationApp
# simulation_app = SimulationApp({**meta_cfg["env"]["simulation"]["init_app_settings"]})
# from omni.isaac.core import World
# my_world = World(stage_units_in_meters=1.0)
# while simulation_app.is_running():        
#     my_world.step(render=True)
# simulation_app.close()
    
import torch
from curobo.types.robot import JointState, RobotConfig
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel

from projects_root.utils.transforms import transform_poses_batched_optimized_for_spheres

def get_sphere_pos_and_r(js:JointState, robot_base_pose:list, crm:CudaRobotModel, frame='W',):
    """
    js: JointState object
    frame: 'W' or 'R'
    robot_base_pose: list of 7 elements [x,y,z,qw, qx,qy,qz]
    """
    js_tensor_2d = js.position # [1,DOF]
    ans = crm.forward(js_tensor_2d)
    p_ee, q_ee, _, _, p_links, q_links, prad = crm.forward() # https://curobo.org/_api/curobo.cuda_robot_model.cuda_robot_model.html#curobo.cuda_robot_model.cuda_robot_model.CudaRobotModelConfig
    d = {'ee': {'p': p_ee, 'q': q_ee}, 'links': {'p': p_links, 'q': q_links}, 'spheres': {'p': prad[:,:,:3], 'r': prad[:,:,3]}}
    
    # We first convert the poses to world frame
    if frame == 'W':
        for key in d.keys():
            pKey = d[key]['p']
            qKey = d[key]['q']
            if 'q' not in d[key].keys():
                qKey = torch.empty(pKey.shape[:-1] + torch.Size([4]), device=pKey.device)
                qKey[...,:] = torch.tensor([1,0,0,0],device=pKey.device, dtype=pKey.dtype)  # [1,0,0,0] is the identity quaternion
            else:
                qKey = d[key]['q']

            # OPTIMIZED VERSION: Use ultra-fast specialized function
            X_world = transform_poses_batched_optimized_for_spheres(torch.cat([pKey, qKey], dim=-1), robot_base_pose)
            pKey = X_world[...,:3]
            qKey = X_world[...,3:]
            d[key]['p'] = pKey
            d[key]['q'] = qKey

    return d    




