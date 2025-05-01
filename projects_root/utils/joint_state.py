
if __name__ == '__main__':
    from omni.isaac.kit import SimulationApp  
    simulation_app = SimulationApp({"headless": True})

import torch
from omni.isaac.core.utils.types import JointsState 
from curobo.types.state import JointState as cuJointState

# def isaac_js_to_curobo_js(isaac_js:JointsState) -> cuJointState:
    
#     # sim_js = JointsState (
#     #     position=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     #     velocity=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     #     acceleration=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     # )
#     # cu_js = cuJointState(torch.from_numpy(isaac_js.positions), torch.from_numpy(isaac_js.velocities), torch.zeros(isaac_js.num_joints), isaac_js.joint_names, torch.zeros(isaac_js.num_joints), self.tensor_args)
#     #     n_dofs = pi_mpc_means.shape[1]
#     #     js_state = JointState(torch.from_numpy(js_state_sim.positions[:n_dofs]), torch.from_numpy(js_state_sim.velocities[:n_dofs]), torch.zeros(n_dofs), self.get_dof_names(),torch.zeros(n_dofs), self.tensor_args)
        