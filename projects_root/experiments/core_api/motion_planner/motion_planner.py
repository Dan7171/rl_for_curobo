from abc import abstractmethod
from curobo.types.math import Pose
from curobo.types.state import JointState
import numpy as np
import torch
from typing_extensions import Union
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenPlanConfig

class MotionPlanner:
    
    def __init__(self, tensor_args):
        self.tensor_args = tensor_args
    
    @abstractmethod
    def yield_action(self, pre_check_kwargs:dict, yield_next_action_kwargs:dict) -> JointState:
        pass

    def _convert_np_to_Pose( self, pose_np:np.ndarray):
        return Pose(position=self.tensor_args.to_device(torch.from_numpy(pose_np[:3])), quaternion=self.tensor_args.to_device(torch.from_numpy(pose_np[3:])))
