"""
shared topics to write and read from during runtime 
"""
 
import torch
import numpy as np
from typing import List, Optional

class Topics:
        
    def __init__(self, n_envs=1, robots_per_env=4):
        self._topics = []
        for i in range(n_envs):
            self._topics.append([])
            for j in range(robots_per_env):
                self._topics[i].append({
                    "mpc_cfg": None,
                    "target": np.zeros(7),
                    "plans": [],
                    "cost": {},
                    "p_SpheresRolloutsFiltered": torch.tensor([]),
                    "rad_SpheresRolloutsFiltered": torch.tensor([])
                })
            

    def get_default_env(self):
        return self._topics[0] 

    @property
    def topics(self):
        return self._topics


# declare runtime_topics as a global variable
runtime_topics: Topics = None  # type: ignore

def get_topics():
    return runtime_topics

def init_runtime_topics(n_envs=1, robots_per_env=4):
    """
    initialize runtime_topics as a global variable
    """
    global runtime_topics
    runtime_topics = Topics(n_envs, robots_per_env)

