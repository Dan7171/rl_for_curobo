from __future__ import annotations

from typing import List
import numpy as np

from .spheres import Sphere, RobotSphereModel

__all__ = [
    "spheres_collide",
    "model_vs_model_collide",
    "model_vs_env_collide",
]


def spheres_collide(s1: Sphere, s2: Sphere) -> bool:
    """Fast sphere–sphere intersection test."""
    dist2 = np.sum((s1.center - s2.center) ** 2)
    return dist2 <= (s1.radius + s2.radius) ** 2


def model_vs_model_collide(m1: RobotSphereModel, m2: RobotSphereModel, base1: np.ndarray, base2: np.ndarray) -> bool:
    for s1 in m1.spheres_W(base1):
        for s2 in m2.spheres_W(base2):
            if spheres_collide(s1, s2):
                return True
    return False


def model_vs_env_collide(model: RobotSphereModel, env_spheres: List[Sphere], base: np.ndarray) -> bool:
    for s_r in model.spheres_W(base):
        if any(spheres_collide(s_r, s_e) for s_e in env_spheres):
            return True
    return False
