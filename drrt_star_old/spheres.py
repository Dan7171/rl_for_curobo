from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np

__all__ = ["Sphere", "RobotSphereModel"]


a = np.ndarray

@dataclass
class Sphere:
    """Light-weight sphere primitive used for collision tests.

    Parameters
    ----------
    center : np.ndarray, shape (3,)
        Cartesian position of the centre in *world* coordinates.
    radius : float
        Radius in metres.
    """

    center: a
    radius: float

    def distance(self, other: "Sphere") -> float:
        return float(np.linalg.norm(self.center - other.center) - self.radius - other.radius)

    def collides(self, other: "Sphere") -> bool:
        return self.distance(other) <= 0.0


class RobotSphereModel:
    """Collection of spheres approximating a single manipulator arm.

    Attributes
    ----------
    spheres : List[Sphere]
        Spheres expressed in the *robot base* frame.
    """

    def __init__(self, spheres: List[Sphere]):
        self._spheres = spheres

    # ------------------------------------------------------------------
    # Transformation helpers
    # ------------------------------------------------------------------
    def spheres_W(self, base_pose_W: np.ndarray) -> List[Sphere]:
        """Returns sphere list expressed in world frame.

        Parameters
        ----------
        base_pose_W : np.ndarray, shape (7,)
            `[x, y, z, qw, qx, qy, qz]` quaternion form.
        """
        p = base_pose_W[:3]
        # No rotation required if spheres are in base frame. If rotation is
        # desired, extend this function.
        return [Sphere(center=s.center + p, radius=s.radius) for s in self._spheres]

    # ------------------------------------------------------------------
    # Collision helpers
    # ------------------------------------------------------------------
    def collides_with(self, other: "RobotSphereModel", base_self: np.ndarray, base_other: np.ndarray) -> bool:
        for s1 in self.spheres_W(base_self):
            for s2 in other.spheres_W(base_other):
                if s1.collides(s2):
                    return True
        return False

    def collides_with_env(self, env_spheres: List[Sphere], base_self: np.ndarray) -> bool:
        for s_robot in self.spheres_W(base_self):
            if any(s_robot.collides(s_e) for s_e in env_spheres):
                return True
        return False

    # Expose underlying list
    @property
    def spheres(self) -> List[Sphere]:
        return self._spheres
