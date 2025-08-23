from __future__ import annotations

from typing import List, Tuple
import numpy as np
from .spheres import RobotSphereModel
from .collision import model_vs_env_collide, model_vs_model_collide

class PRM:
    """Very small disk-based probabilistic road-map in joint space."""

    def __init__(
        self,
        robot_model: RobotSphereModel,
        joint_limits: List[Tuple[float, float]],
        env_spheres: List["Sphere"],
        n_samples: int = 200,
        k: int = 10,
    ):
        self.robot_model = robot_model
        self.joint_limits = np.asarray(joint_limits)
        self.env_spheres = env_spheres
        self.n_samples = n_samples
        self.k = k
        self.vertices: List[np.ndarray] = []
        self.edges: List[Tuple[int, int]] = []

        self._build()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _sample(self) -> np.ndarray:
        low = self.joint_limits[:, 0]
        high = self.joint_limits[:, 1]
        return np.random.uniform(low, high)

    def _collision_free(self, q: np.ndarray) -> bool:
        base = np.zeros(7)  # assume base at origin, identity quat
        return not model_vs_env_collide(self.robot_model, self.env_spheres, base)

    def _build(self):
        attempts = 0
        while len(self.vertices) < self.n_samples and attempts < self.n_samples * 10:
            q = self._sample()
            if self._collision_free(q):
                self.vertices.append(q)
            attempts += 1

        # connect neighbours (brute force k-nearest)
        V = np.asarray(self.vertices)
        for i, qi in enumerate(V):
            dists = np.linalg.norm(V - qi, axis=1)
            nn_idx = np.argsort(dists)[1 : self.k + 1]
            for j in nn_idx:
                if i < j:
                    self.edges.append((i, j))

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def nearest(self, q: np.ndarray) -> int:
        V = np.asarray(self.vertices)
        return int(np.argmin(np.linalg.norm(V - q, axis=1)))

    def neighbours(self, idx: int) -> List[int]:
        nbs = []
        for i, j in self.edges:
            if i == idx:
                nbs.append(j)
            elif j == idx:
                nbs.append(i)
        return nbs
