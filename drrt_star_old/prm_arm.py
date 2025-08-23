from __future__ import annotations

"""prm_arm.py – Probabilistic Road-Map (PRM) builder specialised for articulated
robot arms.

Key differences from the simple `PRM` in `prm.py`:
  • Each state is a *joint configuration* q ∈ ℝᵏ (k = DoF).
  • Collision checking uses forward-kinematics with Pinocchio to place
    per-link spheres in the world frame.
  • Self-collision (between links of the same robot) is optionally checked.
  • The joint limits are read automatically from the Pinocchio model unless
    explicit limits are passed.

This file depends only on:
    pinocchio
    numpy
    drrt_star.spheres  (Sphere)
    drrt_star.collision (spheres_collide)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Sequence, TYPE_CHECKING
import numpy as np
import pinocchio as pin

from drrt_star_old.spheres import Sphere
from drrt_star_old.collision import spheres_collide

# -----------------------------------------------------------------------------
# Helper dataclass that associates one or more spheres with a Pinocchio frame id
# -----------------------------------------------------------------------------
@dataclass
class LinkSpheres:
    frame_id: int              # pinocchio frame id
    local_spheres: List[Sphere]  # positions expressed in *link* frame


class ArmPRM:
    """Build a PRM in joint space for a single manipulator arm."""

    def __init__(
        self,
        urdf_path: str,
        link_spheres: Dict[str, List[Sphere]],
        env_spheres: List[Sphere],
        n_samples: int = 400,
        k: int = 15,
        seed: int | None = None,
        joint_limits: Sequence[Tuple[float, float]] | None = None,
        check_self_collision: bool = True,
    ):
        if seed is not None:
            np.random.seed(seed)

        # ------------------------------------------------------------------
        #  Pinocchio model & FK data
        # ------------------------------------------------------------------
        self.model: pin.Model = pin.buildModelFromUrdf(urdf_path)
        self.data: pin.Data = self.model.createData()

        # joint limits
        if joint_limits is None:
            lower = self.model.lowerPositionLimit
            upper = self.model.upperPositionLimit
            self.joint_limits = np.vstack([lower, upper]).T  # (k,2)
        else:
            self.joint_limits = np.asarray(joint_limits)
        self.dof = self.joint_limits.shape[0]

        # preprocess link_spheres → frame ids
        self.link_spheres: List[LinkSpheres] = []
        for link_name, spheres in link_spheres.items():
            try:
                fid = self.model.getFrameId(link_name)
            except Exception:
                raise ValueError(f"Link '{link_name}' not found in URDF")
            self.link_spheres.append(LinkSpheres(frame_id=fid, local_spheres=spheres))

        self.env_spheres = env_spheres
        self.n_samples = n_samples
        self.k = k
        self.vertices: List[np.ndarray] = []
        self.edges: List[Tuple[int, int]] = []
        self._adj: dict[int, List[Tuple[int, float]]] = {}
        self.check_self_collision = check_self_collision

        self._build()

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------
    def _sample(self) -> np.ndarray:
        low, high = self.joint_limits[:, 0], self.joint_limits[:, 1]
        return np.random.uniform(low, high)

    # ------------------------------------------------------------------
    # Collision checking
    # ------------------------------------------------------------------
    def _fk_world_spheres(self, q: np.ndarray) -> List[Sphere]:
        """Return list of *all* robot spheres expressed in world frame."""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        out: List[Sphere] = []
        for ls in self.link_spheres:
            F = self.data.oMf[ls.frame_id]  # SE3 of link in world
            p = F.translation
            R = F.rotation
            for s in ls.local_spheres:
                centre_W = p + R @ s.center
                out.append(Sphere(center=centre_W, radius=s.radius))
        return out

    def _collision_free(self, q: np.ndarray) -> bool:
        spheres_W = self._fk_world_spheres(q)

        # 1. Env collision
        for s_r in spheres_W:
            if any(spheres_collide(s_r, s_e) for s_e in self.env_spheres):
                return False

        # 2. Self collision (basic O(n²) check; skip within-same-link)
        if self.check_self_collision:
            for i in range(len(spheres_W)):
                for j in range(i + 1, len(spheres_W)):
                    if spheres_collide(spheres_W[i], spheres_W[j]):
                        return False
        return True

    # ------------------------------------------------------------------
    # Build PRM
    # ------------------------------------------------------------------
    def _build(self):
        attempts = 0
        while len(self.vertices) < self.n_samples and attempts < self.n_samples * 20:
            q = self._sample()
            if self._collision_free(q):
                self.vertices.append(q)
            attempts += 1

        # connect k-nearest neighbours w.r.t Euclidean distance in joint space
        V = np.asarray(self.vertices)
        for i, qi in enumerate(V):
            dists = np.linalg.norm(V - qi, axis=1)
            nn_idx = np.argsort(dists)[1 : self.k + 1]
            for j in nn_idx:
                if i < j and self._local_planner(V[i], V[j]):
                    self._add_edge(i, j)

    def _add_edge(self, i: int, j: int):
        cost = float(np.linalg.norm(self.vertices[i] - self.vertices[j]))
        self.edges.append((i, j))
        self._adj.setdefault(i, []).append((j, cost))
        self._adj.setdefault(j, []).append((i, cost))

    def _local_planner(self, q1: np.ndarray, q2: np.ndarray, steps: int = 10) -> bool:
        """Simple linear interpolation & collision check between q1 and q2."""
        for alpha in np.linspace(0.0, 1.0, steps):
            q = (1 - alpha) * q1 + alpha * q2
            if not self._collision_free(q):
                return False
        return True

    # ------------------------------------------------------------------
    # Path search & utilities
    # ------------------------------------------------------------------
    def plan(self, q_start: np.ndarray, q_goal: np.ndarray) -> List[np.ndarray]:
        """Find shortest path on the roadmap, grafting start & goal if needed."""

        s_idx = self._graft_state(q_start)
        g_idx = self._graft_state(q_goal)

        import heapq

        dist = {s_idx: 0.0}
        prev: dict[int, int | None] = {s_idx: None}
        pq = [(0.0, s_idx)]

        while pq:
            d, u = heapq.heappop(pq)
            if u == g_idx:
                break
            if d > dist[u]:
                continue
            for v, w in self._adj.get(u, []):
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))

        if g_idx not in prev:
            return []

        # backtrack
        path_idx = []
        v = g_idx
        while v is not None:
            path_idx.append(v)
            v = prev[v]
        path_idx.reverse()
        return [self.vertices[i] for i in path_idx]

    def _graft_state(self, q: np.ndarray) -> int:
        """Ensure configuration q is in the roadmap; return its index."""
        # exact match?
        for i, v in enumerate(self.vertices):
            if np.allclose(v, q):
                return i

        # else create new vertex
        idx = len(self.vertices)
        self.vertices.append(q)
        self._adj[idx] = []

        V = np.asarray(self.vertices[:-1])
        dists = np.linalg.norm(V - q, axis=1)
        nn_idx = np.argsort(dists)[: self.k]
        for j in nn_idx:
            if self._local_planner(q, self.vertices[j]):
                self._add_edge(idx, j)
        return idx

    # shortcut and discretise helpers
    def shortcut(self, path: List[np.ndarray], iters: int = 100) -> List[np.ndarray]:
        if len(path) < 3:
            return path
        for _ in range(iters):
            i, j = sorted(np.random.choice(len(path), 2, replace=False))
            if j - i <= 1:
                continue
            if self._local_planner(path[i], path[j]):
                path = path[: i + 1] + path[j:]
        return path

    def discretise(self, path: List[np.ndarray], step: float = 0.05) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for a, b in zip(path[:-1], path[1:]):
            dist = np.linalg.norm(b - a)
            n = max(2, int(dist / step) + 1)
            for alpha in np.linspace(0, 1, n, endpoint=False):
                out.append((1 - alpha) * a + alpha * b)
        out.append(path[-1])
        return out

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def nearest(self, q: np.ndarray) -> int:
        V = np.asarray(self.vertices)
        return int(np.argmin(np.linalg.norm(V - q, axis=1)))

    def neighbours(self, idx: int) -> List[int]:
        return [j if i == idx else i for i, j in self.edges if i == idx or j == idx]
