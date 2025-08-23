from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np


@dataclass
class Node:
    """Simple tree/graph node used by dRRT* search."""

    q: np.ndarray
    parent: Optional["Node"] = None
    cost: float = 0.0
    children: List["Node"] = field(default_factory=list)

    def distance(self, other: "Node") -> float:
        return float(np.linalg.norm(self.q - other.q))

    def add_child(self, child: "Node") -> None:
        self.children.append(child)

    def path(self) -> List["Node"]:
        n: Optional[Node] = self
        out: List[Node] = []
        while n is not None:
            out.append(n)
            n = n.parent
        return list(reversed(out))
