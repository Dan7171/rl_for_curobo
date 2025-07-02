from typing import List, Dict, Optional

import numpy as np

try:
    # Third-party – only available when USD/Isaac-Sim present
    from pxr import UsdGeom, Usd
    from curobo.util.usd_helper import UsdHelper  # type: ignore
except ImportError:  # pragma: no cover
    # We deliberately keep this import local to avoid forcing usd-core on
    # environments that do not have it (e.g. unit-tests on CI).
    UsdHelper = None  # type: ignore
    UsdGeom = None  # type: ignore
    Usd = None  # type: ignore


def list_relevant_prims(
    usd_helper: "UsdHelper",
    only_paths: List[str],
    ignore_substring: List[str],
) -> List[str]:
    """Return prim paths under *only_paths* minus *ignore_substring* filters."""
    prim_paths: List[str] = []
    for prim in usd_helper.stage.Traverse():
        p_str = str(prim.GetPath())
        if not any(p_str.startswith(op) for op in only_paths):
            continue
        if any(sub in p_str for sub in ignore_substring):
            continue
        # Skip the root "/World" Xform which matches only_paths but shouldn't be treated
        # as an obstacle.  Keep any other Xform or concrete geometry prim; CuRobo will
        # later decide if the prim can be converted to an obstacle.
        if p_str == "/World":
            continue
        if UsdGeom is not None and not (
            prim.IsA(UsdGeom.Xform)
            or prim.IsA(UsdGeom.Mesh)
            or prim.IsA(UsdGeom.Cube)
            or prim.IsA(UsdGeom.Sphere)
            or prim.IsA(UsdGeom.Cylinder)
            or prim.IsA(UsdGeom.Capsule)
        ):
            continue
        prim_paths.append(p_str)
    return prim_paths


def get_stage_poses(
    usd_helper: "UsdHelper",
    only_paths: List[str],
    reference_prim_path: Optional[str],
    ignore_substring: Optional[List[str]] = None,
) -> Dict[str, List[float]]:
    """Lightweight pose snapshot used each simulation step.

    Returns dict mapping prim-path to 7-DoF pose expressed in *reference_prim_path*.
    """
    if ignore_substring is None:
        ignore_substring = []

    prim_list = list_relevant_prims(usd_helper, only_paths, ignore_substring)
    return usd_helper.get_prim_poses(prim_list, reference_prim_path) 