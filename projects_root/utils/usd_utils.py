"""Utility helpers for quickly spinning up an Isaac-Sim environment.

At the moment this module contains a single public helper:

    load_usd(path: str, *, headless: bool = True) -> Tuple["SimulationApp", "Usd.Stage"]

which starts (or re-uses) a SimulationApp instance and opens the given USD
file into the stage.  The function returns both the
`SimulationApp` handle and the loaded USD stage so callers can continue
working with the scene.

Example
-------
>>> from projects_root.examples.load_env import load_usd
>>> sim_app, stage = load_usd("/home/user/my_scene.usd", headless=False)
>>> # … interact with the stage …
>>> sim_app.close()
"""
from __future__ import annotations

# Standard Library
from typing import Tuple, Any
import pathlib
import os
from projects_root.utils.issacsim import init_app, make_world

# ---------------------------------------------------------------------------
# Isaac-Sim / Omniverse Kit compatibility layer
# ---------------------------------------------------------------------------
# In Isaac-Sim 4.0 the preferred entry-point is :pymod:`isaacsim.simulation_app`,
# whereas older releases expose :pymod:`omni.isaac.kit`.  We try the new path
# first and gracefully fall back to the legacy import so that the helper works
# across a wider range of versions.

SimulationApp = None  # Will be resolved dynamically

try:  # New Isaac-Sim ≥ 4.0
    from isaacsim.simulation_app import SimulationApp as _SimulationApp  # type: ignore

    SimulationApp = _SimulationApp  # type: ignore[assignment]
except ImportError:  # pragma: no cover – probably an older version
    try:
        from omni.isaac.kit import SimulationApp as _SimulationApp  # type: ignore

        SimulationApp = _SimulationApp  # type: ignore[assignment]
    except ImportError:  # pragma: no cover – No Isaac-Sim at all
        SimulationApp = None  # type: ignore[assignment]

# Determine availability flag for quick checks elsewhere in the module.
_ISAAC_AVAILABLE = SimulationApp is not None

# Only attempt to import pxr.Usd for type annotations when present.
try:  # pragma: no cover – optional, only for type hints
    from pxr import Usd  # type: ignore
except ImportError:
    Usd = object  # type: ignore

# Global handle – created lazily for the process so multiple calls share one
# viewer / physics instance.
_simulation_app: Any | None = None  # pylint: disable=invalid-name

# ---------------------------------------------------------------------------
# Stage utilities
# ---------------------------------------------------------------------------


def create_empty_stage(*, headless: bool = True, base_frame: str = "/World") -> Tuple[Any, Any]:
    """Start Isaac-Sim (if necessary) and return a fresh, empty stage.

    Parameters
    ----------
    headless : bool, optional
        Start the application headless if a new instance is created.
    base_frame : str, default="/World"
        Path that will be defined as an ``Xform`` and set as the default prim.

    Returns
    -------
    Tuple[Any, Any]
        ``(simulation_app, stage)``
    """

    sim_app = get_simulation_app(headless)

    # Late import to ensure Kit extensions are initialised.
    import omni.usd  # type: ignore

    ctx = omni.usd.get_context()  # type: ignore[attr-defined]
    ctx.new_stage()
    stage = ctx.get_stage()

    # Set default prim / base frame for convenience.
    from pxr import UsdGeom  # type: ignore

    root = stage.DefinePrim(base_frame, "Xform")
    stage.SetDefaultPrim(root)

    return sim_app, stage


def load_prims_from_usd(
    usd_path: str | os.PathLike[str],
    *,
    prim_paths: list[str] | None = None,
    dest_root: str = "/World",
    stage: Any | None = None,
) -> list[str]:
    """Reference selected prims from *usd_path* into *stage*.

    Parameters
    ----------
    usd_path : str or PathLike
        Source USD to reference.
    prim_paths : list[str] | None, optional
        Specific prim paths inside *usd_path* to reference.  *None* references
        the full file as a single prim.
    dest_root : str, default="/Imported"
        Root path inside the destination stage where the references will be
        inserted.  Sub-prims are created under this path.
    stage : Usd.Stage, optional
        Destination stage.  If *None*, the current stage returned by
        ``omni.usd.get_context()`` is used.
r

    Returns
    -------
    list[str]
        Paths of the prims that were created in *stage*.
    """


    import omni.usd  # type: ignore
    from pxr import Usd  # type: ignore

    ctx = omni.usd.get_context()  # type: ignore[attr-defined]
    if stage is None:
        stage = ctx.get_stage()

    usd_path_str = str(pathlib.Path(usd_path).expanduser())

    created_paths: list[str] = []

    # Ensure the root exists
    root_prim = stage.GetPrimAtPath(dest_root)
    if not root_prim.IsValid():
        root_prim = stage.DefinePrim(dest_root, "Xform")

    if prim_paths is None:
        # Reference the whole USD under *dest_root*.
        ref = root_prim.GetReferences()
        ref.AddReference(assetPath=usd_path_str)
        created_paths.append(dest_root)
        return created_paths

    # Otherwise reference individual prims under dest_root/*basename*.
    for src_path in prim_paths:
        name = src_path.strip("/").split("/")[-1]
        dst_path = f"{dest_root}/{name}"
        prim = stage.DefinePrim(dst_path, "Xform")
        ref = prim.GetReferences()
        ref.AddReference(assetPath=usd_path_str, primPath=src_path)
        created_paths.append(dst_path)

    return created_paths

def get_simulation_app(headless: bool = True):  # -> SimulationApp when available
    sim_app = init_app(headless=headless)
    return sim_app


if __name__ == "__main__":

    # Example: load only the conveyor, skip the ground plane for demonstration
    sim_app = init_app()
    world = make_world(set_default_prim=True, to_Xform=False)
    # created_paths = load_prims_from_usd(
    #     "usd_collection/envs/World-_360_convoyer.usd",
    #     prim_paths=["/World/_360_convoyer"],
    #     dest_root="/World/_360_convoyer",
    #     stage=world.stage,

    # )
    #print('USD loaded to stage. Loaded prims to next paths:\n', created_paths)

    print("Isaac-Sim running – close the window or press Ctrl+C to exit.")
    try:
        while sim_app.is_running():
            sim_app.update()
    except KeyboardInterrupt:
        pass
    finally:
        sim_app.close()


