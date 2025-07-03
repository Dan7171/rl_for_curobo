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


def reference_prims_from_usd(
    usd_path: str | os.PathLike[str],
    *,
    prim_paths: list[str] | None = None,
    dest_root: str = "/Imported",
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
    """Return a running :class:`SimulationApp` instance.

    If an application is already running it is returned unmodified; otherwise
    a new one is started with the given *headless* flag.
    """
    global _simulation_app  # pylint: disable=global-statement

    if _simulation_app is not None:
        return _simulation_app

    if not _ISAAC_AVAILABLE:
        raise RuntimeError(
            "Isaac-Sim Python modules are not available – make sure to run this "
            "function inside an Isaac-Sim environment or container."
        )

    _simulation_app = SimulationApp(  # type: ignore[call-arg]
        {
            "headless": headless,
            "width": "1920",
            "height": "1080",
        }
    )
    return _simulation_app


def load_usd(
    usd_path: str | os.PathLike[str],
    *,
    headless: bool = True,
    prim_paths: list[str] | None = None,
    dest_root: str = "/Imported",
) -> Tuple[Any, Any]:
    """Load *usd_path* into the current Isaac-Sim stage.

    The helper guarantees that a :class:`SimulationApp` is running (starting
    one if necessary) and then opens *usd_path* as the active USD stage.  The
    function blocks for a single update tick so that the stage is fully
    initialised before returning.

    Parameters
    ----------
    usd_path : str or :pyclass:`os.PathLike`
        Path (local file, Nucleus URL, …) pointing to the USD scene.
    headless : bool, optional
        When *True* (default) a new SimulationApp is started in headless
        mode.  Ignored when an app is already running.
    prim_paths : list[str] | None, optional
        Specific prim paths inside *usd_path* to reference.  *None* references
        the full file as a single prim.
    dest_root : str, default="/Imported"
        Root path inside the destination stage where the references will be
        inserted.  Sub-prims are created under this path.

    Returns
    -------
    Tuple[Any, Any]
        ``(simulation_app, stage)`` – The active SimulationApp handle and the
        opened USD stage.  The return type is *Any* to avoid strict runtime
        dependencies on Isaac-Sim for static analysis.
    """

    # Resolve to absolute path for local files so that `open_stage` can locate
    # them even when the current working directory changes later on.
    usd_path_str = str(pathlib.Path(usd_path).expanduser())

    # 1) Ensure an app and empty stage exist.
    sim_app, stage = create_empty_stage(headless=headless)

    # 2) Reference requested prims from the USD into the stage.
    reference_prims_from_usd(
        usd_path_str, prim_paths=prim_paths, dest_root=dest_root, stage=stage
    )

    # Small update tick so assets are visible before returning.
    sim_app.update()

    return sim_app, stage


# ---------------------------------------------------------------------------
# Convenience CLI usage when running this module directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _DEFAULT_USD = "usd_collection/envs/conveyor_track_round.usd"

    # Example: load only the conveyor, skip the ground plane for demonstration
    sim_app, _ = load_usd(
        _DEFAULT_USD,
        headless=False,
        # prim_paths=["/World/conveyor"],
        dest_root="/World/ImportedConveyor",
    )

    print("Isaac-Sim running – close the window or press Ctrl+C to exit.")

    try:
        while sim_app.is_running():
            sim_app.update()
    except KeyboardInterrupt:
        pass
    finally:
        sim_app.close()


