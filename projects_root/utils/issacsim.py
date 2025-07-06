from __future__ import annotations

import time

# ---------------------------------------------------------------------------
# Compatibility helpers
# ---------------------------------------------------------------------------

from typing import Any, Sequence, Optional


# Lazily import the proper SimulationApp path depending on the installed
# Isaac-Sim release.  4.x ships it under ``isaacsim.simulation_app`` whereas
# 2023.x and earlier expose the legacy ``omni.isaac.kit`` entry-point.
def _resolve_sim_app_cls():
    """Return the SimulationApp class for the active Isaac-Sim installation."""

    try:
        # Isaac-Sim ≥ 4.0
        from isaacsim.simulation_app import SimulationApp as _SimApp  # type: ignore

        return _SimApp  # pragma: no cover
    except ImportError:  # fall back to legacy entry-point
        from omni.isaac.kit import SimulationApp as _SimApp  # type: ignore

        return _SimApp


def _import_enable_extension():
    """Return a callable that enables an extension across Isaac-Sim versions."""

    # Path changed in 4.5 – try both.
    candidates = [
        "omni.isaac.core.utils.extensions", # 4.5.0 or before (deprecated in 5.0)  
        "isaacsim.core.utils.extensions", # 5.0

    ]

    for mod_path in candidates:
        try:
            module = __import__(mod_path, fromlist=["enable_extension"])
            return getattr(module, "enable_extension")  # type: ignore[return-value]
        except (ImportError, AttributeError):
            continue

    # Fallback – use Kit extension manager directly.
    try:
        import omni.kit.app as kit_app  # type: ignore

        app = kit_app.get_app()
        mgr = app.get_extension_manager()  # type: ignore[attr-defined]

        def _enable(ext_name: str):  # type: ignore[return-type]
            if not mgr.is_extension_enabled(ext_name):
                mgr.set_extension_enabled(ext_name, True)

        return _enable
    except Exception as exc:  # pragma: no cover – should not happen in runtime envs
        raise RuntimeError("Unable to locate an API to enable Isaac-Sim extensions") from exc


# Shared instance so scripts can introspect if needed
_GLOBAL_SIM_APP: Any | None = None  # pylint: disable=invalid-name


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def init_app(
    app_settings: Optional[dict] = None,
    headless: bool = False,
    *,
    extensions: Optional[Sequence[str]] = [
        "isaacsim.asset.browser",
        "omni.asset_validator.ui",
        "omni.anim.motion_path.bundle",
        "omni.activity.ui",
        "omni.anim.curve.bundle",
        "omni.kit.tool.asset_exporter",
        "omni.kit.browser.asset",
        "omni.anim.graph.bundle",
        "isaacsim.examples.browser",
        "omni.simready.explorer",
        "omni.kit.tool.measure",
        "omni.graph.window.action", 
        "omni.graph.window.core",
        "isaacsim.asset.gen.conveyor",
        "isaacsim.asset.gen.conveyor.ui" ,
        "omni.kit.window.script_editor",
    ],
    preload_extensions: bool = False,
) -> Any:
    """Start (or reuse) an Isaac-Sim SimulationApp instance.

    Parameters
    ----------
    app_settings : dict, optional
        Keyword settings forwarded to :class:`SimulationApp`.  When *None* a
        small default window is opened (800×600 px, UI shown).
    extensions : Sequence[str] | None, optional
        List of extension identifiers to enable.  When *preload_extensions*
        is *True* the list is passed to the SimulationApp constructor so the
        extensions are loaded **before** the first update tick.  Otherwise
        they are activated **after** the app has started.
    preload_extensions : bool, default=False
        Attempt to activate *extensions* already during app launch.  This is
        useful when OmniGraph nodes from those extensions must be available
        immediately.  The feature is best-effort – if the current Isaac-Sim
        version does not support the *exts* launch-option the function falls
        back to post-launch activation.

    Returns
    -------
    SimulationApp
        Running SimulationApp handle. (Type: SimulationApp)
    """

    global _GLOBAL_SIM_APP  # pylint: disable=global-statement

    if _GLOBAL_SIM_APP is not None:
        # Reuse existing instance; optionally load additional extensions.
        if extensions:
            load_extensions(_GLOBAL_SIM_APP, list(extensions))
        return _GLOBAL_SIM_APP

    # Default launch config
    if app_settings is None:
        app_settings = {
            "headless": False,
            "width": "800",
            "height": "600",
        }

    # Best-effort pre-load – only some versions honour this key.
    if preload_extensions and extensions:
        app_settings = dict(app_settings)  # shallow copy to avoid caller side-effects
        app_settings.setdefault("exts", list(extensions))  # type: ignore[arg-type]

    # Ensure the isaacsim package is importable before anything else.
    try:
        import isaacsim  # noqa: F401  # pylint: disable=unused-import  # type: ignore
    except ImportError:
        # Harmless when running outside an Isaac-Sim environment – the import
        # becomes valid once the SimulationApp bootstraps the path.
        pass

    SimulationAppCls = _resolve_sim_app_cls()
    sim_app = SimulationAppCls(app_settings)  # type: ignore[arg-type]

    # Post-launch extension loading when requested or when pre-load failed.
    if extensions and not preload_extensions:
        load_extensions(sim_app, list(extensions))

    _GLOBAL_SIM_APP = sim_app
    return sim_app


def load_extensions(
    target: Any,
    extension_names: Sequence[str],
    *,
    update: bool = True,
) -> None:
    """Enable a list of Isaac-Sim extensions on a running application.

    The *target* argument can be one of:

    • the :class:`SimulationApp` instance
    • an :class:`omni.isaac.core.World` (accesses *world.simulation_app*)
    • a USD :class:`pxr.Usd.Stage` (global app is resolved)

    Parameters
    ----------
    target
        Any object that either *is* a SimulationApp or provides access to one.
    extension_names
        Identifiers such as ``"omni.graph.window.core"``.
    update : bool, default=True
        Call :pymeth:`SimulationApp.update` once after all extensions are
        enabled so their OmniGraph nodes are registered immediately.
    """

    if not extension_names:
        return

    # ------------------------------------------------------------------
    # Resolve SimulationApp handle
    # ------------------------------------------------------------------
    sim_app = None

    # Direct instance
    if hasattr(target, "is_running") and hasattr(target, "update"):
        sim_app = target  # type: ignore[assignment]
    else:
        # Common attribute used by World, etc.
        sim_app = getattr(target, "simulation_app", None)

    # Fallback to global handle
    if sim_app is None:
        sim_app = _GLOBAL_SIM_APP

    if sim_app is None:
        raise RuntimeError("Could not determine a running SimulationApp from the provided target object.")

    # ------------------------------------------------------------------
    # Enable extensions (version-agnostic)
    # ------------------------------------------------------------------
    enable_ext = _import_enable_extension()

    for name in extension_names:
        try:
            enable_ext(str(name))
        except Exception as exc:  # pragma: no cover – extension may be missing
            print(f"[Warning][issacsim] Failed to enable extension '{name}': {exc}")

    if update:
        try:
            sim_app.update()
        except Exception:
            pass

def make_world(ground_plane=True, to_Xform=False, set_default_prim=True)-> World:
    """
    NOTE: isaac sim app must be running already, otherwise it will raise an error.
    means that there is a scene already, and we are adding to it.

    Args:
        ground_plane: bool, default True
            If True, add a default ground plane (comes with lighting) to the world.
        to_Xform: bool, default False
            If True, wrap the world in an Xform.
        set_default_prim: bool, default True
            If True, set the world as the default prim.
    Before:
    # scene structure:
    / # scene root
    └─ Prim1... # some initial prims (or empty (scene root only, /))
    └─ Prim2... 
    └─ Prim3... 
    


    returns:
    / # scene root
    └─ Prim 1... # the same prims from earlier 
    └─ Prim 2... 
    └─ Prim 3... 

    # + new prims added by this function:
    ├─ physicsScene        ← added automatically for PhysX 
    ├─ PhysicsMaterials    ← added automatically for PhysX 
    └─ World(Xform if set_default_prim is True, else no type) ← visual content root that you (and the helper) create
        └─defaultGroundPlane # if ground_plane is True
            └─Looks(Scope)
              └─theGrid(Material)
                └─Shader(Shader)
            └─GroundPlane(Xform)
                └─CollisionPlane(Plane) 
            └─SphereLight(SphereLight)
            └─Environment
                └─Geometry
    
    """
    from omni.isaac.core import World
    from pxr import Usd
    
    # Initialize the world without opening a new stage
    world = World(stage_units_in_meters=1.0)
    
    # Add default ground plane if required
    if ground_plane:
        world.scene.add_default_ground_plane()
    
    # Ensure /World exists and is of type Xform if requested
    world_path = "/World"
    if to_Xform:
        prim = world.stage.GetPrimAtPath(world_path)
        if not prim.IsValid():
            # Create /World as Xform if it doesn't exist
            prim = world.stage.DefinePrim(world_path, "Xform")
        elif prim.GetTypeName() != "Xform":
            # Optionally redefine type (note: not strictly required unless type matters)
            Usd.Prim(prim).GetSpecifier()  # No-op here but placeholder for logging/debug

    # Set /World as default prim if requested
    if set_default_prim:
        world.stage.SetDefaultPrim(world.stage.GetPrimAtPath(world_path))

    return world
    # from omni.isaac.core import World
    # world = World(stage_units_in_meters=1.0)
    # if ground_plane:
    #     world.scene.add_default_ground_plane()
    # if to_Xform:
    #     _xform = world.stage.DefinePrim("/World", "Xform") # define the type of the world to be Xform
    # if set_default_prim:
    #     world.stage.SetDefaultPrim(world.stage.GetPrimAtPath("/World"))

    # return world



def wait_for_playing(my_world, simulation_app, autoplay=False):
    """
    Wait for the simulation to start playing.
    """
    playing = False
    while simulation_app.is_running() and not playing:
        my_world.step(render=True)
        if my_world.is_playing():
            playing = True
        else:
            if autoplay: # if autoplay is enabled, play the simulation immediately
                my_world.play()
                while not my_world.is_playing():
                    print("blocking until playing is confirmed...")
                    time.sleep(0.1)
                playing = True
            else:
                print("Waiting for play button to be pressed...")
                time.sleep(0.1)
    
    my_world.step(render=True)
    my_world.reset()


def activate_gpu_dynamics(my_world):
    """
    Activates GPU dynamics for the given world.
    """
    my_world_physics_context = my_world.get_physics_context()
    if not my_world_physics_context.is_gpu_dynamics_enabled():
        print("GPU dynamics is disabled. Initializing GPU dynamics...")
        my_world_physics_context.enable_gpu_dynamics(True)
        assert my_world_physics_context.is_gpu_dynamics_enabled()
        print("GPU dynamics is enabled")



if __name__ == "__main__":
    sim_app = init_app()
    print("Isaac-Sim running – close the window or press Ctrl+C to exit.")
    try:
        while sim_app.is_running():
            sim_app.update()
    except KeyboardInterrupt:
        pass
    finally:
        sim_app.close()
    # sim_app, stage = load_usd("usd_collection/envs/World-_360_convoyer.usd", headless=False)
    # print("Isaac-Sim running – close the window or press Ctrl+C to exit.")