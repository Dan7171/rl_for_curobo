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
        # "isaacsim.asset.browser",
        # "omni.asset_validator.ui",
        # "omni.anim.motion_path.bundle",
        # "omni.activity.ui",
        # "omni.anim.curve.bundle",
        # "omni.kit.tool.asset_exporter",
        # "omni.kit.browser.asset",
        # "omni.anim.graph.bundle",
        # "isaacsim.examples.browser",
        # "omni.simready.explorer",
        # "omni.kit.tool.measure",
        # "omni.graph.window.action", 
        # "omni.graph.window.core",
        # "isaacsim.asset.gen.conveyor",
        # "isaacsim.asset.gen.conveyor.ui" ,
        # "omni.kit.window.script_editor",
        # "omni.isaac.articulation_inspector",
    ],
    preload_extensions: bool = False,
    performance_settings: Optional[dict] = None,
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
    # # CRITICAL: Isaac Sim must be imported FIRST before any other modules
    # try:
    #     import isaacsim
    # except ImportError:
    #     pass

    global _GLOBAL_SIM_APP  # pylint: disable=global-statement

    if _GLOBAL_SIM_APP is not None:
        # Reuse existing instance; optionally load additional extensions.
        if extensions:
            load_extensions(_GLOBAL_SIM_APP, list(extensions))
        return _GLOBAL_SIM_APP

    # Default launch config
    if app_settings is None:
        # Common:

            # 640×480 (VGA)

            # 1280×720 (HD)

            # 1920×1080 (Full HD)

            # 2560×1440 (QHD)

            # 3840×2160 (4K UHD)


        app_settings = {
            "headless": False,
            # "width": "640",
            # "height": "480",
            # **{"renderer": "RaytracedLighting", "headless": False,"width": "800",  "height": "600",}

            
        }

    # Best-effort pre-load – only some versions honour this key.
    if preload_extensions and extensions:
        app_settings = dict(app_settings)  # shallow copy to avoid caller side-effects
        app_settings.setdefault("exts", list(extensions))  # type: ignore[arg-type]

    # # Ensure the isaacsim package is importable before anything else.
    # try:
    #     import isaacsim  # noqa: F401  # pylint: disable=unused-import  # type: ignore
    #     print("isaacsim imported")
    # except ImportError:
    #     # Harmless when running outside an Isaac-Sim environment – the import
    #     # becomes valid once the SimulationApp bootstraps the path.
    #     pass

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
    
    # Initialize the world without opening a new stage in /World
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
            # Usd.Prim(prim).GetSpecifier()  # No-op here but placeholder for logging/debug
            _ = world.stage.DefinePrim("/World", "Xform")
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

def setup_sim_performance(world, 
                          physics_step_dt:Optional[float]=None, 
                          min_frame_rate:Optional[float]=None, 
                          gpu_dynamics_enabled:Optional[bool]=None, 
                          scene_opt_enabled:Optional[bool]=None, 
                          merge_mesh_tool_enabled:Optional[bool]=None,
                          rtx_mode:Optional[str]=None,
                          diable_materials_and_lights:Optional[bool]=None,
                          dlss_mode:Optional[int]=None
                          ):
    """
    This function allows to set the settings suggested in the next tutorial step by step:
    https://docs.isaacsim.omniverse.nvidia.com/4.5.0/reference_material/sim_performance_optimization_handbook.html

    
    ___________________
    
    Physics Simulation:

    https://docs.omniverse.nvidia.com/extensions/latest/ext_physics/simulation-control/physics-performance.html
    ___________________
    1. Physics Step Size: The physics step size determines the time interval for each physics simulation step. A smaller step size will result in a more accurate simulation but will also require more computational resources and thus slow down the simulation. A larger step size will speed up the simulation but may result in less accurate physics.
    
    2. Minimum Simulation Frame Rate: The minimum simulation frame rate determines the minimum number of physics simulation steps per second. If the actual frame rate drops below this value, the simulation will slow down to maintain the accuracy of the physics.
    
    3. GPU Dynamics: Enabling GPU dynamics can potentially speed up the simulation by offloading the physics calculations to the GPU.
    Note: This will only be beneficial if your GPU is powerful enough and not already fully utilized by other tasks. Enable or disable GPU dynamics in your script using the world.set_gpu_dynamics_enabled(enabled) function, where enabled is a boolean value indicating whether GPU dynamics should be enabled.
    
    ____________________
    
    Scene and Rendering:
    ____________________
    1. Simplify the Scene: Reducing the complexity of the scene, implementing level of detail (LOD), culling invisible objects, and optimizing the physics settings.
        Isaac Sim provides several tools for simplifying your scene
            Scene Optimizer: kit extension that performs scene optimization on the USD level
            Mesh Merge Tool: Isaac Sim utility to merge multiple meshes to a single mesh
    2. rtx:
        real
        - Rtx real time mode: 
            This mode is slightly less accurate than RTX – Interactive (Path Tracing) mode, due to using various shading approximations and optimizations to maintain a high framerate.
            Note: This affects opacity, and user may need to set partial opacity checkbox.
        
        RTX – Interactive (Path Tracing) mode:
            is the most accurate Omniverse RTX Renderer rendering mode and can produce photo-quality images, at the expense of lower framerate than RTX - Real-Time mode.
    
    3. Disable Materials and Lights:
        
        Disabling all materials, set it to -1 to go back to regular.
        Hide lights
        Turn off rendering features in the render settings panel (these will also have equivalent carb settings that can be set in python). There is no non-rtx rendering mode in the Isaac Sim GUI application, but you can disable almost everything (reflections, transparency, etc) to increase execution speed. To disable rendering completely unless explicitly needed by a sensor, you can use the headless application workflow.
    
    4. Adjust DLSS Performance Mode: DLSS performance mode is toggled by the --/rtx/post/dlss/execMode=<value> setting. Values are as follows:
        Adjust DLSS Performance Mode: DLSS performance mode is toggled by the --/rtx/post/dlss/execMode=<value> setting. Values are as follows:
        Performance (0) - the most performant setting, reducing VRAM consumption and rendering time but decreasing render quality. This is the default value in Isaac Sim.
        Balanced (1) - offers both optimized performance and image quality.
        Quality (2) - offers higher image quality than balanced mode, at the cost of increased render time and VRAM consumption.
        Auto (3) - Selects the best DLSS Mode for the current output resolution. When rendering 720p cameras, Auto mode tends to select Quality, so you may see performance impacts by running in Auto mode while rendering cameras at lower resolution.
    
        
    _________________
    
    CPU Thread Count:
    _________________


    """
    # my_world.set_simulation_dt(PHYSICS_STEP_DT, RENDER_DT)
    
    # ___________________
    #
    # Physics Simulation:
    # ___________________

    if physics_step_dt is not None:
        # world.set_physics_step_size(physics_step_dt)
        world.physics_dt = physics_step_dt

    if min_frame_rate is not None:
        # world.set_min_simulation_frame_rate(frame_rate)
        import carb.settings
        settings = carb.settings.get_settings()
        settings.set("/app/frames/minFrameRate", min_frame_rate)  # Or your desired frame rate
    
    if gpu_dynamics_enabled is not None:
       activate_gpu_dynamics(world)
       # world.set_gpu_dynamics_enabled(gpu_dynamics_enabled)

    # ____________________
    #
    # Scene and Rendering:
    # ____________________
    if scene_opt_enabled is not None: # see here, check how to do it in code (they show in gui) https://docs.omniverse.nvidia.com/extensions/latest/ext_scene-optimizer.html
        pass
        # import omni.kit.app
        # import omni.kit.commands
        # from omni.scene.optimizer.core import ExecutionContext
        # from omni.usd import get_context, UsdUtils

        # # 1. Enable the bundle (loads core + UI)
        # ext_mgr = omni.kit.app.get_app().get_extension_manager()
        # ext_mgr.set_extension_enabled_immediate("omni.scene.optimizer.bundle", True)

        # # 2. Prepare your stage and context
        # # stage = get_context().get_stage()
        # stage = world.stage
        # ctx = ExecutionContext()
        # ctx.usdStageId = UsdUtils.StageCache.Get().Insert(stage).ToLongInt()
        # ctx.generateReport = True
        # ctx.captureStats = True

        # # 3. Pick an operation and supply arguments from docs
        # op = "decimateMeshes"
        # args = {
        #     "paths": [],  # empty = all meshes
        #     "reductionFactor": 50.0,
        #     "maxMeanError": 0.0,
        #     "guideDecimation": 0,
        #     "pinBoundaries": False,
        #     "cpuVertexCountThreshold": 100000,
        #     "gpuVertexCountThreshold": 500000
        # }

        # # 4. Execute
        # omni.kit.commands.execute(
        #     "SceneOptimizerOperation",
        #     operation=op,
        #     args=args,
        #     context=ctx
        # )
        # import omni.kit
        # omni.kit.set_extension_enabled("omni.scene_optimizer", True)
        # import omni.kit.app
        # ext_manager = omni.kit.app.get_app().get_extension_manager()
        # ext_manager.set_extension_enabled_immediate("omni.scene_optimizer", True)
        # from omni.scene_optimizer.core import SceneOptimizer
        # # Initialize optimizer with default settings
        # print("optimizing scene...")
        # optimizer = SceneOptimizer()
        # optimizer.optimize_stage(world.stage)
        # # Could also use custom settings
        # # # config = {
        # #     "flatten_transforms": True,
        # #     "merge_meshes": True,
        # #     "simplify_geometry": False,
        # #     # Add other supported options...
        # # }
        # # optimizer = SceneOptimizer(config=config)
        # # optimizer.optimize_stage(stage)
    
    if merge_mesh_tool_enabled is not None:
        pass
        # TODO: see here, check how to do it in code, (also make sure not breaking action graphs etc) https://docs.isaacsim.omniverse.nvidia.com/4.5.0/robot_setup/ext_isaacsim_util_merge_mesh.html#isaac-merge-mesh

    if rtx_mode is not None:
        if rtx_mode == 'real_time':    
            pass
            # stage = world.stage 
            # # Explicitly set RTX renderer to real-time
            # world.renderer.set_input("renderer", "RTX")
            # world.renderer.set_input("renderingMode", "rt")  # 'rt' means real time
            # world.renderer.set_input("enableConnection", True)  # apply changes
            # world.renderer.update()

            # # Now apply:
            # stage.GetPrimAtPath("/OmniverseKit_PxrSettings").GetAttribute("rtSettings.enable").Set(True)

            # Alternatively, if you prefer path tracing fallback:
            # world.renderer.set_input("renderer", "PathTracing")
            # world.renderer.set_input("renderingMode", "rt")
            # world.renderer.update()
            
            pass # TODO: see here how to tune: https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_rt.html#rtx-rt-mode
            
        elif rtx_mode == 'path_tracing':
            # NOTE: This is not yet tested
            # Set renderer to Path Tracing
            world.renderer.set_input("renderer", "PathTracing")
            world.renderer.set_input("renderingMode", "pt")  # 'pt' for path tracing
            world.renderer.set_input("enableConnection", True)
            world.renderer.update()

            # Optional: If you want higher-quality output:
            world.renderer.set_input("maxSamples", 64)      # Increase for better image quality
            world.renderer.set_input("maxBounces", 4)       # Light bounce depth
            world.renderer.set_input("denoiserEnable", True)
            world.renderer.update()

    if diable_materials_and_lights is not None:
        if diable_materials_and_lights: 
            import carb
            carb.settings.get_settings().set_int("/rtx/debugMaterialType", 0) # 0 = Disabling all materials. Set it to -1 to go back to regular.
    
    if dlss_mode is not None: # TODO see here: https://docs.isaacsim.omniverse.nvidia.com/4.5.0/reference_material/sim_performance_optimization_handbook.html
        assert type(dlss_mode) == int and 0 <= dlss_mode <=3

    # _________________
    #
    # CPU Thread Count:
    # _________________
    #  TODO: this should be done in the app level... 

def get_performance_mode_prebuilt_setup(mode='economy'):    
    """ options: 
            economy: the "budget" option: lite physics, rendering and app resources usage.
            balanced: between economy and accuracy. 
            best: the "accuracy" option: heavy physics, rendering and app resources usage 
    """
    
    if mode == 'economy':
        setup = {
            'physics_step_dt': 1/30, 
            'min_frame_rate': 30, 
            'gpu_dynamics_enabled': True, 
            'scene_opt_enabled': True,
            'merge_mesh_tool_enabled': True,
            'rtx_mode': 'real_time',
            'diable_materials_and_lights': True,
            'dlss_mode': 0 

        }
    
    elif mode == 'balanced':
        setup = {
            'physics_step_dt': 1/30, 
            'min_frame_rate': 30, 
            'gpu_dynamics_enabled': True, 
            'scene_opt_enabled': True,
            'merge_mesh_tool_enabled': True,
            'rtx_mode': 'real_time',
            'diable_materials_and_lights': False,
            'dlss_mode': 0 # 

        }
    elif mode == 'best':

        pass # TODO
   
    return setup

  


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










