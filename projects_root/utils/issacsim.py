import time

def init_app(app_settings:dict={"headless": False}):
    """
    Steps that must be done before importing other isaac sim modules.
    https://medium.com/@kabilankb2003/isaac-sim-core-api-for-robot-control-a-hands-on-guide-f9b27f5729ab
    https://forums.developer.nvidia.com/t/cannot-import-omni-isaac-core/242977/3
    """
    try:
        import isaacsim
    except ImportError:
        pass
    from omni.isaac.kit import SimulationApp 
    
    app = SimulationApp(app_settings)
 
    return app



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