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
    return SimulationApp(app_settings)