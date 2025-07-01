import numpy as np
import torch

# Third-party (from CuRobo)
from curobo.geom.types import WorldConfig, Cuboid
from projects_root.utils.world_model_wrapper import WorldModelWrapper


class _DummyStageObstacles:
    """Light-weight stand-in that mimics the return value of
    UsdHelper.get_obstacles_from_stage()."""

    def __init__(self, world_cfg: WorldConfig):
        self._cfg = world_cfg
        self.objects = []
        if world_cfg.cuboid:
            self.objects += world_cfg.cuboid
        if world_cfg.mesh:
            self.objects += world_cfg.mesh

    def get_collision_check_world(self):
        return self._cfg


class _DummyUsdHelper:
    """Very small stub that lets us supply a predefined world configuration."""

    def __init__(self, world_cfg: WorldConfig):
        self._cfg = world_cfg

    def get_obstacles_from_stage(self, *args, **kwargs):  # noqa: D401,E501
        return _DummyStageObstacles(self._cfg)


class _DummyCollisionChecker:
    """Stub that provides the minimal interface used by WorldModelWrapper."""

    def __init__(self):
        self.updated_names = []

    # pylint: disable=unused-argument
    def update_obstacle_pose(self, name, w_obj_pose, env_idx=0):  # noqa: D401,E501
        self.updated_names.append(name)

    def load_collision_model(self, world_model):  # noqa: D401,E501
        # We don't need to do anything here for the test
        pass


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_initialisation_and_name_extraction():
    # Build a simple world with one cuboid
    cube = Cuboid(name="cube_1", pose=[0, 0, 0, 1, 0, 0, 0], dims=[1, 1, 1])
    world_cfg = WorldConfig(cuboid=[cube])

    usd_helper = _DummyUsdHelper(world_cfg)

    wrapper = WorldModelWrapper(world_config=WorldConfig(), base_frame=np.zeros(7))
    wrapper.initialize_from_stage(usd_helper)  # type: ignore[arg-type]

    assert wrapper.is_initialized()
    assert "cube_1" in wrapper.get_obstacle_names()


def test_pose_transform_identity():
    """With identity base frame, the pose should stay the same."""
    wrapper = WorldModelWrapper(world_config=WorldConfig())
    pose = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])
    transformed = wrapper._transform_pose_world_to_base(pose)
    assert np.allclose(pose, transformed)


def test_update_calls_collision_checker():
    cube = Cuboid(name="updatable", pose=[0, 0, 0, 1, 0, 0, 0], dims=[1, 1, 1])
    world_cfg = WorldConfig(cuboid=[cube])

    usd_helper = _DummyUsdHelper(world_cfg)

    wrapper = WorldModelWrapper(world_config=WorldConfig())
    wrapper.initialize_from_stage(usd_helper)  # type: ignore[arg-type]

    dummy_checker = _DummyCollisionChecker()
    wrapper.set_collision_checker(dummy_checker)

    # Move the cube in the dummy stage and call update
    cube.pose = [0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    wrapper.update(usd_helper)  # type: ignore[arg-type]

    assert "updatable" in dummy_checker.updated_names 