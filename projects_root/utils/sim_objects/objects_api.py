"""
See: https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/isaacsim.core.api/docs/index.html#objects
"""


from typing_extensions import Union
from isaacsim.core.api.objects import \
    DynamicCapsule, \
    DynamicCone, \
    DynamicCuboid, \
    DynamicCylinder, \
    DynamicSphere, \
    GroundPlane, \
    VisualCapsule, \
    VisualCone, \
    VisualCuboid, \
    VisualCylinder, \
    VisualSphere, \
    FixedCapsule, \
    FixedCone, \
    FixedCuboid, \
    FixedCylinder, \
    FixedSphere


class DynamicObject:
    def __init__(self, object:Union[DynamicCapsule,DynamicCone,DynamicCuboid,DynamicCylinder,DynamicSphere]):
        self.object = object

    def get_object(self):
        return self.object
    
    
    