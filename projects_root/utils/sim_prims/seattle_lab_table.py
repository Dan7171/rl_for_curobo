"""
This is the non instancble version, meaning its useless for multi environment usage.
To read about instanceable assets, see: https://docs.isaacsim.omniverse.nvidia.com/latest/isaac_lab_tutorials/tutorial_instanceable_assets.html
"""
import abc
from projects_root.utils.sim_prims.Iprim import Iprim
from projects_root.utils.sim_prims.asset_utils.utils import load_asset_to_prim_path
from isaacsim.core.prims  import XFormPrim, SingleXFormPrim, SingleGeometryPrim, GeometryPrim
import numpy as np
from pxr import UsdGeom, Gf, PhysxSchema, UsdPhysics, Sdf


def disable_collision_recursive(root_prim):
 
    # root_prim = stage.GetPrimAtPath(prim_path)
    
    def disable_collision(prim_path):
        # if isinstance(prim, GeometryPrim):
        try:
            # prim_path = prim.prim_paths[0]
            geom = GeometryPrim(prim_path)# SingleGeometryPrim(prim.GetPath())
            geom.disable_collision()
            print(f"Collision disabled for: {prim_path}")
        except Exception as e:
            print(f"Skipped {prim_path} ({e})")
        
        usd_prim = geom.prims[0]
        children = usd_prim.GetChildren()
        children_paths = [child.GetPath().pathString for child in children]
        for child_path in children_paths:
            disable_collision(child_path)

    disable_collision(root_prim)
    
class SeattleLabTable(Iprim):
    
    @classmethod
    def get_path_in_asset_browser(cls):
        return "Props/Mounts/SeattleLabTable/table.usd"
    
    def __init__(self, 
            stage,     
            path:str='',
            position:list[float]=[0,0,0],
            rotation:list[float]=[1,0,0,0],
            collision:bool=True,
            gravity:bool=False,
            collision_approximation:str='meshSimplification',
            scale:list[float]=[1.0, 1.0, 1.0],
        ):
        self._path = self.spawn_prim(path)
        self._xform_prim = SingleXFormPrim(self.get_path())
        self._geometry_prim = GeometryPrim(self.get_path())# SingleGeometryPrim(self.get_path())
        self.set_pose(np.array(position), np.array(rotation))
        # self._xformable = UsdGeom.Xformable(self.get_prim(stage)) # https://openusd.org/dev/api/class_usd_geom_xformable.html
        # self._xform_api = UsdGeom.XformCommonAPI(self.get_prim(stage))
        # self.set_position(Gf.Vec3d(*position))
        # self.set_rotation(Gf.Vec3f(*rotation))
        # if collision:
        #     self.enable_collision(collision_approximation)
        if not collision:
            disable_collision_recursive(self.get_path())
        if not gravity:
            self.disable_gravity(stage)

        if scale != [1.0, 1.0, 1.0]:
            self.set_scale(scale)
    def spawn_prim(self, prim_path):
        return load_asset_to_prim_path(self.get_path_in_asset_browser(), prim_path)
   
    
    def get_prim(self,stage):
        # return stage.GetPrimAtPath(self.get_path(stage))
        return self._xform_prim
    
    def get_path(self,*args,**kwargs):
        return self._path

    def set_pose(self, position, rotation):
        # https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/isaacsim.core.prims/docs/index.html#isaacsim.core.prims.SingleXFormPrim
        self._xform_prim.set_world_pose(position, rotation)
    
    def set_position(self,position):
        # SEE set_pose
        # self._xform_prim.set_world_poses(position)
        # self._get_xform().SetTranslate(position)
        pass
    def set_rotation(self,rotation):
        # SEE set_pose
        # self._get_xform().SetRotate(rotation) 
        pass


    
    def enable_collision(self, collision_approximation):
        geom = self._geometry_prim
        geom.enable_collision()
        geom.set_collision_approximations([collision_approximation])
     
        
    def disable_collision(self):
        geom = self._geometry_prim
        geom.disable_collision()
        # geom.set_collision_enabled(False) # enable_collision(stage, collision_approximation)
        # self._geometry_prim.set_collision_approximation(None)
        
    def enable_gravity(self, stage):
        pass # TODO: implement
        
    def disable_gravity(self, stage):
        # prim = self.get_prim(stage)
        # rigid_api = UsdPhysics.RigidBodyAPI.Apply(prim)
        # rigid_api.CreateKinematicEnabledAttr().Set(True)                
        pass
    def get_pose(self):
        return self._xform_prim.get_world_poses()
    
    def get_velocity(self):
        return self._get_xform().GetTranslate()
    
    def set_velocity(self, velocity, stage):
        pass # TODO: implement

    def set_scale(self, scale=[1.0, 1.0, 1.0]):
        num_envs = 1 # kept this here to see how that can be done for more envs 
        scale = np.tile(np.array(scale), (num_envs, 1))
        xform_prim = XFormPrim(self.get_path())
        xform_prim.set_local_scales(scale)
    
    
    