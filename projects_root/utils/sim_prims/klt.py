import abc
from projects_root.utils.sim_prims.Iprim import Iprim
from projects_root.utils.sim_prims.asset_utils.utils import load_asset_to_prim_path
import numpy as np
from isaacsim.core.prims  import XFormPrim, SingleXFormPrim, SingleGeometryPrim, GeometryPrim
from pxr import UsdGeom, Gf, PhysxSchema, UsdPhysics, Sdf

class Klt(Iprim):

    @classmethod
    def get_path_in_asset_browser(cls):
        return "Props/KLT_Bin/small_KLT_visual.usd"
    
    def __init__(self, 
            stage,     
            path:str='',
            position:list[float]=[0,0,0],
            rotation:list[float]=[0,0,0],
            collision:bool=True,
            gravity:bool=False,
            collision_approximation:str='meshSimplification',
            scale:list[float]=[1.0, 1.0, 1.0],
        ):
        self._path = self.spawn_prim(path)
        self._xform = UsdGeom.XformCommonAPI(self.get_prim(stage))
        self.set_position(Gf.Vec3d(*position))
        self.set_rotation(Gf.Vec3f(*rotation))
        if collision:
            self.enable_collision(stage, collision_approximation)
        if not gravity:
            self.disable_gravity(stage)
        if scale != [1.0, 1.0, 1.0]:
            self.set_scale(scale)


    def spawn_prim(self, prim_path):
        return load_asset_to_prim_path(self.get_path_in_asset_browser(), prim_path)
   
 
    def get_prim(self,stage):
        return stage.GetPrimAtPath(self.get_path(stage))
    
    def get_path(self,*args,**kwargs):
        return self._path

    def set_position(self,position):
        self._get_xform().SetTranslate(position)
    
    def set_rotation(self,rotation):
        self._get_xform().SetRotate(rotation) 


    def _get_xform(self):
        return self._xform
    
    def enable_collision(self, stage, collision_approximation):
        prim = self.get_prim(stage)
        UsdPhysics.CollisionAPI.Apply(prim)
        PhysxSchema.PhysxCollisionAPI.Apply(prim)
        # Set the approximation type (using token type)
        attr = prim .CreateAttribute("physxCollision:approximation", Sdf.ValueTypeNames.Token)
        attr.Set(collision_approximation)
        
    def disable_collision(self, stage):
        pass # TODO: implement
        
    def enable_gravity(self, stage):
        pass # TODO: implement
        
    def disable_gravity(self, stage):
        prim = self.get_prim(stage)
        rigid_api = UsdPhysics.RigidBodyAPI.Apply(prim)
        rigid_api.CreateKinematicEnabledAttr().Set(True)                

    def get_pose(self):
        return self._get_xform().GetTranslate()
    
    def get_velocity(self):
        return self._get_xform().GetTranslate()
    
    def set_velocity(self, velocity, stage):
        pass # TODO: implement
    
    
    def set_scale(self, scale=[1.0, 1.0, 1.0]):
        num_envs = 1 # kept this here to see how that can be done for more envs 
        scale = np.tile(np.array(scale), (num_envs, 1))
        XFormPrim(self.get_path()).set_local_scales(scale)
