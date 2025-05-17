 
import abc
from projects_root.utils.sim_prims.Iprim import Iprim
from projects_root.utils.sim_prims.asset_utils.utils import load_asset_to_prim_path
from isaacsim.core.prims  import XFormPrim, SingleXFormPrim, SingleGeometryPrim, GeometryPrim
from isaacsim.core.api.materials import VisualMaterial
from projects_root.utils.sim_materials.visual_materials.utils import create_material
from projects_root.utils.quaternion import isaacsim_euler2quat
from curobo.util.usd_helper import UsdHelper
import numpy as np
import os
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
    
class SimpleTable(Iprim):
    
    @classmethod
    def get_usd_file_path(cls):
        return os.path.abspath("projects_root/utils/sim_prims/self_made/usd/ simple_table_only_centered.usd")
    
    def __init__(self, 
            stage,     
            path:str='/World/simple_table',
            position:list[float]=[0,0,0],
            rotation:list[float]=[0,0,0],
            collision:bool=True,
            gravity:bool=False,
            collision_approximation:str='meshSimplification',
            scale:list[float]=[1.0, 1.0, 1.0],
        ):
        self._path = self.spawn_prim(path)
        self._xform_prim = SingleXFormPrim(self.get_path())
        self._geometry_prim = GeometryPrim(self.get_path())# SingleGeometryPrim(self.get_path())
        self.set_pose(np.array(position), np.array(rotation))

        if collision:
            self.enable_collision(collision_approximation)
        if not collision:
            disable_collision_recursive(self.get_path())
        if not gravity:
            self.disable_gravity(stage)

        if scale != [1.0, 1.0, 1.0]:
            self.set_scale(scale)

    def spawn_prim(self, prim_path):
        return load_asset_to_prim_path(self.get_usd_file_path(), prim_path, is_fullpath=True)
   
    
    def get_prim(self):
        return self._xform_prim
    
    def get_path(self):
        return self._path

    def set_pose(self, position, rotation):
        # https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/isaacsim.core.prims/docs/index.html#isaacsim.core.prims.SingleXFormPrim
        rotation = isaacsim_euler2quat(*rotation)
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
    
    def set_visual_material(self,material:VisualMaterial):
        """
        https://docs.omniverse.nvidia.com/materials-and-rendering/latest/templates/parameters/OmniPBR_Albedo.html
        https://docs.omniverse.nvidia.com/simready/latest/simready-asset-creation/material-best-practices.html
        https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/isaacsim.core.api/docs/index.html#isaacsim.core.api.materials.OmniPBR

        """
        # mat_path = join_path(object_path, material_name)
        # material_usd = UsdShade.Material.Define(self.stage, mat_path)
        # pbrShader = UsdShade.Shader.Define(self.stage, join_path(mat_path, "PbrShader"))
        # pbrShader.CreateIdAttr("UsdPreviewSurface")
        # pbrShader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(material.roughness)
        # pbrShader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(material.metallic)
        # pbrShader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(color[:3]))
        # pbrShader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(color[:3]))
        # pbrShader.CreateInput("baseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(color[:3]))
        
    def apply_visual_material(self, material_path:str, apply_to_descendts:bool=True):
        """
        https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/isaacsim.core.prims/docs/index.html#isaacsim.core.prims.XFormPrim.apply_visual_materials
        https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/isaacsim.core.api/docs/index.html#materials
        """
        
        xform_prim = XFormPrim(self.get_path())
        weaker_than_descendts = not apply_to_descendts
        xform_prim.apply_visual_materials(material_path, weaker_than_descendts) # can be also done for multiple materials, for multi environment https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/isaacsim.core.prims/docs/index.html#isaacsim.core.prims.XFormPrim.apply_visual_materials:~:text=.post_reset()-,apply_visual_materials,-(
        # self._geometry_prim.set_visual_material(material)