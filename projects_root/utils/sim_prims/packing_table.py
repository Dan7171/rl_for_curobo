        
"""
Currently curobo cant read this because meshes are not triangulated!!!!!!

This is the non instancble version, meaning its useless for multi environment usage.
To read about instanceable assets, see: https://docs.isaacsim.omniverse.nvidia.com/latest/isaac_lab_tutorials/tutorial_instanceable_assets.html

"""
import abc
from projects_root.utils.sim_prims.Iprim import Iprim
from projects_root.utils.sim_prims.asset_utils.utils import load_asset_to_prim_path
from isaacsim.core.prims  import XFormPrim, SingleXFormPrim, SingleGeometryPrim, GeometryPrim
import numpy as np
from pxr import UsdGeom, Gf, PhysxSchema, UsdPhysics, Sdf, UsdUtils
from projects_root.utils.quaternion import isaacsim_euler2quat
import trimesh



# def force_triangulate_mesh(stage, prim_path):
#     prim = stage.GetPrimAtPath(prim_path)
#     mesh = UsdGeom.Mesh(prim)
#     if not mesh:
#         print(f"No mesh found at {prim_path}")
#         return

#     faces = mesh.GetFaceVertexIndicesAttr().Get()
#     counts = mesh.GetFaceVertexCountsAttr().Get()

#     new_faces = []
#     new_counts = []

#     index = 0
#     while index < len(faces):
#         count = counts[index // count if isinstance(counts[0], list) else index]
#         face = faces[index:index + count]

#         if count == 3:
#             new_faces.extend(face)
#             new_counts.append(3)
#         elif count == 4:
#             # Convert quad [0,1,2,3] -> [0,1,2] and [0,2,3]
#             new_faces.extend([face[0], face[1], face[2]])
#             new_faces.extend([face[0], face[2], face[3]])
#             new_counts.extend([3, 3])
#         else:
#             print(f"Skipping face with {count} vertices")
        
#         index += count

#     # Apply back to mesh
#     mesh.GetFaceVertexIndicesAttr().Set(new_faces)
#     mesh.GetFaceVertexCountsAttr().Set(new_counts)

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


# def triangulate_mesh_prim(stage, mesh_prim):
#     """
#     Triangulates the mesh at prim_path on the given stage.
#     Returns the new triangulated mesh prim.
#     THIS IS A WORKAROUND TO PREVERNT MESH LOADING FROM FAILING IN CUROBO  FUNCTION get_mesh_attrs() which requires triangulare meshes otherwise returns None (fails to read them)
#     """
#     #prim = stage.GetPrimAtPath(prim_path)
    
#     assert mesh_prim.IsA(UsdGeom.Mesh)
#     if not mesh_prim.IsValid():
#         print(f"Prim at {mesh_prim} is not valid.")
#         return None

#     # Triangulate mesh
#     UsdUtils.MeshUtil.Triangulate(mesh_prim)

#     return mesh_prim

# def triangulate_usd_mesh(points, face_vertex_indices, face_vertex_counts):
#     mesh = trimesh.Trimesh(
#         vertices=points,
#         faces=trimesh.util.unflatten(face_vertex_indices, face_vertex_counts),
#         process=False  # avoid automatic cleanup
#     )
#     mesh_triangulated = mesh.triangulate()

#     return mesh_triangulated.vertices, mesh_triangulated.faces

# def triangulate_mesh_prims_recursive(stage, prim_path):
#     """
#     Triangulates the mesh at prim_path on the given stage.
#     Returns the new triangulated mesh prim.
#     """

#     try:
#         prim = stage.GetPrimAtPath(prim_path)
#         if prim.IsA(UsdGeom.Mesh):
#             # triangulate_mesh_prim(stage, prim)
#             # force_triangulate_mesh(stage, prim_path)
            
#             print(f"Triangulated mesh at {prim_path}")
        
#         for child in prim.GetChildren():
#             triangulate_mesh_prims_recursive(stage, child.GetPath())

#     except Exception as e:
#         print(f"Skipped {prim_path} ({e})")

#     # triangulate_mesh_prim(stage, prim_path)

class PackingTable(Iprim):
   
        
    
    @classmethod
    def get_path_in_asset_browser(cls):
        return "/Props/PackingTable/props/SM_HeavyDutyPackingTable_C02_01/SM_HeavyDutyPackingTable_C02_01.usd"
    
    def __init__(self, 
            stage,     
            path:str='/World/heavy_duty_packing_table',
            position:list[float]=[0,0,0],
            rotation:list[float]=[0,0,0],
            collision:bool=True,
            gravity:bool=False,
            collision_approximation:str='meshSimplification',
            scale:list[float]=[1.0, 1.0, 1.0],
            # triangulate:bool=True,
        ):
        self._path = self.spawn_prim(path)
        self._xform_prim = SingleXFormPrim(self.get_path())
        self._geometry_prim = GeometryPrim(self.get_path())# SingleGeometryPrim(self.get_path())
        self.set_pose(np.array(position), np.array(rotation))
 
        if collision:
            self.enable_collision(collision_approximation)
        else:
            disable_collision_recursive(self.get_path())
        if not gravity:
            self.disable_gravity(stage)

        if scale != [1.0, 1.0, 1.0]:
            self.set_scale(scale)

        # if triangulate: 
        #     triangulate_mesh_prims_recursive(stage, self.get_path()) # this is a workaround to prevent curobo from failing to read the USD mesh their Mesh object world models
    
    def spawn_prim(self, prim_path):
        return load_asset_to_prim_path('/home/dan/Desktop/table_triangualer_blender/Untitled.usd', prim_path, is_fullpath=True)
   
    
    def get_prim(self,stage):
        # return stage.GetPrimAtPath(self.get_path(stage))
        return self._xform_prim
    
    def get_path(self,*args,**kwargs):
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
    
    
    