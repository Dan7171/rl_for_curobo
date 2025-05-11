try:
    # Third Party
    import isaacsim
except ImportError:
    pass

# Third Party
from typing import Iterable, List, Optional, Union
from matplotlib.pyplot import bone
import torch
import numpy as np
import os
# Initialize the simulation app first (must be before "from omni.isaac.core")

# from omni.isaac.kit import SimulationApp  
# simulation_app = SimulationApp({"headless": False})

# Now import other Isaac Sim modules
# https://medium.com/@kabilankb2003/isaac-sim-core-api-for-robot-control-a-hands-on-guide-f9b27f5729ab
from isaacsim.core.api.objects import DynamicCuboid, DynamicSphere, DynamicCapsule, DynamicCylinder, DynamicCone, GroundPlane
from isaacsim.core.api.materials.physics_material import PhysicsMaterial
from omni.isaac.core import World # https://forums.developer.nvidia.com/t/cannot-import-omni-isaac-core/242977/3
from omni.isaac.core.prims import XFormPrim
from pxr import PhysxSchema, UsdPhysics
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
import copy
# Import helper from curobo examples

# CuRobo
from curobo.geom.types import Capsule, Cuboid, Cylinder, Mesh, Sphere, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose 
from projects_root.projects.dynamic_obs.dynamic_obs_predictor.frame_utils import FrameUtils
# from curobo.util.usd_helper import UsdHelper
from curobo.util_file import get_assets_path, join_path

# Initialize CUDA device
a = torch.zeros(4, device="cuda:0") 

def get_full_path_to_asset(asset_subpath):
    return get_assets_root_path() + '/Isaac/' + asset_subpath

# def read_world_model_from_usd(file_path: str,obstacle_path="/world/obstacles",reference_prim_path="/world",usd_helper=None):
#     """
#     This function reads the world model from a USD file.
#     It aims to read the world model (for static obstacles) from a USD file and return a list of cuboids and spheres.
#     Obstacles are expected to be under the prim path /world/obstacles.

#     NOTE: 
#     Was taken from https://curobo.org/notes/05_usd_api.html
#     Origin in of read_world_from_usd see: /curobo/examples/usd_example.py
#     """        
#     # usd_helper = UsdHelper() # experimental
#     usd_helper.load_stage_from_file(file_path)
#     world_model = usd_helper.get_obstacles_from_stage()
#     return world_model 



def load_asset_to_prim_path(asset_subpath, prim_path='', is_fullpath=False):
    """
    Loads an asset to a prim path.
    Source: https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.replicator.isaac/docs/index.html?highlight=add_reference_to_stage

    asset_subpath: sub-path to the asset to load. Must end with .usd or .usda. Normally starts with /Isaac/...
    To browse, go to: asset browser in simulator and add /Issac/% your subpath% where %your subpath% is the path to the asset you want to load.
    Note: to get the asset exact asset_subpath, In the simulator, open: Isaac Assets -> brows to the asset (usd file) -> right click -> copy url path and paste it here (the subpath is the part after the last /Isaac/).
    Normally the assets are coming from web, but this tutorial can help you use local assets: https://docs.omniverse.nvidia.com/launcher/latest/it-managed-launcher/content_install.html.

    For example: http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.0/Isaac/Props/Mugs/SM_Mug_A2.usd -> asset_subpath should be: Props/Mugs/SM_Mug_A2.usd
        
    prim_path: path to the prim to load the asset to. If not provided, the asset will be loaded to the prim path /World/%asset_subpath%    
    is_fullpath: if True, the asset_subpath is a full path to the asset. If False, the asset_subpath is a subpath to the assets folder in the simulator.
    This is useful if you want to load an asset that is not in the {get_assets_root_path() + '/Isaac/'} folder (which is the root folder for Isaac Sim assets (see asset browser in simulator) but custom assets in your project from a local path.



    Examples:
    load_asset_to_prim_path("Props/Mugs/SM_Mug_A2.usd") will load the asset to the prim path /World/Promps_Mugs_SM_Mug_A2
    load_asset_to_prim_path("Props/Mugs/SM_Mug_A2.usd", "/World/Mug") will load the asset to the prim path /World/Mug
    load_asset_to_prim_path("/home/me/some_folder/SM_Mug_A2.usd", "/World/Mug", is_fullpath=True) will load the asset to the prim path /World/Mug
    load_asset_to_prim_path("http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.0/Isaac/Props/KLT_Bin/small_KLT_visual_collision.usd", "/World/KLT_Bin", is_fullpath=True) will load the asset to the prim path /World/KLT_Bin
    """

    # validate asset 
    if not prim_path:
        # prim_name = asset_subpath.split('/')[-1].split('.')[0]
        asset_subpath_as_prim_name = asset_subpath.replace('/', '_').split('.')[0]
        prim_path = f'/World/{asset_subpath_as_prim_name}'
    else:
        prim_path = prim_path
    
    # define full path to asset
    if not is_fullpath:
        asset_fullpath = get_full_path_to_asset(asset_subpath)
    else:
        asset_fullpath = asset_subpath 
    
    # validate asset path
    assert asset_fullpath.endswith('.usd') or asset_fullpath.endswith('.usda'), "Asset path must end with .usd or .usda"
    
    # load asset to prim path (adds the asset to the stage)
    add_reference_to_stage(asset_fullpath, prim_path)
    return prim_path 

class Obstacle:
    def __init__(self, 
                world:World, 
                name:str,
                curobo_type:str, 
                pose:np.ndarray=np.array([1,1,1,1,0,0,0]), 
                dims:Iterable[float]=np.array([1,1,1]),
                color:Iterable[float]=np.array([0,0,0]), 
                mass:float=1.0, 
                linear_velocity:Iterable[float]=np.array([0,0,0]), 
                angular_velocity:Iterable[float]=np.array([0,0,0]), 
                gravity_enabled=True, 
                sim_collision_enabled=True,
                visual_material=None,
                simulation_enabled=True,
                cchecking_enabled=True,
                usd_path=None,
                mesh_file_sub=None,
                dynamic_obs_prediction_enabled=False,
                ): 
        """
        Creates the representations of the obstacle in the simulation form and in the curobo form (making equivalent representations).
        If world model is provided, the curobo representation of the obstacle is injected into the world model of curobo (in the simulation in happens automatically when simulation representation is created).

        See https://curobo.org/get_started/2c_world_collision.html

        Args:
            name (str): object name. Must be unique.
            pose (_type_): initial position of the obstacle in the world frame ([x,y,z, qw, qx, qy, qz]).
            dims (_type_): if cuboid, scalar (side length (m)).
            curobo_type (np.array): [r,g,b]
            color (_type_): [r,g,b,alpha] (alpha is the opacity of the obstacle)
            mass (_type_): kg
            linear_velocity (_type_): vx, vy, vz (m/s)
            angular_velocity (_type_):wx, wy, wz (rad/s)
            gravity_enabled (bool): enable/disable gravity for the obstacle. If False, the obstacle will not be affected by gravity.
            sim_collision_enabled (bool): enable/disable collision for the obstacle in the simulation.
            visual_material (_type_): visual material of the obstacle.
            simulation_enabled (bool): if True, the obstacle will be simulated in the simulation. Else will only support collision checking representation.
            cchecking_enabled (bool): if True, representation of the obstacle in collision is enabled (else its just a regualar simulation primitive).
            usd_path (str): <Relevant to mesh obstacles only!> path to the usd file, described the mesh obstacle. The usd is a blue print for the simulation representation of the obstacle. It will be loaded from the usd.
            mesh_file_sub (str): <Relevant to mesh obstacles only!> subpath to the mesh file (must be a subpath under curobo/src/curobo/content/assets). Used only for mesh obstacles (this is what tells curobo coll obstacle (Mesh) how it's built. Without it, robots won't know the shape of the obstacle).
        
        """
        assert curobo_type in ["cuboid", "sphere", "mesh", "capsule", "cylinder", "cone"]
        self.obs_type_to_sim_type = {"cuboid": DynamicCuboid, "sphere": DynamicSphere, "mesh": XFormPrim, "capsule": DynamicCapsule, "cylinder": DynamicCylinder}
        self.name = name
        self.path = f'/World/curobo_world_cfg_obs_visual_twins/{name}' # path to the obstacle in the simulation
        self.dims: list[float] = list(dims)
        self.curobo_type = curobo_type
        self.sim_type = self.obs_type_to_sim_type[curobo_type]
        self.tensor_args = TensorDeviceType()  # Add this to handle device placement
        self.simulation_enabled = simulation_enabled
        if self.simulation_enabled:
            self.simulation_representation = self._init_obstacle_in_simulation(world, pose[:3], pose[3:], self.dims, self.sim_type, color, mass, gravity_enabled,sim_collision_enabled,visual_material, linear_velocity, angular_velocity, usd_path)
        self.cchecking_enabled = cchecking_enabled
        self.usd_path = usd_path
        self.mesh_file_sub = mesh_file_sub 
        self.registered_world_models = [] # world models that the obstacle is added to
        self.indices_in_registered_world_models = [] # indices of the obstacle in the registered world models
        self.registered_Wmo_transforms = [] # transforms from world to world model origin for all registered world models
        self.registered_ccheckers = [] # all collision checkers that the obstacle is added to and their transforms       
        self.dynamic_obs_prediction_enabled = dynamic_obs_prediction_enabled
    
    # def update_dims(self, dims):
    #     self.dims = dims
    #     if self.simulation_enabled:
    #         self.simulation_representation.set_size(dims)
    #     if self.cchecking_enabled:
    #         self.curobo_obstacle.dims = dims
            
    def set_simulation_refernce(self, simulation_refernce):
        self.simulation_refernce = simulation_refernce
    
    def _init_obstacle_in_simulation(self, world, position, orientation, dims, sim_type, color,  mass, gravity_enabled,sim_collision_enabled,visual_material, linear_velocity, angular_velocity, usd_path):
        """
        Create a moving obstacle in the simulation.
        
        Args:
            world: Isaac Sim world instance
            position: Initial position [x, y, z]
            size: Size of obstacle (diameter for sphere, side length for cube)
            color: RGB color array (defaults to blue if None)
                        If False, creates a visual-only obstacle that moves without physics.
            mass: Mass in kg
            gravity_enabled: If False, disables gravity for the obstacle 
        """
        linear_velocity = np.array(linear_velocity)
        angular_velocity = np.array(angular_velocity)
        color = np.array(color)
        if sim_type == DynamicCuboid:
            obstacle = self._init_DynamicCuboid_for_simulation(world, position, orientation, dims, color,  mass, gravity_enabled,sim_collision_enabled,visual_material, linear_velocity, angular_velocity)
        elif sim_type == DynamicSphere:
            obstacle = self._init_DynamicSphere_for_simulation(world, position, orientation, dims, color,  mass, gravity_enabled,sim_collision_enabled,visual_material, linear_velocity, angular_velocity)
        elif sim_type == XFormPrim:
            obstacle = self._init_XFormPrim_for_simulation(world, position, orientation, dims, color,  mass, gravity_enabled,sim_collision_enabled,visual_material, linear_velocity, angular_velocity, usd_path)
        
        return obstacle
    
    def _specify_prim_attributes(self, world, prim, material, linear_velocity, angular_velocity, gravity_enabled, sim_collision_enabled):
        
        if material is None:
            material = PhysicsMaterial( # https://www.youtube.com/watch?v=tHOM-OCnBLE
                prim_path=self.path+"/aluminum",  # path to the material prim to create
                dynamic_friction=0,
                static_friction=0,
                restitution=0
            )

        prim.apply_physics_material(material)
        if not np.array_equal(linear_velocity, np.array([0,0,0])):
            prim.set_linear_velocity(linear_velocity)
        if not np.array_equal(angular_velocity, np.array([0,0,0])):
            prim.set_angular_velocity(angular_velocity)
        if not gravity_enabled:
            self.disable_gravity(self.path, world.stage)

        if not sim_collision_enabled: # Disable collision for the obstacle in the simulation
            prim.set_collision_enabled(False)
        world.scene.add(prim)
    
    def _init_DynamicSphere_for_simulation(self,world, position, orientation, dims, color,  mass, gravity_enabled,sim_collision_enabled,visual_material, linear_velocity, angular_velocity):
        """
        Initialize a sphere obstacle.
        """

        prim = DynamicSphere(prim_path=self.path,name=self.name, position=position,orientation=orientation,size=1,color=color,mass=mass,density=0.9,visual_material=visual_material, scale=dims)         
        self._specify_prim_attributes(world, prim, None, linear_velocity, angular_velocity, gravity_enabled, sim_collision_enabled)
        return prim
        
    
    def _init_DynamicCuboid_for_simulation(self,world, position, orientation, dims, color,  mass, gravity_enabled,sim_collision_enabled,visual_material, linear_velocity, angular_velocity):
        """
        Initialize a cube obstacle.
        
        Args:
            world: Isaac Sim world instance
            position: Initial position [x, y, z]
            size: Side length of cube
            color: RGB color array
                        If False, creates a visual-only obstacle that moves without physics.
            mass: Mass in kg  
            gravity_enabled: If False, disables gravity for the obstacle
        """
        prim = DynamicCuboid(prim_path=self.path,name=self.name, position=position,orientation=orientation,size=1,color=color,mass=mass,density=0.9,visual_material=visual_material, scale=dims)         
        self._specify_prim_attributes(world, prim, None, linear_velocity, angular_velocity, gravity_enabled, sim_collision_enabled)        
        return prim

    
    def _init_XFormPrim_for_simulation(self,world, position, orientation, dims, color,  mass, gravity_enabled,sim_collision_enabled,visual_material, linear_velocity, angular_velocity, usd_path):
        """
        Initialize a mesh obstacle.
        """
        
        # path = self.path + self.name
        # prim = XFormPrim(prim_path=self.path,name=self.name, position=position,orientation=orientation,scale=dims)
        # world.scene.add(prim)
        # return prim
        load_asset_to_prim_path(usd_path, self.path, is_fullpath=True)
        # prim = PrimWrapper(self.path)
        prim = XFormPrim(prim_path=self.path,translation=position,orientation=orientation,scale=dims) # , name = self.name)
        prim.initialize() # the prim so it can be used as a normal prim # ⚠️ required before any pose ops
        # prim.set_world_pose(position, orientation)
        # prim.set_local_scale(dims)
        world.scene.add(prim)
        world.step(render=True)

        return prim
    
    def update_world_coll_checker_with_sim_pose(self, world_coll_checker):
        """ 
        SYNCHRONIZE THE COLLISION CHECKER INFORMATION ABOUT THE OBSTACLE TO BE AS THE  SIMULATION INFORMATION!

        This function aims to update (sync) the curobo representation of the obstacle in its collision checker with the simulation representation of the obstacle.
        In general: they are not synced (the simulation is the actual representation and physics in scene but if we won't actively propagate 
        the information from simulation about the obstacle to the collision checker, the collision checker will not know about the actual physics of the obstacle. That's exactly what this function does!),

        This might be expensive (not sure, but probably), so we probablyshould not do it every time step.
        In ideal condition, it shoukld be called after every world.step() in the simulator (which can change the obstacle in simulator and possibly "break" the synchronization of collision checker which stays behind the simulator and therefore needs an update).
        
        Args:
            world_coll_checker (_type_): _description_
        """
        # get the updated pose of the obstacle in the simulation
        position_isaac_dynamic_prim, orient_isaac_dynamic_prim = self.simulation_representation.get_world_pose() 
        # update the current position of the obstacle
        self.cur_pos = position_isaac_dynamic_prim
        # update the curobo representation of the obstacle in its collision checker
        self._update_curobo_obstacle_pose(world_coll_checker, position_isaac_dynamic_prim, orient_isaac_dynamic_prim)
        

        

    def _update_curobo_obstacle_pose(self,world_coll_checker, position_isaac_dynamic_prim, orient_isaac_dynamic_prim):
        """

        Args:
            isaac_dynamic_prim (bool): _description_
            prim_name (_type_): _description_

        Returns:
            _type_: _description_
        """
 
        # convert isaac simulation pose representation to curobo pose representation
        pos_tensor = self.tensor_args.to_device(torch.from_numpy(position_isaac_dynamic_prim))
        rot_tensor = self.tensor_args.to_device(torch.from_numpy(orient_isaac_dynamic_prim))
        w_obj_pose = Pose(pos_tensor, rot_tensor)
        # update the obstacle pose in the curobo collision checker
        world_coll_checker.update_obstacle_pose(self.name, w_obj_pose)

        # self.world_ccheck.update_obstacle_pose_in_world_model(self.name, w_obj_pose)

    def disable_gravity(self,prim_stage_path,stage):
        """
        Disabling gravity for a given object using Python API, Without disabling physics.
        source: https://forums.developer.nvidia.com/t/how-do-i-disable-gravity-for-a-given-object-using-python-api/297501

        Args:
            prim_stage_path (_type_): like "/World/box"
        """
        prim = stage.GetPrimAtPath(prim_stage_path)
        physx_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        physx_api.CreateDisableGravityAttr(True)


    def get_X_Wmo(self, T_Wmo, custom_pose):
        """
        Get the pose of the obstacle in the world model origin frame (Wmo).
        """
        # get pose expressed in the world model frame W
        if len(custom_pose) != 7:
            p_obs, q_obs = self.simulation_representation.get_world_pose() # take pose from simulation (self.simulation_enabled must be True)
        else: # use custom pose
            p_obs, q_obs = custom_pose[:3], custom_pose[3:]

        # express the pose of the obstacle in the world model origin frame (Wmo)
        p_obs_Wmo, q_obs_Wmo = FrameUtils.world_to_F(T_Wmo[:3], T_Wmo[3:], p_obs, q_obs) # get the pose of the obstacle (frame F2) in the world model origin frame (frame F)
        X_obs_Wmo = Pose(self.tensor_args.to_device(torch.from_numpy(p_obs_Wmo)), self.tensor_args.to_device(torch.from_numpy(q_obs_Wmo))) # Pose (X) of the obstacle (obs) expressed in the world model origin frame (Wmo)
        return X_obs_Wmo 
    

    def get_from_cchecker(self, world_model_idx):
        """
        Get the curobo representation of the obstacle from the world model.
        """
        return self.registered_world_models[world_model_idx].objects[self.indices_in_registered_world_models[world_model_idx]]
    
    def _make_world_model_representation(self, T_Wmo, custom_pose=np.array([])) -> Union[Cuboid, Mesh]:
        """

        
        Initialize the curobo representation of the obstacle (not yet injected into the world config).
        https://curobo.org/_api/curobo.types.math.html#curobo.types.math.Pose
        https://curobo.org/get_started/2c_world_collision.html
        
        # Here we initialize the curobo representation of the obstacle in its collision checker.
        # For every type of obstacle, we need to define the curobo representation of the obstacle in its collision checker.
        # Valid options are:
        # - Cuboid
        # - Sphere
        # - Mesh
        # - Capsule
        # - Cylinder
        # - Cone

        # More info: https://curobo.org/get_started/2c_world_collision.html 
        
        """
        
        X_obs_Wmo = self.get_X_Wmo(T_Wmo, custom_pose) # get the pose expressed in the world model origin frame (Wmo)
        pose = X_obs_Wmo.tolist()
        dims = self.dims
        # dims = self.simulation_representation.get_world_scale()* self.simulation_representation.get_size() 
        # ⚠️ pose is in world frame
        # Here we initialize the curobo representation of the obstacle in its collision checker.
        if self.curobo_type == "cuboid":
            # dims = self.simulation_representation.get_world_scale()* self.simulation_representation.get_size() 
            curobo_obstacle = Cuboid(
                name=self.name,
                pose=pose,
                dims=list(dims)
            )
        elif self.curobo_type == "mesh":
            # check if the mesh file exists under default assets path of curobo
            assert os.path.exists(join_path(get_assets_path(), self.mesh_file_sub)), f"Subpath to the mesh file {self.mesh_file_sub} must exist under {get_assets_path()}"
            curobo_obstacle = Mesh(
                name=self.name,
                pose=pose,
                file_path=self.mesh_file_sub,
                scale=dims,
            )
            # mesh.pose = pose
            # mesh.name = self.name
            #curobo_obstacle = mesh


        return curobo_obstacle
    
        
    def add_to_world_model(self, cu_world_model:WorldConfig, T_Wmo:np.ndarray, custom_pose=np.array([])):
        """
        Add the obstacle to a given world model. side not: this is currently happening before solvers are initialized (we pass each solver its world model with the initial pose of the obstacles).
        """
        assert self.cchecking_enabled, "Error: Collision checking must be enabled to add the obstacle to the world model"

        cuObs_Wmo = self._make_world_model_representation(T_Wmo, custom_pose) # curobo representation of the obstacle expressdd in the provided world model frame (Twmo)
        cu_world_model.add_obstacle(cuObs_Wmo) # adding the curobo obs representation to the world model (inplace)
        self.registered_world_models.append(cu_world_model) # Register the world model that the obstacle is added to and the transform from world to world moedl origin
        self.registered_Wmo_transforms.append(T_Wmo)
        self.indices_in_registered_world_models.append(len(self.registered_world_models[-1].objects) - 1)
        world_model_idx = len(self.registered_world_models) - 1
        return world_model_idx # return the index of the world model that the obstacle is added to

    def update_cchecker(self, cchecker_idx, custom_pose=np.array([]), custom_dims=np.array([])):
        """
        Update the pose of the obstacle in a given collision checker (cchecker must be registered first).
        """
        assert self.cchecking_enabled, "Error: Collision checking must be enabled to update the obstacle in the collision checker"
        cchecker = self.registered_ccheckers[cchecker_idx]
        # pose update:
        T_Wmo = self.registered_Wmo_transforms[cchecker_idx]
        X_Wmo = self.get_X_Wmo(T_Wmo, custom_pose) # curobo representation of the obstacle expressdd in the provided world model frame (Twmo)
        cchecker.update_obstacle_pose(self.name, X_Wmo) # update the pose of the obstacle in the collision checker
        
        # dims update:
        # Get the dims to update the curobo representation of the obstacle in the collision checker
        if self.simulation_enabled:
            if self.sim_type == DynamicCuboid:
                cur_dims = self.simulation_representation.get_world_scale()* self.simulation_representation.get_size() 
            elif self.sim_type == DynamicSphere:
                cur_dims = None # TODO
            elif self.sim_type == XFormPrim:
                cur_dims = self.simulation_representation.get_local_scale()
        else:
            if len(custom_dims) == 3:
                cur_dims = list(custom_dims)
            else:
                cur_dims = self.dims
        cur_dims = list(custom_dims)
        
        # update the curobo representation of the obstacle in the collision checker with the updated dims
        self.get_from_cchecker(cchecker_idx).dims = cur_dims

        self.dims = cur_dims
        
    def get_updated_dims(self) -> list[float]:
        """
        Get the updated dims of the obstacle in the collision checker.
        """
        if self.simulation_enabled:
            cur_dims =  self.simulation_representation.get_world_scale()* self.simulation_representation.get_size() 
        else:
            return self.dims
    
    def update_registered_ccheckers(self, custom_pose=np.array([]), custom_dims=np.array([])):
        """
        Update all registered collision checkers with the new pose and dims of the obstacle.
        You must register the collision checkers with the obstacle first using the register_ccheckers method.
        """
        assert self.cchecking_enabled, "Error: Collision checking must be enabled to update the obstacle in the collision checker"
        # custom pose (if passed) should be expressed in world frame
        for cchecker_idx in range(len(self.registered_ccheckers)):
            self.update_cchecker(cchecker_idx, custom_pose, custom_dims)
    
    def register_ccheckers(self, ccheckers):
        """Register the collision checkers with the obstacle.
        Pass the cchekers list in the order of the collision checkers in the world model (and the transform from world to world model origin) corresponding to the order of the ccheckers in the cchecker list.
        (side note: this step can be done only after solvers are initialized.)
        Args:
            ccheckers (_type_): _description_
        """
        assert self.cchecking_enabled, "Error: Collision checking must be enabled to register the collision checkers with the obstacle"
        for cchecker in ccheckers:
            self.registered_ccheckers.append(cchecker)
            
    def get_const_vel_plan(self, horizon:int, dt:float):
        """
        Get the plan of the obstacle, based on the current pose and velocity.
        """
        ans = torch.zeros(horizon, 7) # [x, y, z, qw, qx, qy, qz]
        assert self.sim_type == DynamicSphere, "Error: Only sphere can be planned for now"
        # rot = [1,0,0,0] 
        assert self.simulation_enabled, "Error: Simulation must be enabled to plan the obstacle"
        cur_pose = self.simulation_representation.get_world_pose()
        cur_vel = self.simulation_representation.get_linear_velocity()
        # cur_ang_vel = self.simulation_representation.get_angular_velocity()
        
        for i in range(horizon):
            ans[i, :3] = cur_pose[:3] + cur_vel * dt * i
            ans[i, 3:] = cur_pose[3:]
        
# class PrimWrapper:
#     def __init__(self, prim_path: str):
#         # Init after the prim is added to the stage!
#         self.prim_path = prim_path
#         self.prim = 
        
#     def set_world_pose(self, position, orientation):
#         self.prim.set_world_pose(position, orientation)
    
#     def set_local_pose(self, position, orientation):
#         self.prim.set_local_pose(position, orientation)

#     def set_local_scale(self, scale):
#         self.prim.set_local_scale(scale)
