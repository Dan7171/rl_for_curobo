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

# Initialize the simulation app first (must be before "from omni.isaac.core")

# from omni.isaac.kit import SimulationApp  
# simulation_app = SimulationApp({"headless": False})

# Now import other Isaac Sim modules
# https://medium.com/@kabilankb2003/isaac-sim-core-api-for-robot-control-a-hands-on-guide-f9b27f5729ab
from omni.isaac.core.objects import DynamicCuboid, DynamicSphere, DynamicCapsule, DynamicCylinder, DynamicCone, GroundPlane
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core import World # https://forums.developer.nvidia.com/t/cannot-import-omni-isaac-core/242977/3

from pxr import PhysxSchema, UsdPhysics

# Import helper from curobo examples

# CuRobo
from curobo.geom.types import Capsule, Cuboid, Cylinder, Mesh, Sphere, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose 
from projects_root.projects.dynamic_obs.dynamic_obs_predictor.frame_utils import FrameUtils
# Initialize CUDA device
a = torch.zeros(4, device="cuda:0") 



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
                visual_material=None):
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
            world (_type_): issac sim world instance (related to the simulator)
            cu_world_models (_type_): # cu_world_models is the world model of curobo. (represents the model of the world where obstacles are interlive in curobo)
        """
        assert curobo_type in ["cuboid", "sphere", "mesh", "capsule", "cylinder", "cone"]
        self.obs_type_to_sim_type = {"cuboid": DynamicCuboid, "sphere": DynamicSphere, "mesh": None, "capsule": DynamicCapsule, "cylinder": DynamicCylinder}
        self.name = name
        self.path = f'/World/curobo_world_cfg_obs_visual_twins/{name}' # path to the obstacle in the simulation
        self.dims = dims
        self.curobo_type = curobo_type
        self.sim_type = self.obs_type_to_sim_type[curobo_type]
        self.tensor_args = TensorDeviceType()  # Add this to handle device placement
        self.simulation_representation = self._init_obstacle_in_simulation(world, pose[:3], pose[3:], self.dims, self.sim_type, color, mass, gravity_enabled,sim_collision_enabled,visual_material, linear_velocity, angular_velocity)
        self.world_models = [] # world models that the obstacle is added to
        
  
      
    
    def set_simulation_refernce(self, simulation_refernce):
        self.simulation_refernce = simulation_refernce
    
    def _init_obstacle_in_simulation(self, world, position, orientation, dims, curobo_type, color,  mass, gravity_enabled,sim_collision_enabled,visual_material, linear_velocity, angular_velocity):
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
        if curobo_type == DynamicCuboid:
            obstacle = self._init_DynamicCuboid_for_simulation(world, position, orientation, dims, color,  mass, gravity_enabled,sim_collision_enabled,visual_material, linear_velocity, angular_velocity)

        return obstacle
    
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
        return prim
    
    def update_world_coll_checker_with_sim_pose(self, world_coll_checker:Union[WorldConfig, List[WorldConfig]]):
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

    def _init_world_model_representation(self, T_Wmo) -> Union[Cuboid, Mesh]:
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
        curobo_obstacle = None
        p_obs, q_obs = self.simulation_representation.get_world_pose() # X_obs_W (specified in world frame)
        p_obs_Wmo, q_obs_Wmo = FrameUtils.world_to_F(T_Wmo[:3], T_Wmo[3:], p_obs, q_obs) # get the pose of the obstacle (frame F2) in the world model origin frame (frame F)
        X_obs_Wmo = Pose(torch.from_numpy(p_obs_Wmo), torch.from_numpy(q_obs_Wmo)) # Pose (X) of the obstacle (obs) expressed in the world model origin frame (Wmo)
        
        
        # Here we initialize the curobo representation of the obstacle in its collision checker.
        if self.curobo_type == "cuboid":
            dims = self.simulation_representation.get_world_scale()* self.simulation_representation.get_size() 
            curobo_obstacle = Cuboid(
                name=self.name,
                pose=X_obs_Wmo.tolist(),
                dims=list(dims)
            )
        elif self.curobo_type == "mesh":
            pass
            # TODO: Implement this
            # curobo_obstacle = Mesh(
            #     name=self.name,
            #     pose=X_obs_Wmo.tolist(),
            #     mesh_path=self.simulation_representation.get_mesh_path # type: ignore()
            # )

        return curobo_obstacle
    
  
        
    def add_to_world_model(self, cu_world_model:WorldConfig, T_Wmo:np.ndarray):
        """
        Add the obstacle to a given world model.
        """
        curobo_obs_T_Wmo = self._init_world_model_representation(T_Wmo) # curobo representation of the obstacle expressdd in the provided world model frame (Twmo)
        cu_world_model.add_obstacle(curobo_obs_T_Wmo) # adding the curobo obs representation to the world model (inplace)
        self.world_models.append((cu_world_model, T_Wmo)) # save in the list of world models that the obstacle is added to together with the transform from world to world moedl origin
        return cu_world_model # return the modified world model for convenience
    


    
    

    # def set_cube_dims(self, dims:list):
    #     self.curobo_representation.dims = dims
    #     self.simulation_representation.set_size(dims)

    