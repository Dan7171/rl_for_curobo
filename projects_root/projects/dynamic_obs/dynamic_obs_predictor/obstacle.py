try:
    # Third Party
    import isaacsim
except ImportError:
    pass

# Third Party
from typing import Optional
import torch
import numpy as np

# Initialize the simulation app first (must be before "from omni.isaac.core")

# from omni.isaac.kit import SimulationApp  
# simulation_app = SimulationApp({"headless": False})

# Now import other Isaac Sim modules
# https://medium.com/@kabilankb2003/isaac-sim-core-api-for-robot-control-a-hands-on-guide-f9b27f5729ab
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.materials import PhysicsMaterial
from pxr import PhysxSchema, UsdPhysics

# Import helper from curobo examples

# CuRobo
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose 

# Initialize CUDA device
a = torch.zeros(4, device="cuda:0") 



class Obstacle:
    def __init__(self, name, initial_pose, dims, obstacle_type, color, mass, linear_velocity, angular_velocity, gravity_enabled, world, world_model_curobo:Optional[WorldConfig]=None, sim_collision_enabled=True,visual_material=None):
        """
        Creates the representations of the obstacle in the simulation form and in the curobo form (making equivalent representations).
        If world model is provided, the curobo representation of the obstacle is injected into the world model of curobo (in the simulation in happens automatically when simulation representation is created).

        See https://curobo.org/get_started/2c_world_collision.html

        Args:
            name (str): object name. Must be unique.
            initial_pose (_type_): initial position of the obstacle in the world frame ([x,y,z, qw, qx, qy, qz]).
            dims (_type_): if cuboid, scalar (side length (m)).
            obstacle_type (np.array): [r,g,b]
            color (_type_): [r,g,b,alpha] (alpha is the opacity of the obstacle)
            mass (_type_): kg
            linear_velocity (_type_): vx, vy, vz (m/s)
            angular_velocity (_type_):wx, wy, wz (rad/s)
            gravity_enabled (bool): enable/disable gravity for the obstacle. If False, the obstacle will not be affected by gravity.
            world (_type_): issac sim world instance (related to the simulator)
            world_model_curobo (_type_): # world_model_curobo is the world model of curobo. (represents the model of the world where obstacles are interlive in curobo)
        """
        self.name = name
        self.path = f'/World/new_obstacles/{name}' # path to the obstacle in the simulation
        self.cur_pos = initial_pose[:3] # position of the obstacle in the world frame p_obs_W
        self.cur_rot = initial_pose[3:] # orientation of the obstacle in the world frame q_obs_W
        
        self.dims = dims
        self.prim_type = obstacle_type
        self.tensor_args = TensorDeviceType()  # Add this to handle device placement
        # self.world_model_curobo = world_model_curobo
        self.world_model_curobo = world_model_curobo 
        self.linear_velocity = linear_velocity
        self.angular_velocity = angular_velocity
        # initialize the obstacle in the simulation and the curobo representation of the obstacle in its collision checker
        self.simulation_representation = self._init_obstacle_in_simulation(world, self.cur_pos, self.cur_rot, self.dims, obstacle_type, color, mass, gravity_enabled,sim_collision_enabled,visual_material)
        self.curobo_representation = self._init_obstacle_curobo_rep() # initialize the curobo representation of the obstacle based on the simulation representation


    
    def set_simulation_refernce(self, simulation_refernce):
        self.simulation_refernce = simulation_refernce
    
    def _init_obstacle_in_simulation(self, world, position, orientation, dims, obstacle_type, color=None,  mass=1.0, gravity_enabled=True,sim_collision_enabled=True,visual_material=None):
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
        if color is None:
            color = np.array([0.0, 0.0, 0.1])  # Default blue color
        
        if obstacle_type == DynamicCuboid:
            obstacle = self._init_DynamicCuboid_for_simulation(world, position, orientation, dims, color, mass, gravity_enabled, np.array(self.linear_velocity),np.array(self.angular_velocity),sim_collision_enabled,visual_material)

        return obstacle
    
    def _init_DynamicCuboid_for_simulation(self,world, position, orientation, size, color, mass=1.0, gravity_enabled=True,linear_velocity:np.array=np.nan, angular_velocity:np.array=np.nan,sim_collision_enabled=True,visual_material=None):
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
        prim = DynamicCuboid(prim_path=self.path,name=self.name, position=position,orientation=orientation,size=size,color=color,mass=mass,density=0.9,visual_material=visual_material)         
        material = PhysicsMaterial( # https://www.youtube.com/watch?v=tHOM-OCnBLE
            prim_path=self.path+"/aluminum",  # path to the material prim to create
            dynamic_friction=0,
            static_friction=0,
            restitution=0
        )
        prim.apply_physics_material(material)
        if linear_velocity is not np.nan:
            prim.set_linear_velocity(linear_velocity)
        if angular_velocity is not np.nan:
            prim.set_angular_velocity(angular_velocity)
        if not gravity_enabled:
            self.disable_gravity(self.path, world.stage)

        if not sim_collision_enabled: # Disable collision for the obstacle in the simulation
            prim.set_collision_enabled(False)
        world.scene.add(prim)
        return prim
    
    def update_world_coll_checker_with_sim_pose(self,world_coll_checker):
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

    def _init_obstacle_curobo_rep(self):
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
        position_isaac_dynamic_prim, orient_isaac_dynamic_prim = self.simulation_representation.get_world_pose() # specified in world frame
        w_obj_pose = Pose(torch.from_numpy(position_isaac_dynamic_prim), torch.from_numpy(orient_isaac_dynamic_prim))
        
        
        # Here we initialize the curobo representation of the obstacle in its collision checker.
        if self.prim_type == DynamicCuboid:
            cube_edge_len = self.simulation_representation.get_size()
            curobo_obstacle = Cuboid(
                name=self.name,
                pose=w_obj_pose.tolist(),
                dims=[cube_edge_len, cube_edge_len, cube_edge_len]
            )
        
        # If world model was provided, register (inject) the curobo representation of the obstacle in curobo representation of world.
        if self.world_model_curobo is not None:
            self.inject_curobo_obs(self.world_model_curobo)

        return curobo_obstacle
    

    def inject_curobo_obs(self, world_model_curobo:WorldConfig):
        """ Inject the curobo representation of the obstacle into a given world config (world model, the object that contains the obstacles in world in curobo).

        Args:
            world_model_curobo (_type_): _description_
        """
        if self.world_model_curobo is None:
            self.world_model_curobo = world_model_curobo # sets the world this object will be living in
        self.world_model_curobo.add_obstacle(self.curobo_representation) # adds (injects) the curobo representation to the curobo world model
    
    

    # def set_cube_dims(self, dims:list):
    #     self.curobo_representation.dims = dims
    #     self.simulation_representation.set_size(dims)
