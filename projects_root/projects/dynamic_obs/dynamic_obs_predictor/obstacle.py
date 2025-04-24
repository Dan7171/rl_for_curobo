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
                obstacle_type:str, 
                cu_world_models:Union[WorldConfig, List[WorldConfig]]=[],
                T_Wmos:Union[np.ndarray, List[np.ndarray]]=[],
                X_initial:Iterable[float]=np.array([1,1,1,1,0,0,0]), 
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
            X_initial (_type_): initial position of the obstacle in the world frame ([x,y,z, qw, qx, qy, qz]).
            dims (_type_): if cuboid, scalar (side length (m)).
            obstacle_type (np.array): [r,g,b]
            color (_type_): [r,g,b,alpha] (alpha is the opacity of the obstacle)
            mass (_type_): kg
            linear_velocity (_type_): vx, vy, vz (m/s)
            angular_velocity (_type_):wx, wy, wz (rad/s)
            gravity_enabled (bool): enable/disable gravity for the obstacle. If False, the obstacle will not be affected by gravity.
            world (_type_): issac sim world instance (related to the simulator)
            cu_world_models (_type_): # cu_world_models is the world model of curobo. (represents the model of the world where obstacles are interlive in curobo)
        """
        assert obstacle_type in ["cuboid", "sphere", "mesh", "capsule", "cylinder", "cone"]
        self.obs_type_to_sim_type = {"cuboid": DynamicCuboid, "sphere": DynamicSphere, "mesh": None, "capsule": DynamicCapsule, "cylinder": DynamicCylinder}
        self.obs_type_to_curobo_type = {"cuboid": Cuboid, "sphere": Sphere, "mesh": Mesh, "capsule": Capsule, "cylinder": Cylinder}
        self.curobo_type_to_world_model_type = {"cuboid": Cuboid, "sphere": Mesh, "mesh": Mesh, "capsule": Mesh, "cylinder": Mesh}
        
        self.name = name
        self.path = f'/World/curobo_world_cfg_obs_visual_twins/{name}' # path to the obstacle in the simulation
        self.cur_pos = X_initial[:3] # position of the obstacle in the world frame p_obs_W
        self.cur_rot = X_initial[3:] # orientation of the obstacle in the world frame q_obs_W
        
        self.dims = dims
        self.obstacle_type = obstacle_type
        self.sim_type = self.obs_type_to_sim_type[obstacle_type]
        self.tensor_args = TensorDeviceType()  # Add this to handle device placement
        self.cu_world_models = cu_world_models 
        self.T_Wmos = T_Wmos

        
        # initialize the obstacle in the simulation and the curobo representation of the obstacle in its collision checker
        
        # This is the simulation representation of the obstacle: responsible for physics and 3d visualization in simulator!
        # This is the visual representation of the obstacle in the simulation!
        self.simulation_representation = self._init_obstacle_in_simulation(world, self.cur_pos, self.cur_rot, self.dims, self.sim_type, color, mass, gravity_enabled,sim_collision_enabled,visual_material, linear_velocity, angular_velocity)
        self.add_to_world_models(self.cu_world_models, self.T_Wmos)

        # This is the curobo representation of the obstacle: responsible for collision checking in curobo! (no visual representation, just a twin of the simulation representation)
        # This is invisible in the simulation, but will be visible in the collision checker!
        # self.curobo_representation = self._init_world_model_representation() # initialize the curobo representation of the obstacle based on the simulation representation
        
        # If world model was provided, register (inject) the curobo representation of the obstacle in curobo representation of world.
        # if self.cu_world_models is not None:
        #     self.inject_curobo_obs(self.cu_world_models)

    
    def set_simulation_refernce(self, simulation_refernce):
        self.simulation_refernce = simulation_refernce
    
    def _init_obstacle_in_simulation(self, world, position, orientation, dims, obstacle_type, color,  mass, gravity_enabled,sim_collision_enabled,visual_material, linear_velocity, angular_velocity):
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
            obstacle = self._init_DynamicCuboid_for_simulation(world, position, orientation, dims, obstacle_type, color,  mass, gravity_enabled,sim_collision_enabled,visual_material, linear_velocity, angular_velocity)

        return obstacle
    
    def _init_DynamicCuboid_for_simulation(self,world, position, orientation, dims, obstacle_type, color,  mass, gravity_enabled,sim_collision_enabled,visual_material, linear_velocity, angular_velocity):
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
        p_obs, q_obs = self.simulation_representation.get_world_pose() # specified in world frame
        p_obs_Wmo, q_obs_Wmo = FrameUtils.world_to_F(T_Wmo[:3], T_Wmo[3:], p_obs, q_obs) # get the pose of the obstacle (frame F2) in the world model origin frame (frame F)
        X_obs_Wmo = Pose(torch.from_numpy(p_obs_Wmo), torch.from_numpy(q_obs_Wmo)) # Pose (X) of the obstacle (obs) expressed in the world model origin frame (Wmo)
        
        
        # Here we initialize the curobo representation of the obstacle in its collision checker.
        if self.obstacle_type == "cuboid":
            cube_edge_len = self.simulation_representation.get_size() # self.dims should work too
            curobo_obstacle = Cuboid(
                name=self.name,
                pose=X_obs_Wmo.tolist(),
                dims=[cube_edge_len, cube_edge_len, cube_edge_len]
            )
        elif self.obstacle_type == "mesh":
            pass
            # TODO: Implement this
            # curobo_obstacle = Mesh(
            #     name=self.name,
            #     pose=X_obs_Wmo.tolist(),
            #     mesh_path=self.simulation_representation.get_mesh_path # type: ignore()
            # )

        return curobo_obstacle
    
    def add_to_world_models(self, cu_world_models:Union[WorldConfig, List[WorldConfig]], T_Wmos: Union[np.ndarray, List[np.ndarray]]):
        """ Initiates the curobo representation of the obstacle into a given world config (world model, the object that contains the obstacles in world in curobo). 
        And making a link between the simualtion representation of the object to each of the world models provided.
        
        Args:
            cu_world_models: list of world configs (world models) or a single world config (world model).
            T_Wmo: a same length list or of arrays or a single array, each representing a transform (T) of the world model origin (Wmo) in the world frame  (inspired by drake notations https://drake.mit.edu/doxygen_cxx/group__multibody__quantities.html#:~:text=C-,T_BC,-The%20relationship%20between).
            Those are actually just the poses of the origins of the base frames of the world models in the world frame (normally the origin of the world frame is the same as the origin of the base frame of the robot which the world model is attached to).

        """
    
        if isinstance(self.cu_world_models, list):
            assert len(T_Wmos) == len(self.cu_world_models)
            for i in range(len(self.cu_world_models)):
                self.cu_world_models[i].add_obstacle(self._init_world_model_representation(T_Wmos[i])) # adds (injects) the curobo representation to the curobo world model
        
        else:
            assert isinstance(T_Wmos, np.ndarray)
            T_Wmo = T_Wmos # a single array
            cu_world_model = self.cu_world_models # a single world config
            cu_world_model.add_obstacle(self._init_world_model_representation(T_Wmo))

    
    


    

    

    # def set_cube_dims(self, dims:list):
    #     self.curobo_representation.dims = dims
    #     self.simulation_representation.set_size(dims)

    