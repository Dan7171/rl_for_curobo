try:
    # Third Party
    import isaacsim
except ImportError:
    pass

# Third Party
import torch
import numpy as np

# Initialize the simulation app first (must be before "from omni.isaac.core")

# from omni.isaac.kit import SimulationApp  
# simulation_app = SimulationApp({"headless": False})

# Now import other Isaac Sim modules
# https://medium.com/@kabilankb2003/isaac-sim-core-api-for-robot-control-a-hands-on-guide-f9b27f5729ab
from omni.isaac.core.objects import DynamicCuboid
from pxr import PhysxSchema

# Import helper from curobo examples
from projects_root.utils.helper import add_extensions, add_robot_to_scene

# CuRobo
from curobo.geom.types import Cuboid
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose 
from projects_root.projects.dynamic_obs.dynamic_obs_predictor.dynamic_obs_coll_checker import DynamicObsCollPredictor

# Initialize CUDA device
a = torch.zeros(4, device="cuda:0") 

class Obstacle:
    def __init__(self, name, pose, dims, obstacle_type, color, mass, linear_velocity, angular_velocity, gravity_enabled, world, world_cfg):
        """_summary_
        See https://curobo.org/get_started/2c_world_collision.html

        Args:
            name (_type_): _description_
            pose (_type_): _description_
            dims (_type_): _description_
            obstacle_type (_type_): _description_
            color (_type_): _description_
            mass (_type_): _description_
            gravity_enabled (_type_): _description_
            world (_type_): issac sim world instance (related to the simulator)
            world_cfg (_type_): curobo world config instance (related to the collision checker of curobo)
        """
        self.name = name
        self.path = f'/World/new_obstacles/{name}'
        self.initial_pose = pose
        self.dims = dims
        self.prim_type = obstacle_type
        self.tensor_args = TensorDeviceType()  # Add this to handle device placement
        self.world_cfg = world_cfg
        self.linear_velocity = linear_velocity
        self.angular_velocity = angular_velocity
        #initialize the obstacle in the simulation and the curobo representation of the obstacle in its collision checker
        self.simulation_representation = self._init_obstacle_in_simulation(world, self.initial_pose[:3], self.dims, obstacle_type, color, mass, gravity_enabled)
        self.curobo_representation = self._init_curobo_obstacle() # initialize the curobo representation of the obstacle based on the simulation representation
        self.cur_pos = self.initial_pose[:3]


    
    def set_simulation_refernce(self, simulation_refernce):
        self.simulation_refernce = simulation_refernce
    
    def _init_obstacle_in_simulation(self, world, position, dims, obstacle_type, color=None,  mass=1.0, gravity_enabled=True):
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
            obstacle = self._init_DynamicCuboid_for_simulation(world, position, dims, color, mass, gravity_enabled, np.array(self.linear_velocity),np.array(self.angular_velocity))

        return obstacle
    
    def _init_DynamicCuboid_for_simulation(self,world, position, size, color, mass=1.0, gravity_enabled=True,linear_velocity:np.array=np.nan, angular_velocity:np.array=np.nan):
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
        prim = DynamicCuboid(prim_path=self.path,name=self.name, position=position,size=size,color=color,mass=mass,density=0.9)         

        if linear_velocity is not np.nan:
            prim.set_linear_velocity(linear_velocity)
        if angular_velocity is not np.nan:
            prim.set_angular_velocity(angular_velocity)
        if not gravity_enabled:
            self.disable_gravity(self.path, world.stage)
        world.scene.add(prim)
        return prim
    
    # def update(self,mpc):
    #     # get the updated pose of the obstacle in the simulation
    #     position_isaac_dynamic_prim, orient_isaac_dynamic_prim = self.simulation_representation.get_world_pose()
    #     # update the current position of the obstacle
    #     self.cur_pos = position_isaac_dynamic_prim
    #     # update the curobo representation of the obstacle in its collision checker
    #     self._update_curobo_obstacle_pose(mpc, position_isaac_dynamic_prim, orient_isaac_dynamic_prim)
        
        

    def _update_curobo_obstacle_pose(self,mpc, position_isaac_dynamic_prim, orient_isaac_dynamic_prim):
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
        mpc.world_coll_checker.update_obstacle_pose(self.name, w_obj_pose)

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

    def _init_curobo_obstacle(self):
        """
        Initialize the curobo representation of the obstacle in its collision checker.
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
        
        
        # register the curobo representation of the obstacle in the curobo collision checker
        self.world_cfg.add_obstacle(curobo_obstacle) 
        return curobo_obstacle
    
        