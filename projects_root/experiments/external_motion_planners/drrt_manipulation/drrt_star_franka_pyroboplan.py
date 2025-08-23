"""
drrt_star_franka_pyroboplan.py
--------------------------------
Standalone demonstration that:
  
  Input:  
  n robots. For each robot i: start pose (base), end goal pose (ee) and start config (in c-space) + obstacles (sphere / cube)
  - start pose (base) = initial pose of the robot i
  - end goal pose (ee) = final pose of the robot i
  - start config (in c-space) = initial joint config of the robot i
  - obstacles (sphere / cube) = obstacles in the environment
  
  Initialization: 
  - Launches Isaac-Sim and setting up n robots and obstacles.
  - Extracts Sphere / Cube prims under /World and converts them to PyRoboPlan obstacles.
  - Builds a PRM each robot i ("roadmap Gi"). Accessible through prm_planners[i] (PRMPlanner)
  
  3. While True:
        t = t+1
        env_obstacles = isaac.get_obtacles()
        all_roadmaps_verified = True
        For each robot i: 
            # q_start_t= initial joint config of robot i at time t
            # target_pose_t = # task space goal pose (ee) of robot i at time t

            - collision_models[i].update(env_obstacles) # update collision model with new obstacles
            
            # IK until collision free (or MAX_IK_TRIALS is reached)
            ik_success = False
            - for trial in MAX_IK_TRIALS: 
                q_target_t = IK(target_pose_t) # Perform IK to get a joint config that brings the ee to target_pose_t
                if not check_collisions_at_state(collision_models[i], q_target_t):
                    ik_success = True
                    break
            - if not ik_success:
                continue # tick the world until conditions are met (e.g. obstacle is removed)

            # PRM* / PRM

            # check if start and goal are collision free and can be connected to the (individual) roadmap of robot i   
            if not verified_for_search(i,q_start_t,q_target_t): 
                all_roadmaps_verified = False 
                continue # tick the world until conditions are met (e.g. obstacle is removed)
        
        if all_roadmaps_verified:
            # run drrt* algorithm
            drrt_input = [(q_start_t, q_target_t,prm_planners[i]) for i in range(n_robots)]
            agents_paths = drrt*(drrt_input) # TODO - this is the core of drrt* algorithm

            
            

            
        

         """
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import List
import coal

import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
 

from pyroboplan.models.panda import (
        load_models,
        add_self_collisions,
        add_object_collisions,
    )
from pyroboplan.planning.prm import PRMPlanner, PRMPlannerOptions
from pyroboplan.core.utils import get_random_collision_free_state,check_collisions_at_state
from pyroboplan.planning.graph import Node

from curobo.util_file import load_yaml # to load curobo robot cfg from path (yml)
from curobo.types.math import Pose



class Isaac:
    def __init__(self):
        from omni.isaac.kit import SimulationApp
        self.simulation_app:SimulationApp = None
        self.world = None
        self.usd_help = None
        self.robots = []
        self.robot_prim_paths = []
        self.prims = []


    def run_app(self, headless: bool):
        from omni.isaac.kit import SimulationApp
        self.simulation_app = SimulationApp({"headless": headless})
    
    def _ensure_app_started(self):
        if self.simulation_app is None:
            raise RuntimeError("Must Run Isaac.start_app() before using Isaac.get_stage()")
        
    def close_app(self):
        self._ensure_app_started()
        self.simulation_app.close()

    def get_stage(self):
        self._ensure_app_started()
        from omni.isaac.core.utils.stage import get_current_stage
        return get_current_stage()

    def init_world(self):
        self._ensure_app_started()
        from omni.isaac.core import World
        from curobo.util.usd_helper import UsdHelper
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()
        self.usd_help = UsdHelper()
        self.usd_help.load_stage(self.world.stage)
        return self.world

    def get_prims_from_stage(self):
        self._ensure_app_started()
         
        isaac_cu_world_W =self.usd_help.get_obstacles_from_stage( 
            only_paths=['/World'], # look for obs only under the world prim path
            reference_prim_path='/World', # obstacles are expressed in robot frame! (not world frame). That's why we marked it with 'R'
            ignore_substring=['target','curobo','robot']
        )
        
        return isaac_cu_world_W
        # from isaacsim.core.utils.prims import get_all_matching_child_prims
        # return get_all_matching_child_prims(self.world.scene, prim_path)


    
    def add_robot(self, a_idx: int, robot_cfg_path: str, base_pose: List[float]=[0,0,0,1,0,0,0]):

        self._ensure_app_started()
        from projects_root.utils.helper import add_robot_to_scene # to add robot to scene using robot cfg
        robot_cfg = load_yaml(robot_cfg_path)["robot_cfg"]
        # self.usd_help.add_subroot('/World', f'/World/robot_{a_idx}', Pose.from_list(base_pose))
        robot, robot_prim_path = add_robot_to_scene(robot_cfg, self.world, subroot=f'/World/robot_{a_idx}', robot_name=f'robot_{a_idx}', position=base_pose[:3], orientation=base_pose[3:], initialize_world=False) # add_robot_to_scene(self.robot_cfg, self.world, robot_name=self.robot_name, position=self.p_R)"
        robot.set_world_pose(base_pose[:3], base_pose[3:])
        self.robots.append(robot)
        self.robot_prim_paths.append(robot_prim_path)
   
        

    def add_objects(self, objects:List[SimpleObstacle]):
        self._ensure_app_started()
        from omni.isaac.core.objects import VisualSphere, VisualCuboid
        from pxr import Gf
        
        for i, obj in enumerate(objects):
            obj_type = obj.obj_type
            obj_position = obj.pos
            obj_size = obj.size
            obj_name = f'{obj_type}_{i}'


            if obj_type == 'sphere':
                prim_class = VisualSphere 
                kwargs = {'radius': obj_size}
            elif obj_type == 'cube':
                prim_class = VisualCuboid
                kwargs = {'size': obj_size}
            else:
                raise ValueError(f"Unknown object type: {obj_type}")
            obj = prim_class(prim_path=f'/World/SpawnedObs/{obj_name}', name=f'{obj_name}', position=Gf.Vec3d(obj_position),**kwargs)
            self.prims.append(obj)



# -------------------- Environment sphere extraction --------------------

# If future PyRoboPlan versions expose obstacle helpers import them, otherwise skip.
try:
    from pyroboplan.models.utils import attach_sphere_obstacle # TODO: currently raises an error, warining is printed. Check how to actually add obstacles.
except ImportError:
    attach_sphere_obstacle = None

class SimpleObstacle:
    def __init__(self, obj_type: str, pos: list, size:float, quat: list=[1,0,0,0]):
        self.obj_type = obj_type
        self.pos = pos
        self.size = size
        self.quat = quat

class PRPIsaacBridge:
    """
    PyRoboPlan <-> Isaac Sim bridge, to transfer obstacles and robot model between the two.
    """
    def __init__(self):
        self.isaac = Isaac() # not yet started, will be started when needed
        self.prp = PRPWrap()
        self.ee_names = []

    def isaac_obj_to_simple(self, prim):
        p = np.array(prim.GetAttribute("xformOp:translate").Get())
        obj_name = prim.GetName()
        if "sphere" in prim.GetTypeName().lower():
            r = prim.GetAttribute("radius").Get()
            return SimpleObstacle("sphere", p, r)
        elif "cube" in prim.GetTypeName().lower():
            size = prim.GetAttribute("size").Get()
            return SimpleObstacle("cube", p, size)
        else:
            raise ValueError(f"Unknown object type: {prim.GetTypeName()}")
 
    
        
        

class PRPWrap:
    def __init__(self, open_viz=True):
        self.robot_models = []
        self.robot_datas = []
        self.collision_models = []
        self.collision_datas = []
        self.visual_models = [] 
        self.prm_planners = [] # objects of PRMPlanner class, contains the roadmap (and optional A* planner to plan paths based on the roadmap)
        self.open_viz = open_viz # whether to open the visualizer of pyroboplan
        self.vizs = [] # objects of MeshcatVisualizer class, contains the visualizer of pyroboplan
    
    # def update_col_model(self, col_model, obstacles: List[SimpleObstacle]):
    #     has_changed = False
    #     for obs in obstacles:
    #         cur_pose = obs.centre
    #         prev_pose = # TODO: get the previous pose of the obstacle (from the collision model)
    #         if np.linalg.norm(cur_pose - prev_pose) > 0.01: # TODO: check if the obstacle has moved by more than epsilon meters
    #             # TODO: update the collision model with the new obstacle 
    #             has_changed = True # at least one obstacle has moved                
    #     return has_changed

    def add_robot_model(self, model):
        self.robot_models.append(model)
        self.robot_datas.append(model.createData())

    def add_collision_model(self, collision_model):
        self.collision_models.append(collision_model)
        self.collision_datas.append(collision_model.createData())

    def add_visual_model(self, visual_model):
        self.visual_models.append(visual_model)

    def add_self_collisions(self, model, collision_model):
        add_self_collisions(model, collision_model)

    def has_col_model_changed(self, collision_model):
        return False
    
    def add_roadmap(self,robot_config, env_obstacles: List[SimpleObstacle], add_self_col=True):
        model, collision_model, visual_model = self._init_models(robot_config, env_obstacles, add_self_col)
        self.add_robot_model(model)
        self.add_collision_model(collision_model)
        self.add_visual_model(visual_model)

        a_idx = len(self.robot_models) - 1
        self._add_env_obstacles(a_idx, env_obstacles)
        
        # roadmap options
        opts = PRMPlannerOptions( # was taken from franka example in pyroboplan as a standard config    
            max_step_size=0.05,
            max_neighbor_radius=3.14,
            max_neighbor_connections=15,
            max_construction_nodes=2000,
            construction_timeout=15.0,
            prm_star=True, # PRM* (not PRM)
        )
        # PRMPlanner object, contains the roadmap (and optional A* planner)
        planner = PRMPlanner(model, collision_model, options=opts)
        print("Constructing roadmap …")
        planner.construct_roadmap() # Making the graph Gi in C-space (i is the robot index)
        self.prm_planners.append(planner) 

        # Visualiser
        viz = MeshcatVisualizer(model, collision_model, visual_model)
        viz.initViewer(open=self.open_viz)
        viz.loadViewerModel()   
        self.vizs.append(viz)

    def _init_models(self, robot_config, env_obstacles: List[SimpleObstacle], add_self_col=True):
        if not robot_config.endswith('.urdf'):
            if robot_config == 'panda':
                model, collision_model, visual_model = load_models() # pyroboplan: load robot model, collision model, visual model ()
                
            else:
                model,collision_model, visual_model = None, None, None
                raise NotImplementedError(f"Loading {robot_config} is not implemented yet")

        else:
            model,collision_model, visual_model = None, None, None
            # TODO figure out how to load robot from urdf in pyroboplan / pinocchio
            # model, collision_model, visual_model = load_robot_from_urdf(robot)
            raise NotImplementedError(f"Loading robot from urdf is not implemented yet")


                
        if add_self_col:
            add_self_collisions(model, collision_model)
        
        return model, collision_model, visual_model
    
    def _add_env_obstacles(self, agent_idx, objects: List[SimpleObstacle], inflation_radius=0.0):
        """
        Adds spheres and cubes to the collision and visual models.

        Parameters
        ----------
        model : pinocchio.Model
            The robot model.
        collision_model : pinocchio.GeometryModel
            The collision geometry model.
        visual_model : pinocchio.GeometryModel
            The visual geometry model.
        objects : list of dict
            Each dict describes an object with keys:
                - type: "sphere" or "cube"
                - name: string identifier
                - position: [x, y, z] in meters
                - orientation: [w, x, y, z] quaternion
                - size:
                    * for sphere: radius
                    * for cube: [sx, sy, sz]
                - color: optional [r, g, b, a]
        inflation_radius : float, optional
            Extra radius (in meters) added around objects for collision inflation.
        """
        # model, collision_model, visual_model = self.robot_models[agent_idx], self.prp.collision_models[agent_idx], self.prp.visual_models[agent_idx]
        model, collision_model, visual_model = self.robot_models[agent_idx], self.collision_models[agent_idx], self.visual_models[agent_idx]
        
        # for i, obj in enumerate(objects):
        #     pos = np.array(obj.pos)
        #     quat = np.array(obj.quat)
        #     R = pin.Quaternion(quat[0], quat[1], quat[2], quat[3]).toRotationMatrix()
        #     placement = pin.SE3(R, pos)

        #     if obj.obj_type == "sphere":
        #         radius = obj.size + inflation_radius
        #         geom = coal.Sphere(radius)
        #     elif obj.obj_type == "cube":
        #         sx, sy, sz = obj.size, obj.size, obj.size
        #         geom = coal.Box(
        #             sx + 2.0 * inflation_radius,
        #             sy + 2.0 * inflation_radius,
        #             sz + 2.0 * inflation_radius,
        #         )
        #     else:
        #         raise ValueError(f"Unknown object type: {obj.obj_type}")
        #     obj_name = f'{obj.obj_type}_{i}'

        #     # geom_obj = pin.GeometryObject(obj_name, 0, placement, geom)
        #     geom_obj = pin.GeometryObject(
        #         obj_name, 
        #         0, # attach to universe joint
        #         geom, # geometry (Sphere, Box, etc.)
        #         placement # SE3 placement
        #         )


        #     # Optional color
        #     # if "color" in obj:
        #     geom_obj.meshColor =  np.array([0.0, 1.0, 0.0, 0.5]) # green color

        #     visual_model.addGeometryObject(geom_obj)
        #     collision_model.addGeometryObject(geom_obj)

        for i, obj in enumerate(objects):
            # ---- pose ----
            pos = np.asarray(obj.pos, dtype=float).reshape(3)
            quat = np.asarray(obj.quat, dtype=float).reshape(4)  # (w, x, y, z)

            # guard against zero/NaN quats
            qnorm = np.linalg.norm(quat)
            if not np.isfinite(quat).all() or qnorm == 0.0:
                quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            else:
                quat = quat / qnorm

            R = pin.Quaternion(quat[0], quat[1], quat[2], quat[3]).toRotationMatrix()
            placement = pin.SE3(R, pos)

            # ---- geometry ----
            if obj.obj_type == "sphere":
                radius = float(obj.size) + float(inflation_radius)
                geom = coal.Sphere(radius)
            elif obj.obj_type == "cube":
                # cube → equal side lengths
                s = float(obj.size)
                geom = coal.Box(
                    s + 2.0 * float(inflation_radius),
                    s + 2.0 * float(inflation_radius),
                    s + 2.0 * float(inflation_radius),
                )
            else:
                raise ValueError(f"Unknown object type: {obj.obj_type}")

            obj_name = f"{obj.obj_type}_{i}"

            # ---- add to models (placement BEFORE geometry, like your working code) ----
            visual_obj = pin.GeometryObject(obj_name, 0, placement, geom)
            visual_obj.meshColor = np.array([0.0, 1.0, 0.0, 0.5], dtype=float)
            visual_model.addGeometryObject(visual_obj)

            collision_obj = pin.GeometryObject(obj_name, 0, placement, geom)
            collision_model.addGeometryObject(collision_obj)
 
 
    def verified_for_search(self,agent_idx,q_start,q_goal):
        """
        inspired by PRPPlanner.plan() which does that before planning, we just ommitted the planning part
        """
        # Check start and end pose collisions.
        robot_model = self.robot_models[agent_idx]
        collision_model = self.collision_models[agent_idx]
        robot_data = self.robot_datas[agent_idx]
        collision_data = self.collision_datas[agent_idx] # TODO: check if this is needed
        prmp = self.prm_planners[agent_idx]


        if check_collisions_at_state(
            robot_model, collision_model, q_start, robot_data, collision_data
        ):
            print("Start configuration in collision.")
            return False
        if check_collisions_at_state(
            robot_model, collision_model, q_goal, robot_data, collision_data
        ):
            print("Goal configuration in collision.")
            return False

        # Ensure the start and goal nodes are in the graph.
        start_node = Node(q_start)
        prmp.graph.add_node(start_node)
        goal_node = Node(q_goal)
        prmp.graph.add_node(goal_node)
        
        exception = None
        try:
            # Ensures the start and goal configurations can be connected to the PRM.
            ans = prmp.connect_planning_nodes(start_node, goal_node) # True <=> start and goal can be connected to the PRM (meaning that a path exists between them)
        except Exception as e:
            exception = e
        
        finally: # remove the start and goal nodes from the graph (return graph to its original state)
            prmp.graph.remove_node(start_node)
            prmp.graph.remove_node(goal_node)
            if exception is not None:
                raise exception
        return ans
    
    def plan_path(self,agent_idx,q_start,q_goal):
        """
        """
        pass
    
def main(robot_config='panda',run_isaac=False):

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="", help="USD stage path")
    parser.add_argument("--headless", action="store_true")

    agent_cfg_paths = ['curobo/src/curobo/content/configs/robot/franka.yml']
    agent_cfgs = [load_yaml(agent_cfg_path)["robot_cfg"] for agent_cfg_path in agent_cfg_paths]
    ee_names = [agent_cfg["kinematics"]["ee_link"] for agent_cfg in agent_cfgs]
    
    n_agents = len(agent_cfg_paths)
    
    args = parser.parse_args()

    pi_bridge = PRPIsaacBridge()
    prp = pi_bridge.prp
    isaac = pi_bridge.isaac
    # Isaac Sim app start
    if run_isaac:
        isaac.run_app(args.headless) # start the app
        # Environment setup
        world = isaac.init_world()
        if Path(args.stage).is_file(): # load stage from usd (optional)
            world.scene.add_reference_to_stage(args.stage)
        world.step(render=False) # step to load the stage
        
    # add obstacles to the world

    objects = [['cube',[0,0,0],0.3], ['sphere',[1,0,0],0.5]]
    simple_objects = [SimpleObstacle(*x)  for x in objects] # simple representation of obstacles 
    if run_isaac:
        isaac.add_objects(simple_objects)
    
    for a_idx in range(n_agents):        
        prp.add_roadmap(robot_config,env_obstacles=simple_objects,add_self_col=True)
        if run_isaac:
            isaac.add_robot(a_idx=a_idx,robot_cfg_path=agent_cfg_paths[a_idx])
        
    
    
        
        # world.step(render=True)
        # for col_model in pi_bridge.prp.collision_models:
        #     model_changed = pi_bridge.prp.has_col_model_changed(col_model)
        #         for i in range(n_robots):

    while True:
        if run_isaac:
            if isaac.simulation_app.is_running():
                isaac.world.step(render=True)
            else:
                break

        for a_idx in range(n_agents):
            q_start = get_random_collision_free_state(prp.robot_models[a_idx], prp.collision_models[a_idx]) # should be the start state of the task (the joint angles, will be taken from robot joint state)
            q_goal  = get_random_collision_free_state(prp.robot_models[a_idx], prp.collision_models[a_idx]) # TODO: replace by task space target and then IK to get C-space target
            prp.vizs[a_idx].display(q_start)

            print("Planning …") 
            path = prp.prm_planners[a_idx].plan(q_start, q_goal) # Veifies start and goal are collision free, then verify they are at the same connected component, then runs  A* search
            if not path:
                print("Path not found – growing roadmap and retrying …")
                prp.prm_planners[a_idx].construct_roadmap() # make PRM 
                continue # Try again to plan

            prp.prm_planners[a_idx].visualize(prp.vizs[a_idx],ee_names[a_idx], show_path=True, show_graph=False)

            input("Press Enter to execute path …")
            for q in path:
                print(q)
                prp.vizs[a_idx].display(q)
                time.sleep(0.04)
                
                # world.step(render=True)

            if input("Another query? [y/N]: ").lower() != "y":
                break
            


    if run_isaac:
        isaac.close_app()


if __name__ == "__main__":
    main()
