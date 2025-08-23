"""Quick-start demo: builds two simple UR5e sphere models, collects sphere obstacles from an   
Isaac-Sim stage, constructs PRMs for each arm and prints basic statistics.  

Run inside Isaac-Sim python environment:  
$ ./python.sh drrt_star/run_two_arms_demo.py --headless  
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Isaac-Sim


# Our local modules
from drrt_star_old.spheres import Sphere
from drrt_star_old.prm_arm import ArmPRM




# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():

    # -----------------------------------------------------------------------------
    # Isaac-Sim helpers
    # -----------------------------------------------------------------------------

    def extract_env_spheres() -> List[Sphere]:
        """Collect simple spheres *and* approximate cubes as enclosing spheres."""
        stage = get_current_stage()
        prims = get_all_matching_child_prims("/World")
        spheres: List[Sphere] = []
        for prim in prims:
            typename = prim.GetTypeName()
            if typename == "Sphere":
                radius_attr = prim.GetAttribute("radius")
                r = radius_attr.Get() if radius_attr.IsValid() else 0.1
                p = np.array(prim.GetAttribute("xformOp:translate").Get())
                spheres.append(Sphere(center=p, radius=r))
            elif typename == "Cube":
                size_attr = prim.GetAttribute("size")
                s = size_attr.Get() if size_attr.IsValid() else 0.1
                r = np.sqrt(3) * s / 2  # circumscribed sphere
                p = np.array(prim.GetAttribute("xformOp:translate").Get())
                spheres.append(Sphere(center=p, radius=r))
        return spheres


    def build_link_spheres() -> dict[str, list[Sphere]]:
        """Very rough sphere set per link (UR5e). Centers are in *link* frame."""
        return {
            "base_link": [Sphere(np.array([0.0, 0.0, 0.05]), 0.10)],
            "shoulder_link": [Sphere(np.array([0.0, 0.0, 0.0]), 0.08)],
            "upper_arm_link": [Sphere(np.array([-0.2, 0.0, 0.0]), 0.08)],
            "forearm_link": [Sphere(np.array([-0.18, 0.0, 0.0]), 0.07)],
            "wrist_1_link": [Sphere(np.array([0.0, 0.0, -0.05]), 0.06)],
            "wrist_2_link": [Sphere(np.array([0.0, 0.0, -0.05]), 0.06)],
            "wrist_3_link": [Sphere(np.array([0.0, 0.0, -0.05]), 0.05)],
        }

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="", help="Path to USD stage to open")
    parser.add_argument("--headless", action="store_true")
    from omni.isaac.kit import SimulationApp
    args = parser.parse_args()
    simulation_app = SimulationApp({"headless": args.headless})
    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import get_current_stage
    from isaacsim.core.utils.prims import get_prim_at_path, get_all_matching_child_prims
    
    

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    # open stage if provided
    if Path(args.stage).is_file():
        world.scene.add_reference_to_stage(args.stage)

    # step once to instantiate
    world.step(render=False)

    env_spheres = extract_env_spheres()
    print(f"Found {len(env_spheres)} environment spheres")

    link_spheres = build_link_spheres()
    urdf_path = "/home/dan/rl_for_curobo/curobo/src/curobo/content/assets/robot/ur_description/ur5e.urdf"

    prm1 = ArmPRM(urdf_path, link_spheres, env_spheres, n_samples=150)
    prm2 = ArmPRM(urdf_path, link_spheres, env_spheres, n_samples=150)

    print("ArmPRM 1:", len(prm1.vertices), "vertices,", len(prm1.edges), "edges")
    print("ArmPRM 2:", len(prm2.vertices), "vertices,", len(prm2.edges), "edges")
   
    simulation_app.close()


if __name__ == "__main__":
    main()
