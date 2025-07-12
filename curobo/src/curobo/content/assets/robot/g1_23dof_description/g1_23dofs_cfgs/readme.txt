1. files under lula_robot_description_editor = upper body sphere placement as demonstrated in https://curobo.org/tutorials/1_robot_configuration.html
2. official 23 dofs urdf from unitree github: taken from https://github.com/unitreerobotics/unitree_ros/tree/master/robots/g1_description 




common bug: if lula editor gives you errors which tell meshes are instancible,
"2025-07-12 08:55:17  [Warning] [isaacsim.robot_setup.xrdf_editor.extension] Could not generate spheres for any meshes in link /pelvis.  This is likely due to all meshes nested under /pelvis being instanceable"

run this in script editor (in isaac gui):

from omni.usd import get_context
from pxr import Usd
stage = get_context().get_stage()
for prim in stage.Traverse():
    if prim.GetPath().pathString.startswith("/World/g1_23dof") and prim.IsInstanceable():
        print(f"Deinstancing: {prim.GetPath()}")
        prim.SetInstanceable(False)