"""
Create a text prim in Isaac Sim
"""

# third party modules
import numpy as np
from typing import Optional

# Isaac Sim app initiation and isaac sim modules
from projects_root.utils.issacsim import init_app, wait_for_playing
simulation_app = init_app({
    "headless": False,
    "width": "1920",
    "height": "1080"
})

# Isaac Sim modules (after simulation_app initialization)
from omni.isaac.core import World
from pxr import Gf, UsdGeom, UsdShade, Sdf

def create_text(text="Hello", position=(0, 0, 0)):
    # Get stage
    stage = omni.usd.get_context().get_stage()
    
    # Create text prim
    text_path = "/World/Text"
    mesh = UsdGeom.Mesh.Define(stage, text_path)
    
    # Add text shader
    material = UsdShade.Material.Define(stage, text_path + "/Material")
    shader = UsdShade.Shader.Define(stage, text_path + "/Material/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((1, 1, 1))
    
    # Set text attributes
    text_attr = mesh.CreateAttribute("text", Sdf.ValueTypeNames.String)
    text_attr.Set(text)
    
    # Set transform
    xformable = UsdGeom.Xformable(mesh)
    xformable.AddTranslateOp().Set(Gf.Vec3d(*position))
    
    return text_path

# Initialize simulation
my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()

# Create text
text_prim = create_text("Hello", (0, 0, 1))

# Keep simulation running
while simulation_app.is_running():
    my_world.step(render=True)

simulation_app.close()