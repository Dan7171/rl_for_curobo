"""
Resources:

https://docs.omniverse.nvidia.com/materials-and-rendering/latest/templates/parameters/OmniPBR_Albedo.html
https://docs.omniverse.nvidia.com/simready/latest/simready-asset-creation/material-best-practices.html
https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/isaacsim.core.api/docs/index.html#isaacsim.core.api.materials.OmniPBR
https://docs.omniverse.nvidia.com/materials-and-rendering/latest/materials.html 
https://docs.omniverse.nvidia.com/materials-and-rendering/latest/materials_templates.html 


Key concepts:

MDL:
NVIDIA Material Definition Language (MDL) https://docs.omniverse.nvidia.com/materials-and-rendering/latest/materials.html

PBR:
Physically-Based Rendering. A PBR material correctly handles how energy is absorbed, colored, emitted, and reflected by the surface.
Physically Based Rendering (PBR): MDL supports physically based rendering principles, simulating the behavior of light to achieve realistic material representation.


Visual Material types:

1. UsdPreviewSurface: https://docs.omniverse.nvidia.com/materials-and-rendering/latest/templates/UsdPreviewSurface.html#usdpreviewsurface


2. OmniPBR: https://docs.omniverse.nvidia.com/materials-and-rendering/latest/templates/OmniPBR.html#
OmniPBR is the default Physically based material available in Omniverse. This material can describe most opaque dielectric or non-dielectric materials.

3. OmniGlass:
OmniGlass is an improved physical glass model that simulates light transmission through thin walled and transmissive surfaces.
https://docs.omniverse.nvidia.com/materials-and-rendering/latest/templates/OmniGlass.html


🧭 TL;DR – Which to choose?
Shader Type	Use Case	Portable USD	Visual Fidelity	Transparency
UsdPreviewSurface	General PBR material for compatibility	✅ Yes	🔹 Basic	❌ Limited
OmniPBR	Realistic materials in Omniverse/Isaac Sim	❌ No	✅ High	✅ Yes (some)
OmniGlass	Transparent/glass-like objects	❌ No	✅ High	✅ Best



Key notes: (https://docs.omniverse.nvidia.com/simready/latest/simready-asset-creation/material-best-practices.html):
 - Materials used in a simulation application should use OmniPBR and OmniGlass for best compatibility

 
 
"""

from isaacsim.core.api.materials import OmniPBR, OmniGlass, PreviewSurface
from pxr import Usd, UsdShade, UsdGeom

def bind_material_recursively(stage, root_prim_path: str, material_path: str):
    """
    Bind a material to all geometry (Gprim) prims under the root_prim_path, including the root itself.
    """
    root_prim = stage.GetPrimAtPath(root_prim_path)
    if not root_prim:
        print(f"[ERROR] Prim {root_prim_path} does not exist.")
        return

    material = UsdShade.Material.Get(stage, material_path)
    if not material:
        print(f"[ERROR] Material {material_path} does not exist.")
        return

    for prim in stage.Traverse():
        if not prim.GetPath().IsPrefixedBy(root_prim.GetPath()):
            continue
        if prim.IsA(UsdGeom.Gprim):
            UsdShade.MaterialBindingAPI(prim).Bind(material)
            print(f"[INFO] Bound material to: {prim.GetPath()}")

def unbind_material_recursively(stage, root_prim_path: str):
    """
    Unbind any materials from all geometry (Gprim) prims under the root_prim_path, including the root itself.
    """
    root_prim = stage.GetPrimAtPath(root_prim_path)
    if not root_prim:
        print(f"[ERROR] Prim {root_prim_path} does not exist.")
        return

    for prim in stage.Traverse():
        if not prim.GetPath().IsPrefixedBy(root_prim.GetPath()):
            continue
        if prim.IsA(UsdGeom.Gprim):
            UsdShade.MaterialBindingAPI(prim).UnbindDirectBinding()
            print(f"[INFO] Unbound material from: {prim.GetPath()}")





def create_material(path: str, shader_type="OmniPBR", inputs=None):
    """
    Creates a material using Isaac Sim's high-level material classes.

    Parameters:
    - path (str): Full USD path where the material will be created, e.g., "/World/Looks/MyMaterial"
    - shader_type (str): "OmniPBR", "OmniGlass", or "PreviewSurface"
    - inputs (dict): Dictionary of attribute names and values to set on the material

    Returns:
    - material instance (OmniPBR, OmniGlass, or PreviewSurface)
    """
    material_cls = {
        "OmniPBR": OmniPBR,
        "OmniGlass": OmniGlass,
        "PreviewSurface": PreviewSurface,
    }.get(shader_type)

    if material_cls is None:
        raise ValueError(f"Unknown shader type: {shader_type}")

    material = material_cls(prim_path=path)

    if inputs:
        for attr_name, value in inputs.items():
            setattr(material, attr_name, value)

    return material


if __name__ == "__main__":
    material = create_material(
    "/World/Looks/SimpleRed",
    shader_type="OmniPBR",
    inputs={
        "diffuse_color_constant": (1.0, 0.0, 0.0),  # red
        "metallic_constant": 0.1,
        "roughness_constant": 0.3,
    }
)