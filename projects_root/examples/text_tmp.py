from projects_root.utils.issacsim import init_app, wait_for_playing
simulation_app = init_app({
    "headless": False,
    "width": "1920",
    "height": "1080"
})

from pxr import UsdGeom, Gf
import omni.usd
import omni.kit.commands
import carb
import numpy as np
from PIL import Image, ImageDraw, ImageFont

stage = omni.usd.get_context().get_stage()

# 1. Create a cube
cube_path = "/World/TextCube"
UsdGeom.Cube.Define(stage, cube_path)

# 2. Create a material and assign to the cube
# material_path = "/World/TextCube/Material"
# omni.kit.commands.execute(
#     "CreateMdlMaterialPrim",
#     mtl_url="omniverse://localhost/NVIDIA/Materials/Base/Props/Plastic.mdl",
#     mtl_name="Plastic",
#     prim_path=material_path,
# )
# omni.kit.commands.execute(
#     "BindMaterial",
#     prim_path=cube_path,
#     material_path=material_path,
# )

material = OmniPBR(
    base_color=(1.0, 1.0, 0.0, 1.0),
    roughness=0.5,
    metallic=0.5,
)

# 3. Function to create an image from text and upload as texture
def set_cube_text(text, material_path=material_path):
    # Create image with text
    img = Image.new("RGBA", (256, 256), (0, 0, 0, 255))
    draw = ImageDraw.Draw(img)
    # You can specify a TTF font file if you want
    # font = ImageFont.truetype("arial.ttf", 40)
    font = ImageFont.load_default()
    draw.text((10, 100), text, font=font, fill=(255, 255, 0, 255))
    img_np = np.array(img)

    # Upload to texture
    import omni.kit.pipapi
    import omni.kit.material.library as matlib
    matlib.set_material_texture_from_numpy(
        stage, material_path, "diffuse", img_np
    )

# 4. Example: Set initial text
set_cube_text("Hello, Isaac Sim!")

# 5. In your control loop, call set_cube_text("New Text") to update

# Save the stage (optional)
stage.GetRootLayer().Export("/home/dan/rl_for_curobo/projects_root/examples/my_scene.usd")

simulation_app.update()
input("Press Enter to exit...")
simulation_app.close()