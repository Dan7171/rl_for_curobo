from curobo.geom.types import WorldConfig, Cuboid, Mesh, Capsule, Cylinder, Sphere
from curobo.util_file import get_assets_path, join_path

obstacle_1 = Cuboid(
     name="cube_1",
     pose=[0.0, 0.0, 0.0, 0.043, -0.471, 0.284, 0.834],
     dims=[0.2, 1.0, 0.2],
     color=[0.8, 0.0, 0.0, 1.0],
 )

# describe a mesh obstacle
# import a mesh file:

mesh_file = join_path(get_assets_path(), "scene/nvblox/srl_ur10_bins.obj")

obstacle_2 = Mesh(
   name="mesh_1",
   pose=[0.0, 2, 0.5, 0.043, -0.471, 0.284, 0.834],
   file_path=mesh_file,
   scale=[0.5, 0.5, 0.5],
)

obstacle_3 = Capsule(
   name="capsule",
   radius=0.2,
   base=[0, 0, 0],
   tip=[0, 0, 0.5],
   pose=[0.0, 5, 0.0, 0.043, -0.471, 0.284, 0.834],
   color=[0, 1.0, 0, 1.0],
)

obstacle_4 = Cylinder(
   name="cylinder_1",
   radius=0.2,
   height=0.1,
   pose=[0.0, 6, 0.0, 0.043, -0.471, 0.284, 0.834],
   color=[0, 1.0, 0, 1.0],
)

obstacle_5 = Sphere(
   name="sphere_1",
   radius=0.2,
   pose=[0.0, 7, 0.0, 0.043, -0.471, 0.284, 0.834],
   color=[0, 1.0, 0, 1.0],
)

world_model = WorldConfig(
   mesh=[obstacle_2],
   cuboid=[obstacle_1],
   capsule=[obstacle_3],
   cylinder=[obstacle_4],
   sphere=[obstacle_5],
)

# assign random color to each obstacle for visualization
world_model.randomize_color(r=[0.2, 0.7], g=[0.8, 1.0])

file_path = "debug_mesh.obj"
world_model.save_world_as_mesh(file_path)