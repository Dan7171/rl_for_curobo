TLDR:
1. USD.Prim = ANYTHING (any node in the scene (mesh, light, camera, robot, cube, pointer to another scene and any other node  you can think of in the scene graph))
2. UsdGeom.Xformable = ANYTHING THAT CAN MOVE (Xformable is the base class for all prims that can have transformation ops (like translate, rotate, scale).)
3. UsdGeom.XformCommonAPI = SIMPLIFIED WRAPPER OVER UsdGeom.Xformable 
4. isaacsim.core.prims.XformPrim is a wrapper instead using UsdGeom directory (https://docs.robotsfan.com/isaacsim/latest/py/source/extensions/isaacsim.core.prims/docs/index.html#isaacsim.core.prims.XFormPrim)



1. USD.Prim
What is a Prim?
In USD (Universal Scene Description), a Prim (short for "primitive") is the basic building block of a scene. It represents anything in the scene graph, like:
a mesh
a light
a camera
a transform node
a group
a reference to another USD file
A prim is like a node in a hierarchy tree. But by default, it’s just a container — it doesn’t do anything unless it's typed and schema-augmented.


2. UsdGeom.Xformable:
is the base class for all prims that can have transformation ops (like translate, rotate, scale).

3. UsdGeom.XformCommonAPI:

This is a simplified wrapper over Xformable that assumes: One translate op One rotate op One scale op In a specific order: S * R * T
