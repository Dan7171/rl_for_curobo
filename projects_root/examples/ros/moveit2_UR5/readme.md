fully based on https://www.youtube.com/watch?v=pGje2slp6-s 

# Step 0:
download (from here) https://drive.google.com/drive/folders/1Bl-ULA9X3lplRguwXPpz8ZI_T7EpyADH
and move them anywhere you want in the env with the ros2 humble and isaac sim 4.5 (I worked in docker)
I put mine under "/workspaces" (from curobo_isaac45v3 docker conatainer) directory (but I dont think it matters)...

# Step 1: TODO: summarize
Follow all steps up to ~10:30 in the youtube link...
most of them are implemented. 


# Step 2: run isaac sim (load usd from scene (fast) or import urdf and build the action graph yourself (slower))

1. terminal 1:
source /workspace/moveit2_UR5/install/setup.sh

ros2 launch ur5_moveit_config demo.launch.py
Meaning: 
    ros2 launch ur5_moveit_config = launch from the ros package we earlier built (workspace/moveit2_UR5/src/ur5_moveit_config)
    demo.launch.py: the script at the package src folder: workspace/moveit2_UR5/src/ur5_moveit_config/launch/demo.launch.py 

    



2. terminal 2:
/isaac-sim/isaacsim.sh


* tip to save a lot of time: just import the usd file from workspace/moveit2_UR5/ur5_moveit.usd and skip directly to "plan and execute" (skip the urdf loading, action graph making etc... it leads to the same result)
import urdf of the ur5 (toggle to fixed base) ur5.urdf
add ground plane (so robot wont fall)
move robot up 10 cm above ground
create>visual scripting>action graph
edit action graph
add nodes as shown here: https://youtu.be/pGje2slp6-s?t=788
set the base link as articulation root:
right click on base_link in stage > physics > articulation root

# set the ros topics from the xacro of the robot to the action graph:
1. take the topic name from here: 
    workspace/moveit2_UR5/src/ur5_moveit_config/config/ur5.ros2_control.xacro
the topic name is "isaac_joint_commands" (row 11)
copy it to the node "ros 2 subscribe joint state" in the action graph under "inputs">"topicName"

2. similarly to the Ros2 publish joint state node in the action graph,
copy (row 12) the topic name "isaac_joint_states" to the "inputs - topicName" of this graph node...

3. Select the isaac" acticulation controller node" (omni graph node, most right) and set the "target prim to be /World/ur5/base_link (thats the articulation root we defined before)

# click play and run simulation, make sure no errors in the log
* if joints has errors related to inifinite joint limit, set them to -360 and 360 as in the video.
* see "trick" in ~17:00 (adress close loop control problem)


# Plan and execute:
root@de:/# source /workspace/moveit2_UR5/install/setup.sh 
root@de:/# ros2 launch ur5_moveit_config demo.launch.py 
should see both in rviz and isaac sim...


