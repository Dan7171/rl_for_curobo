based on https://www.youtube.com/watch?v=pGje2slp6-s

running instructions:
# in host - terminal 1
# start the container (will be called 'cu_is45_v3', enter as root)
./curobo/docker/start_docker_isaac_sim45v3.sh
# in container (terminal 1):
cd /workspace

# in host-terminal 2:
docker ps (# verify the container name is cu_is45_v3)
docker exec -it cu_is45_v3 bash # eneter living container in interactive mode, enter bash


# in container we now run isaac sim (terminal 2):
root@de:~/rl_for_curobo# source /opt/ros/humble/setup.bash # (echo $SHELL) this is crucial for ros2 bridge when running isaac
root@de:~/rl_for_curobo# /isaac-sim/isaac-sim.sh # run isaac sim
# in gui - File>open> /workspace/moveit2_UR5/ur5_moveit.usd # open usd

# finally
click play in isaac sim
and in rviz set targets to the robot and execut  

# terminal 3:
# enter docker container as did in terminal 2
# run projects_root/ros/workspaces/isaac_moveit_rviz_ur5_example/src/send_goal_pose_to_moveit.py