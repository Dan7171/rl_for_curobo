

# Run basic examples (test ros2 bridge + rclpy)

## What to run? 
## 1. In host:

```bash
rl_for_curobo$ ./curobo/docker/start_docker_isaac_sim45rootv2.sh
```

What happens when you run it?

1. a new container of the image curobo_isaac45v2 starts. (image with ubuntu 22.04, isaacsim 4.5 (with python 3.10 interpreter named omni_python on it which we'll use), ros2 humble (supports python 3.10))
2. it starts with running:
    "
    docker run 
    ...
    --entrypoint bash # means we enter bash in the container...
    ...

    CONTAINER_REGISTRY/{IMAGE_NAME}:$IMAGE_TAG} \
    -c "cd $REPO_PATH_CONTAINER && \
    /isaac-sim/python.sh -m pip uninstall -y rl_for_curobo && \
    /isaac-sim/python.sh -m pip install -e REPO_PATH_CONTAINER && \
    source /opt/ros/humble/setup.sh && \
    DOCKER_CMD"
    "
    which means: 
        run the next commands: 
            # part 1 below is irrelevant for our ros2 example, can be removed
            
            1. cd to the mounted container of this repository in container , uninstall and install the module of the repo.
            
            #  part 2 is super relevant to us!!, sourcing ros2 is required before launching standalone example
            
            2. source ros2 huble in terminal (source /opt/ros/humble/setup.sh)
            
            # part 3 is just enter bash terminal
            
            3. run a command (DOCKER_CMD). This is optional, currently just running 'bash' meaning it stays in terminal.

## 2. In container:

```bash
root@de:/workspace/rl_for_curobo# omni_python /isaac-sim/standalone_examples/api/isaacsim.ros2.bridge/subscriber.py 

```
- what happens when you run it?

'omni_python' is the isaac sim python env (alias to /isaac-sim/python.sh). Note that its already configured to run our rl_for_curobo env too, but in this example we are not using it at all!

and '/isaac-sim/standalone_examples/api/isaacsim.ros2.bridge/subscriber.py' is the absolute path of the example we run.
To make sure we did it correctly, we should see on screen:

for host command:

```bash
dan@de:~/rl_for_curobo$ curobo/docker/start_docker_isaac_sim45rootv2.sh 
--------------------------------
Docker run (as root) script started!
--------------------------------
using depth camera, you can now run examples like omni_python /workspace/rl_for_curobo/curobo/examples/isaac_sim/realsense_reacher.py
Setting up X11 forwarding...
non-network local connections being added to access control list
Found existing installation: rl_for_curobo 0.1.0
Uninstalling rl_for_curobo-0.1.0:
  Successfully uninstalled rl_for_curobo-0.1.0
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv
Obtaining file:///workspace/rl_for_curobo
  Preparing metadata (setup.py) ... done
Installing collected packages: rl_for_curobo
  Attempting uninstall: rl_for_curobo
    Found existing installation: rl_for_curobo 0.1.0
    Can't uninstall 'rl_for_curobo'. No files were found to uninstall.
  Running setup.py develop for rl_for_curobo
Successfully installed rl_for_curobo-0.1.0
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv

[notice] A new release of pip is available: 24.0+nv1 -> 25.1.1
[notice] To update, run: /isaac-sim/kit/python/bin/python3 -m pip install --upgrade pip

```

For guest command:

```bash
Simulation App Startup Complete
[ext: isaacsim.ros2.bridge-4.1.15] startup
Attempting to load system rclpy
rclpy loaded

```



## Summary:
Things to remember:
1. the example shows running of ros2 bridge with rclpy using an image with:
    a. isaac sim 4.5 (omni_python with python version 3.10 on it) 
    b. sourced ros2 humble in terminal that runs command! (we ran source /opt/ros/humble/setup.sh )
    The ros2 version must match the isaac sim version and ubuntu version! (humble & 4.5 & 22.04 match!)
    Only then, it could worl properly!
2. To make sure ros2 is soured, in guest terminal run ros2 --help (or any other comamnd of ros) and see if it knows it... 
root@de:/workspace/rl_for_curobo# ros2 --help
