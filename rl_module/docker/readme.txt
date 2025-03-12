
# 1. Files:

## 1.1 The file ./host/start_docker_modified.sh (a modified version of /rl_for_curobo/curobo/docker/start_docker.sh) 
#### changes w.r. to /rl_for_curobo/curobo/docker/start_docker.sh:
    a. hard coded image parameter with the command " input_arg="isaac_sim_4.0.0" "
    b. mounting the host home dir ("~") on the container (to "/host") with the command "-v ~:/host"
    c. running from within the container ("guest") the script "startup.sh" with the command "-c "source /host/rl_for_curobo/rl_module/docker/guest/startup.sh && bash""

## 1.2 The file ./guest/startup.sh



# 2. Running instructions:

# In host:
cp ~/rl_for_curobo/rl_module/docker # go inside the directory of the modified "docker run" script
bash ./host/start_docker_modified.sh # run the modified "docker run" script (the "isaac_sim_4.0.0" was moved inside the script) 

# In guest:
root@dan-US-Desktop-Codex-R:/pkgs/curobo# omni_python %PATH/TO/YOUR/PYTHON/FILE.py%
## Examples: 
## RUN PYTHON SCRIPT FROM HOST FILES (MOUNTED IN THE MODIFIED DOCKER RUN SCRIPT): omni_python /host/rl_for_curobo/curobo/examples/isaac_sim/mpc_example.py
## RUN PYTHON SCRIPT FROM GUEST FILES: omni_python ./examples/isaac_sim/mpc_example.py
