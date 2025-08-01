


1. start instance:
rl_for_curobo/curobo/docker/start_apptrainer_isaac_sim45v5user.sh

2. run instance:
apptainer shell instance://my_instance
(see instances:
[evrond@ise-4090-12 ~]$ apptainer instance list
INSTANCE NAME    PID       IP    IMAGE
my_instance      687696          /home/evrond/curobo_isaac45v5.sif

stop instance:
apptainer instance stop my_instance


)

3. to enter mounted repo-run:
Apptainer> cd /workspace/rl_for_curobo/


4. to push changes to remote:
do it from
Apptainer> cd /workspace/rl_for_curobo/
and not from host (host doesent have git-lfs but image does)





