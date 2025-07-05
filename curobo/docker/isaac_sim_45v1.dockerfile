########################
# This docker file builds:
# Base image: de257/curobo_isaac45
# Installing modules - rl_for_curobo (and curobo in rl_for_curobo) in omni_python environment

# Note! 
# This docker file is *aimed* to reproduce the making of image: 
# de257/curobo_isaac45:v1_rl_for_curobo_module_installed
# sha256:bc99ed7e4ea3af807c8743d25ce7cba139b3c9c240b45ceec77b5511a81ea26a 
# https://hub.docker.com/repository/docker/de257/curobo_isaac45/tags/v1_rl_for_curobo_module_installed/sha256:bc99ed7e4ea3af807c8743d25ce7cba139b3c9c240b45ceec77b5511a81ea26a?tab=layers

# But its not guaranteed to be identical, because the image is built on top of the original image, and the original image is built on top of the base image.
# And the image mentioned above is was made by docker commit (and not by docker build as in the original docker file)


# YOU NOW HAVE 3 OPTIONS- Currently you are in option 2

# - Option 1: Pull the image from the registry (totally building 0 images)
# To do the direct pull without building the image, use, just comment out every line in this file, and uncomment the line below:
# FROM de257/curobo_isaac45@sha256:bc99ed7e4ea3af807c8743d25ce7cba139b3c9c240b45ceec77b5511a81ea26a


# - Option 2: Pulling base and building new (totally building 1 image)
# e.g. Pulling the base image only: the version that we built in curobo/docker/isaac_sim_45.dockerfile
FROM de257/curobo_isaac45@sha256:5f76f7fdf0a7caf1327eeb73392d7b490d851e008aacef14b4ae5d66209098f6


# Option 3: Building both base and new image (totally building 2 images)
# You can also build the image from this docker file, by running: "cd rl_for_curobo && docker build -t curobo_isaac45 -f curobo/docker/isaac_sim_45.dockerfile"
# After that, replace the FROM line of option 2 with: 'FROM curobo_isaac45' (the image you just built)


# If you build the new image (e.g. option 2 or option 3) keep next lines not commented out
RUN cd / && git clone https://github.com/Dan7171/rl_for_curobo.git
RUN cd /rl_for_curobo && pip install -e .
RUN cd /rl_for_curobo/curobo && SETUPTOOLS_SCM_PRETEND_VERSION=1.0.0 omni_python -m pip install -e .[isaacsim] --no-build-isolation








