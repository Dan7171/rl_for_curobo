
# -------------------
# Step 1: Pull/build 
# the base image*, (*base image =isaacsim 4.5.0 + ros2 + curobo (under /pkgs/curobo), but no rl_for_curobo installed),

# Select one of the following:
# -------------------

#Option 1: pull
# To use the pull option, keep this file unchanged, and run:
# cd rl_for_curobo && docker build -t curobo_isaac45v2 -f %path to this file% . 

FROM de257/curobo_isaac45@sha256:5f76f7fdf0a7caf1327eeb73392d7b490d851e008aacef14b4ae5d66209098f6
# If not working, try: FROM de257/curobo_isaac45:latest  (same image as the sha one)


# Option 2: build
# First, before runnign this script, run: 
# cd rl_for_curobo && docker build -t curobo_isaac45 -f curobo/docker/isaac_sim_45.dockerfile .
# Then, comment out the line above (FROM...) and run:
# cd rl_for_curobo && docker build -t curobo_isaac45v2 -f %path to this file% .

# -------------------
# Step 2: Build the new image by running:
# -------------------

# Copy the rl_for_curobo repo to the container
COPY . /workspace/rl_for_curobo

# Set the working directory to the rl_for_curobo repo
WORKDIR /workspace/rl_for_curobo
# Install to omni_python's pip environment in editable mode
RUN $omni_python -m pip install -e .

# Install curobo pakages not from /pkgs/curobo (original unmodified curobo package)
# But from /workspace/rl_for_curobo/curobo (modified curobo package)
WORKDIR /workspace/rl_for_curobo/curobo
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0
RUN $omni_python -m pip install -e . --no-build-isolation -y

# Last, set the working directory to start the container in it
WORKDIR /workspace/rl_for_curobo

# Now you can run the container,
# using curobo/docker/start_docker_isaac_sim45root.sh pointing the image name and image tag to the new image you just built (modify the script if needed)
