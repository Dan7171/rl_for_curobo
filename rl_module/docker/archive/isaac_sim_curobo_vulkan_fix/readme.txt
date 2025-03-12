In short: This is original dockerfile from curobo repository (making a 30gb docker image with curobo and isaac-sim4.0.0), except a fix in the row: ARG VULKAN_SDK_VERSION= to be VULKAN_SDK_VERSION=1.3.236.0

Including:
1. isaac-sim v4.0.0 (the matching version to curobo)
2. curobo v0.7.6 (the matching version to curobo. (/rl_for_curobo/isaac_sim-4.0.0/python.sh -c "import curobo; print(curobo.__version__) -> 0.7.6"
0.7.6) 
3. bugfix to VULKAN_SDK_VERSION paramter in this dockerfile(See above). source:   https://github.com/NVlabs/curobo/issues/413 