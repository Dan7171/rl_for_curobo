# Running helpful commands on guest (container) before starting to run isaac-sim and curobo files

export DISPLAY=:0
echo "- display exporting: configued successfully V"

cp /host/rl_for_curobo//isaac_sim-4.0.0/exts/omni.isaac.nucleus/omni/isaac/nucleus/nucleus.py ../../isaac-sim/exts/omni.isaac.nucleus/omni/isaac/nucleus/nucleus.py 
echo "- fast boot of isaac-sim 4.0.0: configued successfully V" # for more info see https://forums.developer.nvidia.com/t/detected-a-blocking-function-this-will-cause-hitches-or-hangs-in-the-ui-please-switch-to-the-async-version/271191/12#:~:text=I%20found%20a,function%20to%20below
echo "- ready to execute curobo simulations! (example: root/@USERNAME/pkgs/curoboomni_python ./examples/isaac_sim/mpc_example.py)"
