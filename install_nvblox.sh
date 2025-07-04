# 0. (Only once) start a clean terminal and activate the env
source ~/anaconda3/etc/profile.d/conda.sh
conda activate isaac-sim            # prompt changes to (isaac-sim)

# 1. Fix the Thrust header inside the env (one-time patch)
sed -i 's/^#define THRUST_VERSION *\([0-9]*\).*/#define THRUST_VERSION \1/' \
    "$CONDA_PREFIX/include/thrust/version.h"

# 2. Build & install nvblox  (still inside the env!)
export CUDACXX=/usr/bin/nvcc
export PATH=/usr/bin:$PATH
cd ~/rl_for_curobo
git clone https://github.com/valtsblukis/nvblox.git   # skips if already cloned
cd nvblox/nvblox
rm -rf build && mkdir build && cd build

cmake .. \
  -DPRE_CXX11_ABI_LINKABLE=ON \
  -DBUILD_TESTING=OFF \
  -DGLOG_INCLUDE_DIR="/usr/include" \
  -DGLOG_INCLUDE_DIRS="/usr/include" \
  -DGLOG_LIBRARY="/usr/lib/x86_64-linux-gnu/libglog.so" \
  -Dgflags_INCLUDE_DIR="/usr/include" \
  -Dgflags_INCLUDE_DIRS="/usr/include" \
  -Dgflags_LIBRARIES="/usr/lib/x86_64-linux-gnu/libgflags.so" \
  -DCUDA_nvToolsExt_INCLUDE_DIR="/usr/include" \
  -DCUDA_nvToolsExt_LIBRARY="/usr/lib/x86_64-linux-gnu/libnvToolsExt.so" \
  -DCMAKE_CUDA_FLAGS="-I/usr/include" \
  -DCMAKE_CXX_FLAGS="-I/usr/include" \
  || exit 1

make -j$(nproc) || exit 1
sudo make install || exit 1           # puts libnvblox.so under /usr/local/lib

# 3. Install the Python wrapper nvblox_torch (still in the same env)
cd ~/rl_for_curobo
git clone https://github.com/NVlabs/nvblox_torch.git   # if not already there
cd nvblox_torch
chmod +x install.sh
./install.sh "$(python - <<PY
import torch.utils, sys; print(torch.utils.cmake_prefix_path)
PY
)" || exit 1
pip install -e . || exit 1

# 4. Quick sanity check
python - <<'PY'
import nvblox_torch as nt
print("nvblox_torch version:", nt.__version__)
PY