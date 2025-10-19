# cuda_env.sh — source this AFTER `conda activate markovian`

# 1) Point CUDA to the Conda "targets" layout (CUDA 12+ on conda)
export CUDA_HOME="$CONDA_PREFIX/targets/x86_64-linux"   # docs confirm this layout. :contentReference[oaicite:0]{index=0}

# 2) Tool binaries (nvcc) + NVVM device compiler (cicc)
export PATH="$CUDA_HOME/bin:$CUDA_HOME/nvvm/bin:$PATH"  # cicc lives under nvvm/bin. :contentReference[oaicite:1]{index=1}

# 3) Headers for JIT/AOT builds (PyTorch cpp extensions expect CUDA_HOME/include)
export CPATH="$CUDA_HOME/include:${CPATH}"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/include:${CPLUS_INCLUDE_PATH}"  # used by some build chains. :contentReference[oaicite:2]{index=2}

# 4) Link-time search paths (so -lcudart / -lcuda resolve during FlashInfer’s link)
#    Note: cudart is typically under .../lib (not lib64) in this layout; libcuda.so stub is in .../lib[/lib64]/stubs.
export LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:$CUDA_HOME/lib/stubs:$CUDA_HOME/lib64/stubs:${LIBRARY_PATH}"  # :contentReference[oaicite:3]{index=3}

# 5) Run-time search paths (loader finds cudart/nvvm)
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:$CUDA_HOME/lib/stubs:$CUDA_HOME/lib64/stubs:$CUDA_HOME/nvvm/lib64:${LD_LIBRARY_PATH}"  # :contentReference[oaicite:4]{index=4}

# 6) Embed rpaths into built .so files (so they load without env vars at runtime)
export LDFLAGS="-Wl,-rpath,$CUDA_HOME/lib:-rpath,$CUDA_HOME/lib64:-rpath,$CUDA_HOME/nvvm/lib64 ${LDFLAGS}"
