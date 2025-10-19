# --- CUDA (Conda "targets" layout) for FlashInfer/SGLang ---

# 0) clean up any leftovers from previous sessions (optional but helps avoid dupes)
unset CUDA_HOME
unset LDFLAGS LIBRARY_PATH LD_LIBRARY_PATH CPATH CPLUS_INCLUDE_PATH

# 1) root of the Conda CUDA toolkit
export CUDA_HOME="$CONDA_PREFIX/targets/x86_64-linux"

# 2) tool binaries (nvcc) + NVVM device compiler (cicc)
export PATH="$CUDA_HOME/bin:$CUDA_HOME/nvvm/bin:$PATH"

# 3) headers (so torch cpp extensions see CUDA_HOME/include)
export CPATH="$CUDA_HOME/include:${CPATH}"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/include:${CPLUS_INCLUDE_PATH}"

# 4) link-time search paths — put the *correct* dirs FIRST (lib + stubs),
#    then any lib64 variants in case something expects them.
export LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib/stubs:$CUDA_HOME/lib64:$CUDA_HOME/lib64/stubs:${LIBRARY_PATH}"

# 5) run-time loader paths (also include nvvm libs)
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib/stubs:$CUDA_HOME/lib64:$CUDA_HOME/lib64/stubs:$CUDA_HOME/nvvm/lib64:${LD_LIBRARY_PATH}"

# 6) make the linker prefer the right dirs and embed rpaths so .so files load later
export LDFLAGS="-L$CUDA_HOME/lib -L$CUDA_HOME/lib/stubs -Wl,-rpath,$CUDA_HOME/lib ${LDFLAGS}"


# # cuda_env.sh — source this AFTER `conda activate markovian`

# # 1) Point CUDA to the Conda "targets" layout (CUDA 12+ on conda)
# export CUDA_HOME="$CONDA_PREFIX/targets/x86_64-linux"   # docs confirm this layout. :contentReference[oaicite:0]{index=0}

# # 2) Tool binaries (nvcc) + NVVM device compiler (cicc)
# export PATH="$CUDA_HOME/bin:$CUDA_HOME/nvvm/bin:$PATH"  # cicc lives under nvvm/bin. :contentReference[oaicite:1]{index=1}

# # 3) Headers for JIT/AOT builds (PyTorch cpp extensions expect CUDA_HOME/include)
# export CPATH="$CUDA_HOME/include:${CPATH}"
# export CPLUS_INCLUDE_PATH="$CUDA_HOME/include:${CPLUS_INCLUDE_PATH}"  # used by some build chains. :contentReference[oaicite:2]{index=2}

# # 4) Link-time search paths (so -lcudart / -lcuda resolve during FlashInfer’s link)
# #    Note: cudart is typically under .../lib (not lib64) in this layout; libcuda.so stub is in .../lib[/lib64]/stubs.
# export LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:$CUDA_HOME/lib/stubs:$CUDA_HOME/lib64/stubs:${LIBRARY_PATH}"  # :contentReference[oaicite:3]{index=3}

# # 5) Run-time search paths (loader finds cudart/nvvm)
# export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:$CUDA_HOME/lib/stubs:$CUDA_HOME/lib64/stubs:$CUDA_HOME/nvvm/lib64:${LD_LIBRARY_PATH}"  # :contentReference[oaicite:4]{index=4}

# # 6) Embed rpaths into built .so files (so they load without env vars at runtime)
# export LDFLAGS="-Wl,-rpath,$CUDA_HOME/lib:-rpath,$CUDA_HOME/lib64:-rpath,$CUDA_HOME/nvvm/lib64 ${LDFLAGS}"


# # Trying to fix stuff
# # /datastor1/tlake/miniconda3/envs/markovian/bin/../lib/gcc/x86_64-conda-linux-gnu/11.2.0/../../../../x86_64-conda-linux-gnu/bin/ld: cannot find -lcudart: No such file or directory
# # /datastor1/tlake/miniconda3/envs/markovian/bin/../lib/gcc/x86_64-conda-linux-gnu/11.2.0/../../../../x86_64-conda-linux-gnu/bin/ld: cannot find -lcuda: No such file or directory
# # collect2: error: ld returned 1 exit status
# # ninja: build stopped: subcommand failed.

# # Point CUDA to Conda’s targets layout
# export CUDA_HOME="$CONDA_PREFIX/targets/x86_64-linux"

# # Ensure toolchain + nvvm are visible (you already did this)
# export PATH="$CUDA_HOME/bin:$CUDA_HOME/nvvm/bin:$PATH"

# # Link-time & run-time search paths
# export LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib/stubs:$CUDA_HOME/lib64:$CUDA_HOME/lib64/stubs:${LIBRARY_PATH}"
# export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib/stubs:$CUDA_HOME/lib64:$CUDA_HOME/lib64/stubs:$CUDA_HOME/nvvm/lib64:${LD_LIBRARY_PATH}"

# # CRUCIAL: put the correct -L dirs *first* so they beat the lib64 ones
# export LDFLAGS="-L$CUDA_HOME/lib -L$CUDA_HOME/lib/stubs -Wl,-rpath,$CUDA_HOME/lib ${LDFLAGS}"
