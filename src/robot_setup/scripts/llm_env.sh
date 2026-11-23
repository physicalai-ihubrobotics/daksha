# shellcheck shell=bash
# Sourced by install_llm_agent.sh and start_llm_server.sh (not run directly).
#
# The Jetson Thor torch wheel links against NVPL BLAS/LAPACK and cuDSS, which
# come as pip wheels (nvpl-blas, nvpl-lapack, nvidia-cudss-cu13<1: torch links
# libcudss.so.0, cuDSS 1.x ships .so.1) but live
# outside torch/lib, the only directory torch's RUNPATH searches. Put their
# lib directories on LD_LIBRARY_PATH.
#
#   llm_lib_path <venv>

THOR_EXTRA_PKGS=(nvpl-blas nvpl-lapack "nvidia-cudss-cu13<1")
# huggingface_hub (httpx2) needs brotli>=1.2; the apt python3-brotli is 1.1.
LLM_COMMON_PKGS=(huggingface_hub "brotli>=1.2")

llm_lib_path() {
    local venv="$1" site dirs
    site="$("$venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    dirs="$(find "$site/nvpl" "$site/nvidia" -maxdepth 4 \
                \( -name 'libnvpl_*.so*' -o -name 'libcudss.so*' \) \
                -printf '%h\n' 2>/dev/null | sort -u | paste -sd:)"
    if [ -n "$dirs" ]; then
        export LD_LIBRARY_PATH="$dirs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    # Triton JIT-compiles a C helper that includes cuda.h; the Jetson triton
    # wheel doesn't bundle CUDA headers, so use the system toolkit's.
    local cuda="${CUDA_HOME:-/usr/local/cuda}"
    if [ -f "$cuda/include/cuda.h" ]; then
        export CUDA_HOME="$cuda"
        export C_INCLUDE_PATH="$cuda/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
    fi
    # Triton's bundled ptxas is CUDA 12.9 and rejects Thor's sm_110a; the
    # system CUDA 13 ptxas supports it.
    if [ -x "$cuda/bin/ptxas" ]; then
        export TRITON_PTXAS_PATH="$cuda/bin/ptxas"
        export TRITON_PTXAS_BLACKWELL_PATH="$cuda/bin/ptxas"
    fi
}
