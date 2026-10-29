# Search for most recent version, only once on init
# apt-cache search '^nsight-compute-'
# apt-cache search '^nsight-systems-'

apt-get update && apt-get install -y --no-install-recommends \
    nsight-compute-2026.3.1 \
    nsight-systems-2026.3.2 \
    cuda-profiler-api-13-0 \
    cuda-nvtx-13-0
PATH="/opt/nvidia/nsight-compute/2026.3.1:$PATH" \
ncu --version
PATH="/opt/nvidia/nsight-systems/2026.3.2:$PATH" \
nsys --version

cd $1
python -m pip install --no-build-isolation -e . -v
