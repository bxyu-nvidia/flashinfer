nvcc -lineinfo --cudart=shared vecAdd_unifiedMemory_dev.cu -o vecAdd_unifiedMemory_dev.o \
&& (CUDA_LOG_FILE=stderr \
nsys profile \
    --force-overwrite=true \
    --trace=cuda,nvtx \
    --sample=none \
    --cpuctxsw=none \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --cuda-graph-trace=node \
    -o $(pwd)/vecAdd_unifiedMemory_dev_profile \
./vecAdd_unifiedMemory_dev.o 4096 \
; nsys stats \
    --force-export=true \
    --timeunit usec \
    --report cuda_gpu_kern_sum,nvtx_sum \
    vecAdd_unifiedMemory_dev_profile.nsys-rep \
)
