#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <cuda/cmath>
#include <cstdio>
#include <cassert>
#include <cuda_profiler_api.h>


// C[M,N] = A[M,N] + B[M,N]
__global__ void add(const float* A, const float* B, float* C, const int M, const int N)
{
    int workIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if(workIndex < vectorLength)
    {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}
