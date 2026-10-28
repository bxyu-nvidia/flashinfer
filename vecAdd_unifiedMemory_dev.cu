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


__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
    int workIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if(workIndex < vectorLength)
    {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}

void initArray(float* A, int length)
{
     std::srand(std::time({}));
    for(int i=0; i<length; i++)
    {
        A[i] = rand() / (float)RAND_MAX;
    }
}

void serialVecAdd(float* A, float* B, float* C,  int length)
{
    for(int i=0; i<length; i++)
    {
        C[i] = A[i] + B[i];
    }
}

bool vectorApproximatelyEqual(float* A, float* B, int length, float epsilon=0.00001)
{
    for(int i=0; i<length; i++)
    {
        if(fabs(A[i] -B[i]) > epsilon)
        {
            printf("Index %d mismatch: %f != %f", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}


void vecAddBenchmark(int vectorLength)
{
    // ----------------------------------------
    // Setup
    // ----------------------------------------
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));

    cudaMallocManaged(&A, vectorLength*sizeof(float));
    cudaMallocManaged(&B, vectorLength*sizeof(float));
    cudaMallocManaged(&C, vectorLength*sizeof(float));

    initArray(A, vectorLength);
    initArray(B, vectorLength);

    cudaProfilerStart();

    // ----------------------------------------
    // vecAdd
    // ----------------------------------------

    // Launch the kernel. Unified memory will make sure A, B, and C are
    // accessible to the GPU
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    // Warmup
    for (int i = 0; i < 1<<7 ; ++i) {
        vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
    }
    cudaDeviceSynchronize();
    nvtxRangePushA("vecAdd");
    for (int i = 0; i < 1<<10 ; ++i) {
        vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
    }
    nvtxRangePop();

    // ----------------------------------------
    // vecAdd with Cuda graphs
    // ----------------------------------------

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    // Warmup
    for (int i = 0; i < 1<<7 ; ++i) {
        vecAdd<<<blocks, threads, 0, stream>>>(A, B, C, vectorLength);
    }
    cudaStreamSynchronize(stream);

    // Capture
    cudaGraph_t graph;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    for (int i = 0; i < 1024; ++i) {
        vecAdd<<<blocks, threads, 0, stream>>>(A, B, C, vectorLength);
    }
    cudaStreamEndCapture(stream, &graph);

    cudaGraphExec_t executable;
    cudaGraphInstantiate(&executable, graph, 0);

    nvtxRangePushA("vecAdd with Cuda graphs");
    cudaGraphLaunch(executable, stream);
    cudaStreamSynchronize(stream);
    nvtxRangePop();

    cudaGraphExecDestroy(executable);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);

    // ----------------------------------------
    // serialVecAdd (CPU)
    // ----------------------------------------
    // Perform computation serially on CPU for comparison
    // Warmup
    for (int i = 0; i < 1<<7 ; ++i) {
        serialVecAdd(A, B, comparisonResult, vectorLength);
    }
    nvtxRangePushA("serialVecAdd");
    for (int i = 0; i < 1<<10 ; ++i) {
        serialVecAdd(A, B, comparisonResult, vectorLength);
    }
    nvtxRangePop();

    cudaProfilerStop();

    assert(vectorApproximatelyEqual(C, comparisonResult, vectorLength) && "GPU results must match CPU results");

    // Clean Up
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(comparisonResult);
}

void vecSumBenchmark(int vectorLength) {
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));

    cudaMallocManaged(&A, vectorLength*sizeof(float));
    cudaMallocManaged(&B, vectorLength*sizeof(float));
    cudaMallocManaged(&C, vectorLength*sizeof(float));

    initArray(A, vectorLength);
    initArray(B, vectorLength);

    cudaProfilerStart();

    cudaProfilerStop();

    assert(vectorApproximatelyEqual(C, comparisonResult, vectorLength) && "GPU results must match CPU results");

    // Clean Up
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(comparisonResult);
}


int main(int argc, char** argv)
{
    int vectorLength = 1024;
    if(argc >=2)
    {
        vectorLength = std::atoi(argv[1]);
    }
    vecAddBenchmark(vectorLength);
    return 0;
}
