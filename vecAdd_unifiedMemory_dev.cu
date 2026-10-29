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

bool scalarApproximatelyEqual(const float* reference, const float* actual, const float rtol=1e-5f, const float atol=1e-6f) {
    if(fabs(*actual - *reference) >= atol + rtol * fabs(*reference))
    {
        printf("Mismatch: actual %f != reference %f", *actual, *reference);
        return false;
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

void serialVecSum(const float* A, float* C, const int vectorLength) {
    *C = 0.0f;
    for (int i=0; i < vectorLength; ++i) {
        *C += A[i];
    }
}

void validateVecSum(const int vectorLength) {
    assert(
        vectorLength > 0 &&
        (vectorLength & (vectorLength - 1)) == 0 &&
        vectorLength >= 4096 && 
        "vectorLength must be a power of two and >= 4096"
    );
}

// Within a single block
__global__ void vecSum(const float* A, float* C, const int vectorLength) {
    extern __shared__ float partial[];

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    int i;
    for (i=threadIdx.x; i < vectorLength; i += 16 * blockDim.x) {
        sum0 += A[i] + A[i + blockDim.x] + A[i + 2 * blockDim.x] + A[i + 3 * blockDim.x];
        sum1 += A[i + 4 * blockDim.x] + A[i + 5 * blockDim.x] + A[i + 6 * blockDim.x] + A[i + 7 * blockDim.x];
        sum2 += A[i + 8 * blockDim.x] + A[i + 9 * blockDim.x] + A[i + 10 * blockDim.x] + A[i + 11 * blockDim.x];
        sum3 += A[i + 12 * blockDim.x] + A[i + 13 * blockDim.x] + A[i + 14 * blockDim.x] + A[i + 15 * blockDim.x];
    }

    partial[threadIdx.x] = (sum0 + sum1) + (sum2 + sum3);
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *C = partial[0];
    }
}

void vecSumReduceBenchmark(int vectorLength) {
    float* A = nullptr;
    float* C;
    float* comparisonResult = (float*)malloc(sizeof(float));

    cudaMallocManaged(&A, vectorLength*sizeof(float));
    cudaMallocManaged(&C, sizeof(float));

    initArray(A, vectorLength);

    cudaProfilerStart();

    for (int i=0; i < 1<<7; ++i) {
        serialVecSum(A, comparisonResult, vectorLength);
    }
    nvtxRangePushA("serialVecSum");
    for (int i=0; i < 1<<10; ++i) {
        serialVecSum(A, comparisonResult, vectorLength);
    }
    nvtxRangePop();

    const int blocks = 1;  // vecSum is within block right now
    const int threads = 256;
    const int sharedBytes = threads * sizeof(float);
    validateVecSum(vectorLength);
    for (int i=0; i < 1<<7; ++i) {
        vecSum<<<blocks, threads, sharedBytes>>>(A, C, vectorLength);
    }
    cudaDeviceSynchronize();
    nvtxRangePushA("vecSum");
    for (int i=0; i < 1<<10; ++i) {
        vecSum<<<blocks, threads, sharedBytes>>>(A, C, vectorLength);
    }
    cudaDeviceSynchronize();
    nvtxRangePop();

    cudaProfilerStop();

    assert(scalarApproximatelyEqual(comparisonResult, C) && "GPU results must match CPU results");

    // Clean Up
    cudaFree(A);
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
    // vecAddBenchmark(vectorLength);
    vecSumReduceBenchmark(vectorLength);
    return 0;
}
