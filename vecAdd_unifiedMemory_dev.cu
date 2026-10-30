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

void initArray(float* A, int length) {
    std::srand(std::time({}));
    for(int i=0; i<length; i++) {
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

// TODO @bxyu-nvidia: Refactor the scalar and vector approximately equal functions to leverage this shared implementation
bool approximatelyEqual(const float* reference, const float* actual, const int length, const float rtol=1e-5f, const float atol=1e-6f) {
    for (int i=0; i < length; ++i) {
        if(fabs(actual[i] - reference[i]) >= atol + rtol * fabs(reference[i])) {
            printf("Mismatch at %d: actual %f != reference %f", i, actual[i], reference[i]);
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
    extern __shared__ float warpSums[];

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

    sum0 += sum1 + (sum2 + sum3);

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum0 += __shfl_down_sync(0xffffffffu, sum0, offset);
    }

    if (threadIdx.x % 32 == 0) {
        warpSums[threadIdx.x / 32] = sum0;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        *C = 0.0f;
        for (int i=0; i < blockDim.x / 32; ++i) {
            *C += warpSums[i];
        }
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
    const int sharedBytes = (threads / 32) * sizeof(float);
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


struct MatMulInput {
    // For matmuls below, we consider A[M, N] x B[N, K] = C[M, K]
    // Matrices are held in row-major format i.e. A[m, n] = m * N + n
    const float* A;
    const float* B;
    float* C;

    const int M;
    const int N;
    const int K;

    bool _ownsData;

    MatMulInput(const int M, const int N, const int K) : A(nullptr), B(nullptr), C(nullptr), M(M), N(N), K(K), _ownsData(true) {
        validate(M, N, K);

        float* A = nullptr;
        float* B = nullptr;

        cudaMallocManaged(&A, size_t(this->M) * this->N * sizeof(float));
        cudaMallocManaged(&B, size_t(this->N) * this->K * sizeof(float));
        cudaMallocManaged(&this->C, size_t(this->M) * this->K * sizeof(float));

        initArray(A, this->M * this->N);
        initArray(B, this->N * this->K);

        this->A = A;
        this->B = B;
    }

    MatMulInput(
        const float* A,
        const float* B,
        float* C,
        const int M,
        const int N,
        const int K
    ) : A(A), B(B), C(C), M(M), N(N), K(K), _ownsData(false)
    {
        validate(M, N, K);
    }

    void validate(const int M, const int N, const int K) {
        assert(
            M > 0 && N > 0 && K > 0
            && (M & (M - 1)) == 0 && (N & (N - 1)) == 0 && (K & (K - 1)) == 0
            && N % 256 == 0 && K % 256 == 0
            && "M, N, and K must be a power of two and N and K must be multiples of 256"
        );
    }

    ~MatMulInput() {
        if (!this->_ownsData) {
            return;
        }

        cudaFree(const_cast<float*>(this->A));
        cudaFree(const_cast<float*>(this->B));
        cudaFree(this->C);
    }

    MatMulInput(const MatMulInput&) = delete;
    MatMulInput& operator=(const MatMulInput&) = delete;
};


void serialMatMul(const MatMulInput& input) {
    for (int m=0; m < input.M; ++m) {
        for (int k=0; k < input.K; ++k) {
            input.C[m * input.K + k] = 0.0f;
            for (int n=0; n < input.N; ++n) {
                input.C[m * input.K + k] += input.A[m * input.N + n] * input.B[n * input.K + k];
            }
        }
    }
}


// Naive implementation: Within a single block, each thread calculates one output block in the C matrix
// For row-major layout, the memory looks like:
// A: a1,1 a1,2 ... a1,N a2,1 ... a2,N ... aM,N
// B: b1,1 b1,2 ... b1,K b2,1 ... b2,K ... bN,K
// C: c1,1 = a1,1 * b1,1 + a1,2 * b2,1 + ... + a1,N * bN,1
// For single warps of 32 threads:
// Access a1,1 and b1,1 ... b1,32 to write into c1,1 ... c1,32
__global__ void matMul(const MatMulInput& input) {
    float currSum;
    for (int m=0; m < input.M; ++m) {
        for (int k=threadIdx.x; k < input.K; k+=blockDim.x) {
            currSum = 0.0f;
            for (int n=0; n < input.N; ++n) {
                currSum += input.A[m * input.N + n] * input.B[n * input.K + k];
            }
            input.C[m * input.K + k] = currSum;
        }
    }
}

__global__ void matMulWithBlocks(const MatMulInput& input) {
    float currSum;
    for (int m=blockIdx.x; m < input.M; m+=gridDim.x) {
        for (int k=threadIdx.x; k < input.K; k+=blockDim.x) {
            currSum = 0.0f;
            for (int n=0; n < input.N; ++n) {
                currSum += input.A[m * input.N + n] * input.B[n * input.K + k];
            }
            input.C[m * input.K + k] = currSum;
        }
    }
}

__global__ void matMulWithIndepAccum4(const MatMulInput& input) {
    float sum0;
    float sum1;
    float sum2;
    float sum3;
    for (int m=blockIdx.x; m < input.M; m+=gridDim.x) {
        for (int k=threadIdx.x; k < input.K; k+=blockDim.x) {
            sum0 = 0.0f;
            sum1 = 0.0f;
            sum2 = 0.0f;
            sum3 = 0.0f;
            for (int n=0; n < input.N; n+=4) {
                sum0 += input.A[m * input.N + n] * input.B[n * input.K + k];
                sum1 += input.A[m * input.N + n + 1] * input.B[(n + 1) * input.K + k];
                sum2 += input.A[m * input.N + n + 2] * input.B[(n + 2) * input.K + k];
                sum3 += input.A[m * input.N + n + 3] * input.B[(n + 3) * input.K + k];
            }
            input.C[m * input.K + k] = (sum0 + sum1) + (sum2 + sum3);
        }
    }
}

__global__ void matMulWithFmaf(const MatMulInput& input) {
    float sum0;
    float sum1;
    float sum2;
    float sum3;
    for (int m=blockIdx.x; m < input.M; m+=gridDim.x) {
        for (int k=threadIdx.x; k < input.K; k+=blockDim.x) {
            sum0 = 0.0f;
            sum1 = 0.0f;
            sum2 = 0.0f;
            sum3 = 0.0f;
            for (int n=0; n < input.N; n+=4) {
                sum0 = fmaf(input.A[m * input.N + n], input.B[n * input.K + k], sum0);
                sum1 = fmaf(input.A[m * input.N + n + 1], input.B[(n + 1) * input.K + k], sum1);
                sum2 = fmaf(input.A[m * input.N + n + 2], input.B[(n + 2) * input.K + k], sum2);
                sum3 = fmaf(input.A[m * input.N + n + 3], input.B[(n + 3) * input.K + k], sum3);
            }
            input.C[m * input.K + k] = (sum0 + sum1) + (sum2 + sum3);
        }
    }
}

constexpr int TILE = 16;

__global__ void matMulTiledReference(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int m = blockIdx.y * TILE + ty;
    const int k = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    for (int base = 0; base < N; base += TILE) {
        // Each thread loads one element of each input tile.
        tileA[ty][tx] =
            (m < M && base + tx < N)
                ? A[size_t(m) * N + base + tx]
                : 0.0f;

        tileB[ty][tx] =
            (base + ty < N && k < K)
                ? B[size_t(base + ty) * K + k]
                : 0.0f;

        // All tile elements must be ready before using them.
        __syncthreads();

        #pragma unroll
        for (int n = 0; n < TILE; ++n) {
            sum = fmaf(tileA[ty][n], tileB[n][tx], sum);
        }

        // All threads must finish reading before tiles are overwritten.
        __syncthreads();
    }

    if (m < M && k < K) {
        C[size_t(m) * K + k] = sum;
    }
}

__global__ void matMulTiled(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int m = blockIdx.y * TILE + ty;
    const int k = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    for (int base = 0; base < N; base += TILE) {
        // Each thread loads one element of each input tile.
        tileA[ty][tx] = A[m * N + base + tx];
        tileB[ty][tx] = B[(base + ty) * K + k];

        // All tile elements must be ready before using them.
        __syncthreads();

        #pragma unroll
        for (int n = 0; n < TILE; ++n) {
            sum = fmaf(tileA[ty][n], tileB[n][tx], sum);
        }

        // All threads must finish reading before tiles are overwritten.
        __syncthreads();
    }

    if (m < M && k < K) {
        C[size_t(m) * K + k] = sum;
    }
}

void matMulBenchmark(const int M, const int N, const int K) {
    MatMulInput input = MatMulInput{M, N, K};

    float* comparisonResult = (float*)malloc(size_t(M) * K * sizeof(float));
    MatMulInput comparisonResultInput = MatMulInput{input.A, input.B, comparisonResult, M, N, K};

    cudaProfilerStart();

    for (int i=0; i < 1; ++i) {
        serialMatMul(comparisonResultInput);
    }
    nvtxRangePushA("serialMatMul");
    for (int i=0; i < 1<<1; ++i) {
        serialMatMul(comparisonResultInput);
    }
    nvtxRangePop();

    int blocks = 1;
    int threads = 256;
    for (int i=0; i < 1<<2; ++i) {
        matMul<<<blocks, threads>>>(input);
    }
    cudaDeviceSynchronize();
    nvtxRangePushA("matMul");
    for (int i=0; i < 1<<3; ++i) {
        matMul<<<blocks, threads>>>(input);
    }
    cudaDeviceSynchronize();
    nvtxRangePop();

    assert(approximatelyEqual(comparisonResult, input.C, M * K) && "matMul results must match serialMatMul results");

    blocks = 256;
    threads = 256;
    for (int i=0; i < 1<<2; ++i) {
        matMulWithBlocks<<<blocks, threads>>>(input);
    }
    cudaDeviceSynchronize();
    nvtxRangePushA("matMulWithBlocks");
    for (int i=0; i < 1<<3; ++i) {
        matMulWithBlocks<<<blocks, threads>>>(input);
    }
    cudaDeviceSynchronize();
    nvtxRangePop();

    assert(approximatelyEqual(comparisonResult, input.C, M * K) && "matMulWithBlocks results must match serialMatMul results");

    for (int i=0; i < 1<<2; ++i) {
        matMulWithIndepAccum4<<<blocks, threads>>>(input);
    }
    cudaDeviceSynchronize();
    nvtxRangePushA("matMulWithIndepAccum4");
    for (int i=0; i < 1<<3; ++i) {
        matMulWithIndepAccum4<<<blocks, threads>>>(input);
    }
    cudaDeviceSynchronize();
    nvtxRangePop();

    assert(approximatelyEqual(comparisonResult, input.C, M * K) && "matMulWithIndepAccum4 results must match serialMatMul results");

    for (int i=0; i < 1<<2; ++i) {
        matMulWithFmaf<<<blocks, threads>>>(input);
    }
    cudaDeviceSynchronize();
    nvtxRangePushA("matMulWithFmaf");
    for (int i=0; i < 1<<3; ++i) {
        matMulWithFmaf<<<blocks, threads>>>(input);
    }
    cudaDeviceSynchronize();
    nvtxRangePop();

    assert(approximatelyEqual(comparisonResult, input.C, M * K) && "matMulWithFmaf results must match serialMatMul results");

    dim3 threadsTiled(TILE, TILE);
    dim3 blocksTiled(input.K / TILE, input.M / TILE);
    for (int i=0; i < 1<<2; ++i) {
        matMulTiledReference<<<blocksTiled, threadsTiled>>>(
            input.A, input.B, input.C,
            input.M, input.N, input.K
        );
    }
    cudaDeviceSynchronize();
    nvtxRangePushA("matMulTiledReference");
    for (int i=0; i < 1<<3; ++i) {
        matMulTiledReference<<<blocksTiled, threadsTiled>>>(
            input.A, input.B, input.C,
            input.M, input.N, input.K
        );
    }
    cudaDeviceSynchronize();
    nvtxRangePop();

    assert(approximatelyEqual(comparisonResult, input.C, M * K) && "matMulTiledReference results must match serialMatMul results");

    for (int i=0; i < 1<<2; ++i) {
        matMulTiled<<<blocksTiled, threadsTiled>>>(
            input.A, input.B, input.C,
            input.M, input.N, input.K
        );
    }
    cudaDeviceSynchronize();
    nvtxRangePushA("matMulTiled");
    for (int i=0; i < 1<<3; ++i) {
        matMulTiled<<<blocksTiled, threadsTiled>>>(
            input.A, input.B, input.C,
            input.M, input.N, input.K
        );
    }
    cudaDeviceSynchronize();
    nvtxRangePop();

    assert(approximatelyEqual(comparisonResult, input.C, M * K) && "matMulTiled results must match serialMatMul results");

    cudaProfilerStop();

    // Clean Up
    free(comparisonResult);
}

int main(int argc, char** argv)
{
    // int vectorLength = 1024;
    // if(argc >=2)
    // {
    //     vectorLength = std::atoi(argv[1]);
    // }
    // vecAddBenchmark(vectorLength);
    // vecSumReduceBenchmark(vectorLength);

    // Input batch: A[T, D1] x [D1, D2]
    matMulBenchmark(16, 1024, 2048);
    // matMulBenchmark(16, 2048, 1024);

    return 0;
}
