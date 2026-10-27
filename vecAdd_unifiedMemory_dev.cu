#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <cuda/cmath>
#include <cstdio>


struct Timer {
    cudaEvent_t _start;
    cudaEvent_t _stop;

    std::string _label;
    float _milliseconds;

    Timer() {
        cudaEventCreate(&this->_start);
        cudaEventCreate(&this->_stop);

        this->_milliseconds = 0.0;
    }

    ~Timer() {
        cudaEventDestroy(this->_start);
        cudaEventDestroy(this->_stop);
    }
    
    // @bxyu-nvidia: Copy and copy assignment guards. I don't fully appreciate these yet.
    Timer(const Timer&) = delete;
    Timer& operator=(const Timer&) = delete;

    void start(const std::string& label) {
        this->_label = label;
        cudaEventRecord(this->_start);
    }

    void stop()
    {
        cudaEventRecord(this->_stop);
        cudaEventSynchronize(this->_stop);

        cudaEventElapsedTime(&this->_milliseconds, this->_start, this->_stop);
        printf("%s time: %.3f us\n", this->_label.c_str(), 1000 * this->_milliseconds);
    }
};


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


//unified-memory-begin
void unifiedMemExample(int vectorLength)
{
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));

    Timer timer;

    timer.start("Unified memory buffer malloc");
    cudaMallocManaged(&A, vectorLength*sizeof(float));
    cudaMallocManaged(&B, vectorLength*sizeof(float));
    cudaMallocManaged(&C, vectorLength*sizeof(float));
    timer.stop();

    timer.start("Initialize vectors on the host");
    initArray(A, vectorLength);
    initArray(B, vectorLength);
    timer.stop();

    // Launch the kernel. Unified memory will make sure A, B, and C are
    // accessible to the GPU
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    timer.start("vecAdd");
    for (int counter = 0; counter < 32; counter++) {
        vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
        // Wait for the kernel to complete execution
        cudaDeviceSynchronize();
    }
    timer.stop();

    // Perform computation serially on CPU for comparison
    timer.start("serialVecAdd");
    serialVecAdd(A, B, comparisonResult, vectorLength);
    timer.stop();

    // Confirm that CPU and GPU got the same answer
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Unified Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Unified Memory: Error - CPU and GPU answers do not match\n");
    }

    // Clean Up
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(comparisonResult);

}
//unified-memory-end


int main(int argc, char** argv)
{
    int vectorLength = 1024;
    if(argc >=2)
    {
        vectorLength = std::atoi(argv[1]);
    }
    unifiedMemExample(vectorLength);		
    return 0;
}
