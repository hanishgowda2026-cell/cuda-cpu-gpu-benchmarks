#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;
using namespace chrono;

#define CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        cerr << "CUDA error: " << cudaGetErrorString(err) \
             << " at line " << __LINE__ << endl; \
        return 1; \
    } \
} while (0)

const int THREADS = 256;
const int ITEMS_PER_BLOCK = THREADS * 2;

__global__ void reduceSum(const float *input, float *output, int n) {
    extern __shared__ float shared[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + tid;

    float value = 0.0f;

    if (i < n)
        value += input[i];

    if (i + blockDim.x < n)
        value += input[i + blockDim.x];

    shared[tid] = value;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2;
         stride > 0; stride >>= 1) {
        if (tid < stride)
            shared[tid] += shared[tid + stride];

        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = shared[0];
}

float cpuSum(const vector<float>& input) {
    float sum = 0.0f;
    for (float value : input)
        sum += value;
    return sum;
}

// Reduce all the way to one value on the GPU.
void launchFullReduction(const float *input,
                         float *temp1,
                         float *temp2,
                         int n) {
    int currentN = n;
    const float *currentInput = input;
    float *currentOutput = temp1;

    while (currentN > 1) {
        int blocks = (currentN + ITEMS_PER_BLOCK - 1)
                     / ITEMS_PER_BLOCK;

        reduceSum<<<blocks, THREADS, THREADS * sizeof(float)>>>(
            currentInput, currentOutput, currentN);

        currentInput = currentOutput;
        currentOutput = (currentOutput == temp1) ? temp2 : temp1;
        currentN = blocks;
    }
}

int main() {
    int sizes[] = {1 << 20, 1 << 22, 1 << 24};
    const int repetitions = 20;

    cout << "Multi-Stage CUDA Parallel Reduction\n\n";
    cout << "Elements\tCPU(ms)\tGPU Kernels(ms)"
         << "\tGPU Total(ms)\tKernel Speedup\tCheck\n";

    for (int n : sizes) {
        vector<float> input(n, 1.0f);

        auto cpuStart = high_resolution_clock::now();
        float cpuResult = cpuSum(input);
        auto cpuEnd = high_resolution_clock::now();

        double cpuMs =
            duration<double, milli>(cpuEnd - cpuStart).count();

        float *d_input, *d_temp1, *d_temp2;
        int capacity = (n + ITEMS_PER_BLOCK - 1)
                       / ITEMS_PER_BLOCK;

        CHECK(cudaMalloc(&d_input, (size_t)n * sizeof(float)));
        CHECK(cudaMalloc(&d_temp1, (size_t)capacity * sizeof(float)));
        CHECK(cudaMalloc(&d_temp2, (size_t)capacity * sizeof(float)));

        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));

        auto totalStart = high_resolution_clock::now();

        CHECK(cudaMemcpy(d_input, input.data(),
                         (size_t)n * sizeof(float),
                         cudaMemcpyHostToDevice));

        // Warm-up
        launchFullReduction(d_input, d_temp1, d_temp2, n);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        // Measure the full sequence of GPU reduction kernels.
        CHECK(cudaEventRecord(start));

        for (int r = 0; r < repetitions; r++)
            launchFullReduction(d_input, d_temp1, d_temp2, n);

        CHECK(cudaGetLastError());
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));

        float kernelMs;
        CHECK(cudaEventElapsedTime(&kernelMs, start, stop));
        kernelMs /= repetitions;

        // Obtain the final result from the GPU.
        float gpuResult = 0.0f;
        CHECK(cudaMemcpy(&gpuResult, d_temp1, sizeof(float),
                         cudaMemcpyDeviceToHost));

        // Correctly identify which temporary buffer holds the result.
        int remaining = n;
        int stages = 0;
        while (remaining > 1) {
            remaining = (remaining + ITEMS_PER_BLOCK - 1)
                        / ITEMS_PER_BLOCK;
            stages++;
        }

        float *resultBuffer = (stages % 2 == 1) ? d_temp1 : d_input;

        // Re-run once so the result buffer is known and current.
        launchFullReduction(d_input, d_temp1, d_temp2, n);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        // Locate the final output based on the actual ping-pong stages.
        resultBuffer = (stages % 2 == 1) ? d_temp1 : d_temp2;

        CHECK(cudaMemcpy(&gpuResult, resultBuffer, sizeof(float),
                         cudaMemcpyDeviceToHost));

        auto totalEnd = high_resolution_clock::now();

        double totalMs =
            duration<double, milli>(totalEnd - totalStart).count();

        bool correct = fabs(cpuResult - gpuResult) < 1.0f;

        cout << n << "\t" << cpuMs << "\t"
             << kernelMs << "\t" << totalMs << "\t"
             << cpuMs / kernelMs << "x\t\t"
             << (correct ? "PASS" : "FAIL") << endl;

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
        CHECK(cudaFree(d_input));
        CHECK(cudaFree(d_temp1));
        CHECK(cudaFree(d_temp2));
    }

    return 0;
}
