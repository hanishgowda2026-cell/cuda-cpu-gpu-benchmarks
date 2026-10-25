#include <iostream>
#include <vector>
#include <chrono>
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
const int ITEMS = THREADS * 2;

__global__ void blockScan(const int *input, int *output,
                          int *blockSums, int n) {
    __shared__ int s[ITEMS];

    int tid = threadIdx.x;
    int base = blockIdx.x * ITEMS;

    int i0 = base + tid;
    int i1 = base + tid + THREADS;

    s[tid] = (i0 < n) ? input[i0] : 0;
    s[tid + THREADS] = (i1 < n) ? input[i1] : 0;
    __syncthreads();

    // Inclusive scan. Read from the previous iteration's snapshot.
    for (int offset = 1; offset < ITEMS; offset *= 2) {
        int add0 = (tid >= offset) ? s[tid - offset] : 0;
        int add1 = (tid + THREADS >= offset)
            ? s[tid + THREADS - offset] : 0;

        __syncthreads();

        if (tid >= offset)
            s[tid] += add0;

        if (tid + THREADS >= offset)
            s[tid + THREADS] += add1;

        __syncthreads();
    }

    if (i0 < n)
        output[i0] = s[tid];

    if (i1 < n)
        output[i1] = s[tid + THREADS];

    if (tid == THREADS - 1)
        blockSums[blockIdx.x] = s[ITEMS - 1];
}

__global__ void addBlockOffsets(int *output, const int *offsets,
                                int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int block = i / ITEMS;

    if (i < n && block > 0)
        output[i] += offsets[block];
}

void cpuScan(const vector<int>& input, vector<int>& output) {
    int sum = 0;
    for (size_t i = 0; i < input.size(); i++) {
        sum += input[i];
        output[i] = sum;
    }
}

int main() {
    int sizes[] = {1 << 20, 1 << 22, 1 << 24};

    cout << "Corrected CUDA Prefix Sum Benchmark\n\n";
    cout << "Elements\tCPU(ms)\tGPU(ms)\tCheck\n";

    for (int n : sizes) {
        vector<int> input(n, 1), cpu(n), gpu(n);

        auto cpuStart = high_resolution_clock::now();
        cpuScan(input, cpu);
        auto cpuEnd = high_resolution_clock::now();

        double cpuMs =
            duration<double, milli>(cpuEnd - cpuStart).count();

        int blocks = (n + ITEMS - 1) / ITEMS;

        vector<int> blockSums(blocks), offsets(blocks);
        int *d_input, *d_output, *d_sums, *d_offsets;

        CHECK(cudaMalloc(&d_input, n * sizeof(int)));
        CHECK(cudaMalloc(&d_output, n * sizeof(int)));
        CHECK(cudaMalloc(&d_sums, blocks * sizeof(int)));
        CHECK(cudaMalloc(&d_offsets, blocks * sizeof(int)));

        CHECK(cudaMemcpy(d_input, input.data(), n * sizeof(int),
                         cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));

        CHECK(cudaEventRecord(start));

        blockScan<<<blocks, THREADS>>>(d_input, d_output, d_sums, n);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemcpy(blockSums.data(), d_sums,
                         blocks * sizeof(int), cudaMemcpyDeviceToHost));

        int sum = 0;
        for (int i = 0; i < blocks; i++) {
            offsets[i] = sum;
            sum += blockSums[i];
        }

        CHECK(cudaMemcpy(d_offsets, offsets.data(),
                         blocks * sizeof(int), cudaMemcpyHostToDevice));

        addBlockOffsets<<<(n + 255) / 256, 256>>>(
            d_output, d_offsets, n);
        CHECK(cudaGetLastError());

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));

        float gpuMs;
        CHECK(cudaEventElapsedTime(&gpuMs, start, stop));

        CHECK(cudaMemcpy(gpu.data(), d_output, n * sizeof(int),
                         cudaMemcpyDeviceToHost));

        bool correct = (cpu == gpu);

        cout << n << "\t" << cpuMs << "\t" << gpuMs << "\t"
             << (correct ? "PASS" : "FAIL") << endl;

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
        CHECK(cudaFree(d_input));
        CHECK(cudaFree(d_output));
        CHECK(cudaFree(d_sums));
        CHECK(cudaFree(d_offsets));
    }

    return 0;
}
