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

const int TILE = 16;

__global__ void basicMatMul(const float *A, const float *B,
                            float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
            sum += A[row * N + k] * B[k * N + col];

        C[row * N + col] = sum;
    }
}

__global__ void tiledMatMul(const float *A, const float *B,
                            float *C, int N) {
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < N / TILE; t++) {
        tileA[threadIdx.y][threadIdx.x] =
            A[row * N + t * TILE + threadIdx.x];

        tileB[threadIdx.y][threadIdx.x] =
            B[(t * TILE + threadIdx.y) * N + col];

        __syncthreads();

        for (int k = 0; k < TILE; k++)
            sum += tileA[threadIdx.y][k] *
                   tileB[k][threadIdx.x];

        __syncthreads();
    }

    C[row * N + col] = sum;
}

void cpuMatMul(const vector<float>& A,
               const vector<float>& B,
               vector<float>& C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }
}

int main() {
    int sizes[] = {128, 256, 512};

    cout << "CUDA Matrix Multiplication Benchmark\n\n";
    cout << "N\tCPU(ms)\tBasic GPU(ms)\tTiled GPU(ms)"
         << "\tBasic Check\tTiled Check\n";

    for (int N : sizes) {
        size_t bytes = (size_t)N * N * sizeof(float);

        vector<float> A(N * N), B(N * N);
        vector<float> cpuC(N * N), basicC(N * N), tiledC(N * N);

        for (int i = 0; i < N * N; i++) {
            A[i] = float(i % 7) * 0.1f;
            B[i] = float(i % 5) * 0.2f;
        }

        auto cpuStart = high_resolution_clock::now();
        cpuMatMul(A, B, cpuC, N);
        auto cpuEnd = high_resolution_clock::now();

        double cpuMs =
            duration<double, milli>(cpuEnd - cpuStart).count();

        float *d_A, *d_B, *d_C;
        CHECK(cudaMalloc(&d_A, bytes));
        CHECK(cudaMalloc(&d_B, bytes));
        CHECK(cudaMalloc(&d_C, bytes));

        CHECK(cudaMemcpy(d_A, A.data(), bytes,
                         cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_B, B.data(), bytes,
                         cudaMemcpyHostToDevice));

        dim3 threads(TILE, TILE);
        dim3 blocks(N / TILE, N / TILE);

        // Warm-up both kernels
        basicMatMul<<<blocks, threads>>>(d_A, d_B, d_C, N);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        tiledMatMul<<<blocks, threads>>>(d_A, d_B, d_C, N);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));

        const int reps = 10;

        CHECK(cudaEventRecord(start));
        for (int r = 0; r < reps; r++)
            basicMatMul<<<blocks, threads>>>(d_A, d_B, d_C, N);
        CHECK(cudaGetLastError());
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));

        float basicMs;
        CHECK(cudaEventElapsedTime(&basicMs, start, stop));
        basicMs /= reps;

        CHECK(cudaMemcpy(basicC.data(), d_C, bytes,
                         cudaMemcpyDeviceToHost));

        CHECK(cudaEventRecord(start));
        for (int r = 0; r < reps; r++)
            tiledMatMul<<<blocks, threads>>>(d_A, d_B, d_C, N);
        CHECK(cudaGetLastError());
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));

        float tiledMs;
        CHECK(cudaEventElapsedTime(&tiledMs, start, stop));
        tiledMs /= reps;

        CHECK(cudaMemcpy(tiledC.data(), d_C, bytes,
                         cudaMemcpyDeviceToHost));

        bool basicOK = true, tiledOK = true;

        for (int i = 0; i < N * N; i++) {
            if (fabs(cpuC[i] - basicC[i]) > 1e-2f)
                basicOK = false;
            if (fabs(cpuC[i] - tiledC[i]) > 1e-2f)
                tiledOK = false;
        }

        cout << N << "\t" << cpuMs << "\t"
             << basicMs << "\t" << tiledMs << "\t"
             << (basicOK ? "PASS" : "FAIL") << "\t\t"
             << (tiledOK ? "PASS" : "FAIL") << endl;

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
        CHECK(cudaFree(d_A));
        CHECK(cudaFree(d_B));
        CHECK(cudaFree(d_C));
    }

    return 0;
}
