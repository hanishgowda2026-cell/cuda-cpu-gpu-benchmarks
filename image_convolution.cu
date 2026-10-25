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
const int SHARED = TILE + 2;

__global__ void basicBlur(const float *in, float *out,
                          int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float sum = 0.0f;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && nx < width &&
                ny >= 0 && ny < height)
                sum += in[ny * width + nx];
        }
    }

    out[y * width + x] = sum / 9.0f;
}

__global__ void tiledBlur(const float *in, float *out,
                           int width, int height) {
    __shared__ float tile[SHARED][SHARED];

    int tid = threadIdx.y * TILE + threadIdx.x;
    int threads = TILE * TILE;

    int baseX = blockIdx.x * TILE;
    int baseY = blockIdx.y * TILE;

    // Cooperatively load the tile and its one-pixel halo.
    for (int i = tid; i < SHARED * SHARED; i += threads) {
        int sx = i % SHARED;
        int sy = i / SHARED;

        int gx = baseX + sx - 1;
        int gy = baseY + sy - 1;

        if (gx >= 0 && gx < width &&
            gy >= 0 && gy < height)
            tile[sy][sx] = in[gy * width + gx];
        else
            tile[sy][sx] = 0.0f;
    }

    __syncthreads();

    int x = baseX + threadIdx.x;
    int y = baseY + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float sum = 0.0f;

    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++)
            sum += tile[threadIdx.y + 1 + dy]
                       [threadIdx.x + 1 + dx];

    out[y * width + x] = sum / 9.0f;
}

void cpuBlur(const vector<float>& in, vector<float>& out,
             int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;

            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int nx = x + dx;
                    int ny = y + dy;

                    if (nx >= 0 && nx < width &&
                        ny >= 0 && ny < height)
                        sum += in[ny * width + nx];
                }
            }

            out[y * width + x] = sum / 9.0f;
        }
    }
}

int main() {
    int sizes[] = {512, 1024, 2048};
    const int repetitions = 10;

    cout << "CUDA Image Convolution Benchmark\n\n";
    cout << "Size\tCPU(ms)\tBasic GPU(ms)"
         << "\tTiled GPU(ms)\tBasic Check\tTiled Check\n";

    for (int N : sizes) {
        size_t count = (size_t)N * N;
        size_t bytes = count * sizeof(float);

        vector<float> input(count), cpu(count);
        vector<float> basic(count), tiled(count);

        for (size_t i = 0; i < count; i++)
            input[i] = float(i % 256) / 255.0f;

        auto cpuStart = high_resolution_clock::now();
        cpuBlur(input, cpu, N, N);
        auto cpuEnd = high_resolution_clock::now();

        double cpuMs =
            duration<double, milli>(cpuEnd - cpuStart).count();

        float *d_in, *d_basic, *d_tiled;
        CHECK(cudaMalloc(&d_in, bytes));
        CHECK(cudaMalloc(&d_basic, bytes));
        CHECK(cudaMalloc(&d_tiled, bytes));

        CHECK(cudaMemcpy(d_in, input.data(), bytes,
                         cudaMemcpyHostToDevice));

        dim3 threads(TILE, TILE);
        dim3 blocks((N + TILE - 1) / TILE,
                    (N + TILE - 1) / TILE);

        // Warm-up
        basicBlur<<<blocks, threads>>>(d_in, d_basic, N, N);
        CHECK(cudaGetLastError());

        tiledBlur<<<blocks, threads>>>(d_in, d_tiled, N, N);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));

        CHECK(cudaEventRecord(start));
        for (int r = 0; r < repetitions; r++)
            basicBlur<<<blocks, threads>>>(d_in, d_basic, N, N);
        CHECK(cudaGetLastError());
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));

        float basicMs;
        CHECK(cudaEventElapsedTime(&basicMs, start, stop));
        basicMs /= repetitions;

        CHECK(cudaEventRecord(start));
        for (int r = 0; r < repetitions; r++)
            tiledBlur<<<blocks, threads>>>(d_in, d_tiled, N, N);
        CHECK(cudaGetLastError());
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));

        float tiledMs;
        CHECK(cudaEventElapsedTime(&tiledMs, start, stop));
        tiledMs /= repetitions;

        CHECK(cudaMemcpy(basic.data(), d_basic, bytes,
                         cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(tiled.data(), d_tiled, bytes,
                         cudaMemcpyDeviceToHost));

        bool basicOK = true, tiledOK = true;

        for (size_t i = 0; i < count; i++) {
            if (fabs(cpu[i] - basic[i]) > 1e-5f)
                basicOK = false;
            if (fabs(cpu[i] - tiled[i]) > 1e-5f)
                tiledOK = false;
        }

        cout << N << "x" << N << "\t"
             << cpuMs << "\t" << basicMs << "\t"
             << tiledMs << "\t"
             << (basicOK ? "PASS" : "FAIL") << "\t\t"
             << (tiledOK ? "PASS" : "FAIL") << endl;

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
        CHECK(cudaFree(d_in));
        CHECK(cudaFree(d_basic));
        CHECK(cudaFree(d_tiled));
    }

    return 0;
}
