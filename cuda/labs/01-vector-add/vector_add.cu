#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void checkCuda(cudaError_t status, const char* expression,
                      const char* file, int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s:%d: %s failed: %s\n",
                     file, line, expression, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(expr) checkCuda((expr), #expr, __FILE__, __LINE__)

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

static bool runCase(int n) {
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    std::vector<float> a(n), b(n), c(n);
    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i % 97) * 0.25f;
        b[i] = static_cast<float>(i % 31) * 0.5f;
    }

    float *da = nullptr, *db = nullptr, *dc = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&da), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&db), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dc), bytes));
    CUDA_CHECK(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));

    constexpr int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    vectorAdd<<<blocks, threads>>>(da, db, dc, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(c.data(), dc, bytes, cudaMemcpyDeviceToHost));

    bool passed = true;
    for (int i = 0; i < n; ++i) {
        const float expected = a[i] + b[i];
        if (!std::isfinite(c[i]) || std::fabs(c[i] - expected) > 1e-6f) {
            std::fprintf(stderr, "N=%d index=%d got=%g expected=%g\n",
                         n, i, c[i], expected);
            passed = false;
            break;
        }
    }

    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));
    std::printf("N=%d: %s\n", n, passed ? "PASS" : "FAIL");
    return passed;
}

int main() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count == 0) {
        std::fprintf(stderr, "No CUDA device found\n");
        return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s\nCompute capability: %d.%d\nVRAM: %.0f MiB\n",
                prop.name, prop.major, prop.minor,
                static_cast<double>(prop.totalGlobalMem) / (1024 * 1024));

    const int sizes[] = {1, 31, 32, 33, 255, 256, 257, 1000003};
    bool allPassed = true;
    for (int n : sizes) {
        if (!runCase(n)) {
            allPassed = false;
        }
    }
    std::puts(allPassed ? "ALL TESTS PASSED" : "TEST FAILED");
    return allPassed ? EXIT_SUCCESS : EXIT_FAILURE;
}
