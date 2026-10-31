#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s at %s:%d\n", \
                     cudaGetErrorString(e), __FILE__, __LINE__); \
        return 2; \
    } \
} while (0)

int main() {
    int dev = 0, managed = 0, concurrent = 0;
    CHECK(cudaGetDevice(&dev));
    CHECK(cudaDeviceGetAttribute(&managed, cudaDevAttrManagedMemory, dev));
    CHECK(cudaDeviceGetAttribute(&concurrent, cudaDevAttrConcurrentManagedAccess, dev));
    std::printf("managedMemory=%d concurrentManagedAccess=%d\n", managed, concurrent);
    // Negative test: force failure to verify that env_check.sh stops and fails loudly
    return (managed && concurrent) ? 0 : 1;
}
