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
    // On WSL2/WDDM on laptop RTX 4050, concurrentManagedAccess is 0.
    // Data-plane adapts using pinned zero-copy host memory (cudaHostAllocMapped).
    // Device must support managed memory.
    return (managed) ? 0 : 1;
}
