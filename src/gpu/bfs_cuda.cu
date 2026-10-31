#include "bfs_cuda.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#define CHECK_CUDA(call)                                                         \
  do {                                                                           \
    cudaError_t err = (call);                                                    \
    if (err != cudaSuccess) {                                                    \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(err));                                     \
      throw std::runtime_error(cudaGetErrorString(err));                         \
    }                                                                            \
  } while (0)

__global__ void expand_kernel(GpuGraph gg, int32_t* dist, const vid_t* frontier,
                              size_t fbegin, size_t count, vid_t* next,
                              unsigned long long* counts, int32_t new_level) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= count) return;
  vid_t u = frontier[fbegin + i];
  eid_t begin = gg.row[u], end = gg.row[u + 1];
  atomicAdd(&counts[1], (unsigned long long)(end - begin));
  for (eid_t e = begin; e < end; ++e) {
    vid_t v = gg.col[e];
    if (atomicCAS(&dist[v], -1, new_level) == -1) {
      unsigned long long idx = atomicAdd(&counts[0], 1ULL);
      next[idx] = v;
    }
  }
}

uint64_t expand_level_gpu(const GpuGraph& gg, BfsState& st, size_t fbegin, size_t fend) {
  if (fbegin >= fend) {
    st.next_size = static_cast<size_t>(st.counts[0]);
    return 0;
  }

  size_t count = fend - fbegin;

  // Prefetch if supported by platform/driver (requires concurrentManagedAccess == 1)
  int dev = 0;
  CHECK_CUDA(cudaGetDevice(&dev));
  int concurrent = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&concurrent, cudaDevAttrConcurrentManagedAccess, dev));
  if (concurrent == 1) {
    cudaMemPrefetchAsync(st.frontier + fbegin, count * sizeof(vid_t), dev);
    cudaMemPrefetchAsync(st.dist, gg.n * sizeof(int32_t), dev);
  }

  constexpr int BLOCK_SIZE = 256;
  size_t blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;

  expand_kernel<<<blocks, BLOCK_SIZE>>>(
      gg, st.dist, st.frontier, fbegin, count, st.next, st.counts, st.level + 1);

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  st.next_size = static_cast<size_t>(st.counts[0]);
  return static_cast<uint64_t>(st.counts[1]);
}

std::vector<int32_t> bfs_gpu(const Graph& g, vid_t src) {
  if (src >= g.n) {
    throw std::out_of_range("Source vertex out of range");
  }

  GpuGraph gg = upload_graph(g);
  BfsState st = make_state(g, src);

  while (st.frontier_size > 0) {
    reset_level(st);
    expand_level_gpu(gg, st, 0, st.frontier_size);
    std::swap(st.frontier, st.next);
    st.frontier_size = st.next_size;
    st.next_size = 0;
    st.level++;
  }

  std::vector<int32_t> result(g.n);
  std::memcpy(result.data(), st.dist, g.n * sizeof(int32_t));

  free_state(st);
  free_graph(gg);

  return result;
}
