#include "bfs_state.h"
#include <algorithm>
#include <cstdio>
#include <cuda_runtime.h>
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

BfsState make_state(const Graph& g, vid_t src) {
  if (src >= g.n) {
    throw std::out_of_range("Source vertex out of range");
  }

  BfsState st;
  size_t n_alloc = std::max<size_t>(1, g.n);

  CHECK_CUDA(cudaMallocManaged(&st.dist, sizeof(int32_t) * n_alloc));
  CHECK_CUDA(cudaMallocManaged(&st.frontier, sizeof(vid_t) * n_alloc));
  CHECK_CUDA(cudaMallocManaged(&st.next, sizeof(vid_t) * n_alloc));
  CHECK_CUDA(cudaMallocManaged(&st.counts, sizeof(unsigned long long) * 2));

  for (size_t i = 0; i < g.n; ++i) {
    st.dist[i] = -1;
  }
  st.dist[src] = 0;

  st.frontier[0] = src;
  st.frontier_size = 1;
  st.next_size = 0;
  st.counts[0] = 0;
  st.counts[1] = 0;
  st.level = 0;

  return st;
}

void reset_level(BfsState& st) {
  if (st.counts) {
    st.counts[0] = 0;
    st.counts[1] = 0;
  }
  st.next_size = 0;
}

void free_state(BfsState& st) {
  if (st.dist) {
    CHECK_CUDA(cudaFree(st.dist));
    st.dist = nullptr;
  }
  if (st.frontier) {
    CHECK_CUDA(cudaFree(st.frontier));
    st.frontier = nullptr;
  }
  if (st.next) {
    CHECK_CUDA(cudaFree(st.next));
    st.next = nullptr;
  }
  if (st.counts) {
    CHECK_CUDA(cudaFree(st.counts));
    st.counts = nullptr;
  }
  st.frontier_size = 0;
  st.next_size = 0;
  st.level = 0;
}
