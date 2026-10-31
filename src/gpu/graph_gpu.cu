#include "graph_gpu.h"
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

GpuGraph upload_graph(const Graph& g) {
  GpuGraph gg;
  gg.n = g.n;
  gg.m = g.m;

  size_t row_bytes = (g.n + 1) * sizeof(eid_t);
  size_t col_bytes = std::max<size_t>(1, g.m) * sizeof(vid_t);

  CHECK_CUDA(cudaMallocManaged(&gg.row, row_bytes));
  CHECK_CUDA(cudaMallocManaged(&gg.col, col_bytes));

  std::memcpy(gg.row, g.row_offsets.data(), row_bytes);
  if (g.m > 0) {
    std::memcpy(gg.col, g.col_indices.data(), g.m * sizeof(vid_t));
  }

  int dev = 0;
  CHECK_CUDA(cudaGetDevice(&dev));

  CHECK_CUDA(cudaMemAdvise(gg.row, row_bytes, cudaMemAdviseSetReadMostly, dev));
  if (g.m > 0) {
    CHECK_CUDA(cudaMemAdvise(gg.col, g.m * sizeof(vid_t), cudaMemAdviseSetReadMostly, dev));
  }

  return gg;
}

void free_graph(GpuGraph& gg) {
  if (gg.row) {
    CHECK_CUDA(cudaFree(gg.row));
    gg.row = nullptr;
  }
  if (gg.col) {
    CHECK_CUDA(cudaFree(gg.col));
    gg.col = nullptr;
  }
  gg.n = 0;
  gg.m = 0;
}
