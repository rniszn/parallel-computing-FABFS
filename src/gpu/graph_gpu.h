#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include "graph/types.h"

struct GpuGraph {
  vid_t n;
  eid_t m;
  eid_t* row;  // cudaMallocManaged, size n+1
  vid_t* col;  // cudaMallocManaged, size m
};

// Allocates managed memory for row and col, copies data from g,
// and sets read-mostly memory advice.
GpuGraph upload_graph(const Graph& g);

// Frees managed memory in gg and resets fields.
void free_graph(GpuGraph& gg);
