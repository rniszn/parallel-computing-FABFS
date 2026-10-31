#pragma once

#include <cstdint>
#include <vector>
#include "common/bfs_state.h"
#include "gpu/graph_gpu.h"

// Expands frontier[fbegin..fend) at st.level. Sets st.next_size. Returns edges scanned.
// Resets counts[0] (next counter) and counts[1] (edge counter) at the start.
// Does NOT swap frontier and next.
uint64_t expand_level_gpu(const GpuGraph& gg, BfsState& st, size_t fbegin, size_t fend);

// Full BFS. Uses make_state and free_state from the shared state.
std::vector<int32_t> bfs_gpu(const Graph& g, vid_t src);
