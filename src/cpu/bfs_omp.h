#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "common/bfs_state.h"
#include "graph/types.h"

// Expands frontier[fbegin..fend) at the current level. Appends each newly
// claimed vertex to st.next. Returns the number of edges scanned.
// Does not swap frontier and next; the caller does that.
uint64_t expand_level_cpu(const Graph& g, BfsState& st, size_t fbegin, size_t fend,
                          int threads);

// Runs the whole BFS with all frontier vertices. Returns dist as plain int32.
std::vector<int32_t> bfs_cpu(const Graph& g, vid_t src, int threads);
