#pragma once

#include <cstdint>
#include <vector>
#include "../graph/types.h"

// dist[v] = hops from src, or -1 if unreachable. Input must be symmetric.
// Throws std::out_of_range if src >= g.n.
std::vector<int32_t> bfs_ref(const Graph& g, vid_t src);

struct Mismatch {
  bool ok;           // true if the two arrays are identical
  vid_t vertex;      // first vertex that differs (valid when !ok)
  int32_t expected;  // value in the reference array
  int32_t got;       // value in the array being checked
};

// Returns the first difference. If the lengths differ, returns ok=false
// with vertex=0, expected=-999, got=-999.
Mismatch verify(const std::vector<int32_t>& ref, const std::vector<int32_t>& got);
