#pragma once

#include <cstddef>
#include <cstdint>
#include "graph/types.h"

struct BfsState {
  int32_t* dist;               // managed, -1 = unvisited
  vid_t* frontier;             // managed, capacity n
  vid_t* next;                 // managed, capacity n
  unsigned long long* counts;  // managed, [0] = next count, [1] = edge count
  size_t frontier_size;
  size_t next_size;
  int32_t level;
};

BfsState make_state(const Graph& g, vid_t src);
void free_state(BfsState& st);
