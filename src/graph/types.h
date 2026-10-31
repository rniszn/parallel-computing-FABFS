#pragma once

#include <cstdint>
#include <vector>

using vid_t = uint32_t;
using eid_t = uint64_t;

struct Graph {
  vid_t n;
  eid_t m;                          // directed entries after symmetrization
  std::vector<eid_t> row_offsets;   // size n+1
  std::vector<vid_t> col_indices;   // size m
};
