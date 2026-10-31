#pragma once

#include "types.h"
#include <string>

struct GraphBuildStats {
  vid_t n = 0;
  eid_t m = 0;                      // directed entries after symmetrization and dedup
  uint64_t self_loops_removed = 0;
  uint64_t duplicates_removed = 0;
  uint64_t renumbered = 0;
};

// Builds a Graph in CSR format from a SNAP-format text file using two passes.
Graph build_csr_from_snap(const std::string& filepath, GraphBuildStats& stats);
