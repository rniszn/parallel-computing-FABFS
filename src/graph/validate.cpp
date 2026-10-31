#include "validate.h"
#include <algorithm>
#include <string>

std::string validate_graph(const Graph& g) {
  // Check 1: row_offsets sizes, row_offsets[0] == 0, non-decreasing, row_offsets[n] == m
  if (g.row_offsets.size() != static_cast<size_t>(g.n) + 1) {
    return "row_offsets size is not n+1";
  }
  if (g.col_indices.size() != static_cast<size_t>(g.m)) {
    return "col_indices size does not match m";
  }
  if (g.row_offsets[0] != 0) {
    return "row_offsets[0] is not 0";
  }
  for (vid_t i = 0; i < g.n; ++i) {
    if (g.row_offsets[i + 1] < g.row_offsets[i]) {
      return "row_offsets is decreasing at vertex " + std::to_string(i);
    }
  }
  if (g.row_offsets[g.n] != g.m) {
    return "row_offsets[n] != m";
  }

  // Check 2: every col_indices[i] < n
  for (eid_t i = 0; i < g.m; ++i) {
    if (g.col_indices[i] >= g.n) {
      return "col_indices out of range: neighbor " + std::to_string(g.col_indices[i]) +
             " >= n (" + std::to_string(g.n) + ")";
    }
  }

  // Check 3: no self-loops
  for (vid_t u = 0; u < g.n; ++u) {
    eid_t start = g.row_offsets[u];
    eid_t end = g.row_offsets[u + 1];
    for (eid_t e = start; e < end; ++e) {
      if (g.col_indices[e] == u) {
        return "self-loop detected at vertex " + std::to_string(u);
      }
    }
  }

  // Check 4: each neighbor list strictly increasing (no duplicates)
  for (vid_t u = 0; u < g.n; ++u) {
    eid_t start = g.row_offsets[u];
    eid_t end = g.row_offsets[u + 1];
    for (eid_t e = start; e + 1 < end; ++e) {
      if (g.col_indices[e] >= g.col_indices[e + 1]) {
        return "neighbor list not strictly increasing for vertex " + std::to_string(u) +
               ": " + std::to_string(g.col_indices[e]) + " >= " +
               std::to_string(g.col_indices[e + 1]);
      }
    }
  }

  // Check 5: symmetry: for every (u, v), u appears in v's list (binary search)
  for (vid_t u = 0; u < g.n; ++u) {
    eid_t start = g.row_offsets[u];
    eid_t end = g.row_offsets[u + 1];
    for (eid_t e = start; e < end; ++e) {
      vid_t v = g.col_indices[e];
      eid_t v_start = g.row_offsets[v];
      eid_t v_end = g.row_offsets[v + 1];
      auto it_begin = g.col_indices.begin() + v_start;
      auto it_end = g.col_indices.begin() + v_end;
      if (!std::binary_search(it_begin, it_end, u)) {
        return "asymmetric edge: (" + std::to_string(u) + ", " + std::to_string(v) +
               ") exists but (" + std::to_string(v) + ", " + std::to_string(u) + ") is missing";
      }
    }
  }

  return "ok";
}
