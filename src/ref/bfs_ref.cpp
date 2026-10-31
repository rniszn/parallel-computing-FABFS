#include "bfs_ref.h"
#include <stdexcept>

std::vector<int32_t> bfs_ref(const Graph& g, vid_t src) {
  if (src >= g.n) {
    throw std::out_of_range("Source vertex out of range");
  }

  std::vector<int32_t> dist(g.n, -1);
  dist[src] = 0;

  std::vector<vid_t> queue;
  queue.push_back(src);
  size_t head = 0;

  while (head < queue.size()) {
    vid_t u = queue[head++];
    for (eid_t i = g.row_offsets[u]; i < g.row_offsets[u + 1]; ++i) {
      vid_t v = g.col_indices[i];
      if (dist[v] == -1) {
        dist[v] = dist[u] + 1;
        queue.push_back(v);
      }
    }
  }

  return dist;
}

Mismatch verify(const std::vector<int32_t>& ref, const std::vector<int32_t>& got) {
  if (ref.size() != got.size()) {
    return {false, 0, -999, -999};
  }

  for (size_t i = 0; i < ref.size(); ++i) {
    if (ref[i] != got[i]) {
      return {false, static_cast<vid_t>(i), ref[i], got[i]};
    }
  }

  return {true, 0, 0, 0};
}
