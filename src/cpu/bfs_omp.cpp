#include "bfs_omp.h"
#include <algorithm>
#include <atomic>
#include <cstring>
#include <omp.h>
#include <stdexcept>
#include <utility>
#include <vector>

uint64_t expand_level_cpu(const Graph& g, BfsState& st, size_t fbegin, size_t fend,
                          int threads) {
  if (threads < 1) {
    threads = 1;
  }
  omp_set_num_threads(threads);

  if (fbegin >= fend || fbegin >= st.frontier_size) {
    return 0;
  }

  // Pre-size thread-local buffers before parallel region to avoid allocation in loop
  eid_t frontier_degree = 0;
  for (size_t i = fbegin; i < fend; ++i) {
    vid_t u = st.frontier[i];
    frontier_degree += (g.row_offsets[u + 1] - g.row_offsets[u]);
  }

  std::vector<std::vector<vid_t>> local_buffers(threads);
  size_t per_thread_reserve = (frontier_degree / static_cast<size_t>(threads)) + 64;
  for (int t = 0; t < threads; ++t) {
    local_buffers[t].reserve(per_thread_reserve);
  }

  std::vector<uint64_t> thread_edges(threads, 0);
  std::atomic<int> sync_threads{0};

  #pragma omp parallel num_threads(threads)
  {
    int tid = omp_get_thread_num();

    #pragma omp for schedule(static)
    for (size_t i = fbegin; i < fend; ++i) {
      vid_t u = st.frontier[i];
      eid_t start = g.row_offsets[u];
      eid_t end = g.row_offsets[u + 1];
      thread_edges[tid] += (end - start);

      for (eid_t e = start; e < end; ++e) {
        vid_t v = g.col_indices[e];
        int32_t expected = -1;
        if (__atomic_compare_exchange_n(&st.dist[v], &expected, st.level + 1,
                                        false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
          local_buffers[tid].push_back(v);
        }
      }
    }

    sync_threads.fetch_add(1, std::memory_order_release);
  }

  while (sync_threads.load(std::memory_order_acquire) < threads) {
  }

  uint64_t total_edges = 0;
  size_t total_new = 0;
  for (int t = 0; t < threads; ++t) {
    total_edges += thread_edges[t];
    total_new += local_buffers[t].size();
  }

  // After the join: copy local buffers into st.next in thread order using atomic reservation
  if (total_new > 0) {
    unsigned long long base_idx =
        __atomic_fetch_add(&st.counts[0], (unsigned long long)total_new, __ATOMIC_RELAXED);
    size_t cursor = base_idx;
    for (int t = 0; t < threads; ++t) {
      if (!local_buffers[t].empty()) {
        std::memcpy(st.next + cursor, local_buffers[t].data(),
                    local_buffers[t].size() * sizeof(vid_t));
        cursor += local_buffers[t].size();
      }
    }
  }

  __atomic_fetch_add(&st.counts[1], (unsigned long long)total_edges, __ATOMIC_RELAXED);
  st.next_size = st.counts[0];

  return total_edges;
}

std::vector<int32_t> bfs_cpu(const Graph& g, vid_t src, int threads) {
  if (src >= g.n) {
    throw std::out_of_range("Source vertex out of range");
  }

  BfsState st = make_state(g, src);

  while (st.frontier_size > 0) {
    reset_level(st);
    expand_level_cpu(g, st, 0, st.frontier_size, threads);
    std::swap(st.frontier, st.next);
    st.frontier_size = st.next_size;
    st.next_size = 0;
    st.level++;
  }

  std::vector<int32_t> dist(st.dist, st.dist + g.n);
  free_state(st);
  return dist;
}
