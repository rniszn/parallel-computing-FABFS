#include "csr.h"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {

class FastEdgeReader {
 public:
  explicit FastEdgeReader(const std::string& path) : path_(path), fp_(nullptr) {
    fp_ = std::fopen(path.c_str(), "rb");
    if (!fp_) {
      throw std::runtime_error("Failed to open file: " + path);
    }
    // Set 2MB buffer for high-throughput disk I/O
    buffer_.resize(2 * 1024 * 1024);
    std::setvbuf(fp_, buffer_.data(), _IOFBF, buffer_.size());
  }

  ~FastEdgeReader() {
    if (fp_) {
      std::fclose(fp_);
    }
  }

  void rewind() {
    std::rewind(fp_);
  }

  // Returns true if an edge (u, v) was read, false at EOF
  bool next_edge(uint64_t& u, uint64_t& v) {
    while (true) {
      skip_whitespace();
      int c = std::fgetc(fp_);
      if (c == EOF) return false;

      if (c == '#') {
        // Skip comment line
        while (c != '\n' && c != EOF) {
          c = std::fgetc(fp_);
        }
        continue;
      }

      std::ungetc(c, fp_);

      if (!read_uint64(u)) return false;
      skip_inline_whitespace();
      if (!read_uint64(v)) return false;

      // Skip remainder of the line
      c = std::fgetc(fp_);
      while (c != '\n' && c != EOF) {
        c = std::fgetc(fp_);
      }
      return true;
    }
  }

 private:
  void skip_whitespace() {
    int c;
    while ((c = std::fgetc(fp_)) != EOF) {
      if (!std::isspace(c)) {
        std::ungetc(c, fp_);
        break;
      }
    }
  }

  void skip_inline_whitespace() {
    int c;
    while ((c = std::fgetc(fp_)) != EOF) {
      if (c != ' ' && c != '\t') {
        std::ungetc(c, fp_);
        break;
      }
    }
  }

  bool read_uint64(uint64_t& val) {
    int c = std::fgetc(fp_);
    if (c == EOF) return false;
    while (c != EOF && !std::isdigit(c)) {
      c = std::fgetc(fp_);
    }
    if (c == EOF) return false;

    val = 0;
    while (c != EOF && std::isdigit(c)) {
      val = val * 10 + (c - '0');
      c = std::fgetc(fp_);
    }
    if (c != EOF) {
      std::ungetc(c, fp_);
    }
    return true;
  }

  std::string path_;
  FILE* fp_;
  std::vector<char> buffer_;
};

}  // namespace

Graph build_csr_from_snap(const std::string& filepath, GraphBuildStats& stats) {
  FastEdgeReader reader(filepath);

  std::unordered_map<uint64_t, vid_t> id_to_dense;
  std::vector<uint64_t> dense_to_id;
  std::vector<eid_t> deg;

  stats.self_loops_removed = 0;
  stats.duplicates_removed = 0;
  stats.renumbered = 0;

  // --- PASS 1: Map vertex IDs and count degrees ---
  uint64_t u_raw = 0, v_raw = 0;
  while (reader.next_edge(u_raw, v_raw)) {
    if (u_raw == v_raw) {
      stats.self_loops_removed++;
      continue;
    }

    auto get_dense = [&](uint64_t raw) -> vid_t {
      auto it = id_to_dense.find(raw);
      if (it != id_to_dense.end()) {
        return it->second;
      }
      vid_t new_id = static_cast<vid_t>(dense_to_id.size());
      id_to_dense[raw] = new_id;
      dense_to_id.push_back(raw);
      deg.push_back(0);
      return new_id;
    };

    vid_t u = get_dense(u_raw);
    vid_t v = get_dense(v_raw);
    deg[u]++;
    deg[v]++;
  }

  vid_t n = static_cast<vid_t>(dense_to_id.size());
  for (vid_t i = 0; i < n; ++i) {
    if (dense_to_id[i] != static_cast<uint64_t>(i)) {
      stats.renumbered++;
    }
  }

  // Prefix sums into row_offsets
  Graph g;
  g.n = n;
  g.row_offsets.assign(n + 1, 0);
  for (vid_t i = 0; i < n; ++i) {
    g.row_offsets[i + 1] = g.row_offsets[i] + deg[i];
  }
  eid_t total_tentative = g.row_offsets[n];
  g.col_indices.resize(total_tentative);

  // Write cursors for Pass 2
  std::vector<eid_t> cursor = g.row_offsets;

  // --- PASS 2: Fill col_indices (symmetrizing both directions) ---
  reader.rewind();
  while (reader.next_edge(u_raw, v_raw)) {
    if (u_raw == v_raw) {
      continue;
    }
    vid_t u = id_to_dense[u_raw];
    vid_t v = id_to_dense[v_raw];
    g.col_indices[cursor[u]++] = v;
    g.col_indices[cursor[v]++] = u;
  }

  // Release ID mapping memory before sorting/compaction
  id_to_dense.clear();
  dense_to_id.clear();
  deg.clear();

  // --- In-place sort and deduplication per vertex ---
  std::vector<eid_t> new_offsets(g.n + 1, 0);
  eid_t write_idx = 0;

  for (vid_t u = 0; u < g.n; ++u) {
    eid_t start = g.row_offsets[u];
    eid_t end = g.row_offsets[u + 1];

    std::sort(g.col_indices.begin() + start, g.col_indices.begin() + end);

    for (eid_t e = start; e < end; ++e) {
      if (e == start || g.col_indices[e] != g.col_indices[e - 1]) {
        g.col_indices[write_idx++] = g.col_indices[e];
      } else {
        stats.duplicates_removed++;
      }
    }
    new_offsets[u + 1] = write_idx;
  }

  g.m = write_idx;
  g.col_indices.resize(write_idx);
  g.col_indices.shrink_to_fit();
  g.row_offsets = std::move(new_offsets);

  stats.n = g.n;
  stats.m = g.m;
  return g;
}
