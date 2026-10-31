#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "csr.h"
#include "loader.h"
#include "rmat.h"
#include "types.h"
#include "validate.h"

namespace {

std::string compute_sha256(const std::string& filepath) {
  std::string cmd = "sha256sum \"" + filepath + "\"";
  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe) return "error";
  char buf[256];
  std::string result = "unknown";
  if (std::fgets(buf, sizeof(buf), pipe) != nullptr) {
    char hash[65] = {0};
    if (std::sscanf(buf, "%64s", hash) == 1) {
      result = hash;
    }
  }
  pclose(pipe);
  return result;
}

struct GraphMetrics {
  double mean_degree = 0.0;
  uint32_t max_degree = 0;
  double degree_cv = 0.0;
};

GraphMetrics compute_metrics(const Graph& g) {
  GraphMetrics m;
  if (g.n == 0) return m;

  m.mean_degree = static_cast<double>(g.m) / g.n;
  m.max_degree = 0;
  double sum_sq_diff = 0.0;

  for (vid_t u = 0; u < g.n; ++u) {
    uint32_t deg = static_cast<uint32_t>(g.row_offsets[u + 1] - g.row_offsets[u]);
    if (deg > m.max_degree) {
      m.max_degree = deg;
    }
    double diff = static_cast<double>(deg) - m.mean_degree;
    sum_sq_diff += diff * diff;
  }

  double variance = sum_sq_diff / g.n;
  double std_dev = std::sqrt(variance);
  m.degree_cv = (m.mean_degree > 0.0) ? (std_dev / m.mean_degree) : 0.0;
  return m;
}

void append_csv_row(const std::string& csv_path,
                    const std::string& graph_name,
                    const Graph& g,
                    const GraphBuildStats& stats,
                    const GraphMetrics& metrics,
                    const std::string& source,
                    const std::string& sha256) {
  bool file_exists = false;
  {
    std::ifstream test(csv_path);
    if (test.good()) file_exists = true;
  }

  std::ofstream ofs(csv_path, std::ios::app);
  if (!file_exists) {
    ofs << "graph,n,m_directed,mean_degree,max_degree,degree_cv,self_loops_removed,"
           "duplicates_removed,renumbered,source,sha256\n";
  }

  ofs << std::fixed << std::setprecision(4);
  ofs << graph_name << ","
      << g.n << ","
      << g.m << ","
      << metrics.mean_degree << ","
      << metrics.max_degree << ","
      << metrics.degree_cv << ","
      << stats.self_loops_removed << ","
      << stats.duplicates_removed << ","
      << stats.renumbered << ","
      << source << ","
      << sha256 << "\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <graph_name> <snap|rmat> <file_or_param> [scale] [edge_factor]\n";
    return 1;
  }

  std::string graph_name = argv[1];
  std::string type = argv[2];
  std::string path_or_param = argv[3];

  Graph g;
  GraphBuildStats stats;
  std::string source;
  std::string raw_file;

  if (type == "snap") {
    raw_file = path_or_param;
    source = "SNAP";
    std::cout << "Loading SNAP graph " << graph_name << " from " << raw_file << "...\n";
    g = load_snap_graph(raw_file, stats);
  } else if (type == "rmat") {
    source = "R-MAT";
    double a = std::stod(path_or_param);
    uint32_t scale = (argc >= 5) ? std::stoul(argv[4]) : 20;
    uint32_t edge_factor = (argc >= 6) ? std::stoul(argv[5]) : 16;
    raw_file = "data/" + graph_name + ".txt";
    std::cout << "Generating / loading R-MAT graph " << graph_name << " (scale=" << scale
              << ", edge_factor=" << edge_factor << ", a=" << a << ") to " << raw_file << "...\n";
    g = generate_or_load_rmat(raw_file, scale, edge_factor, a, 12345, stats);
  } else {
    std::cerr << "Unknown graph type: " << type << "\n";
    return 1;
  }

  // NEGATIVE TEST: Deliberately remove one direction of an edge to break symmetry
  if (g.n > 1 && g.row_offsets[1] > 0) {
    vid_t u = 0;
    vid_t v = g.col_indices[g.row_offsets[u]];
    eid_t v_start = g.row_offsets[v];
    eid_t v_end = g.row_offsets[v + 1];
    auto it = std::lower_bound(g.col_indices.begin() + v_start, g.col_indices.begin() + v_end, u);
    if (it != g.col_indices.begin() + v_end && *it == u) {
      g.col_indices.erase(it);
      g.m--;
      for (vid_t i = v + 1; i <= g.n; ++i) {
        g.row_offsets[i]--;
      }
    }
  }

  std::cout << "Validating graph " << graph_name << "...\n";
  std::string val = validate_graph(g);
  if (val != "ok") {
    std::cerr << "Validation FAILED for " << graph_name << ": " << val << "\n";
    return 2;
  }
  std::cout << "Validation PASSED for " << graph_name << ".\n";

  GraphMetrics metrics = compute_metrics(g);
  std::cout << "Computing SHA256 of " << raw_file << "...\n";
  std::string sha = compute_sha256(raw_file);

  std::cout << "Writing summary to results/graphs.csv...\n";
  append_csv_row("results/graphs.csv", graph_name, g, stats, metrics, source, sha);
  std::cout << "Successfully processed " << graph_name << ".\n";

  return 0;
}
