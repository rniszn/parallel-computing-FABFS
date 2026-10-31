#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <sys/stat.h>

#include "common/bfs_state.h"
#include "cpu/bfs_omp.h"
#include "graph/csr.h"
#include "graph/loader.h"
#include "graph/types.h"
#include "ref/bfs_ref.h"

namespace {

bool file_exists(const std::string& path) {
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
}

struct Fixture {
  Graph g;
  vid_t source;
  std::vector<int32_t> expected;
};

Fixture load_fixture(const std::string& edges_path, const std::string& expect_path) {
  std::ifstream efile(edges_path);
  if (!efile) {
    throw std::runtime_error("Cannot open edges file: " + edges_path);
  }

  vid_t n = 0;
  eid_t m = 0;
  if (!(efile >> n >> m)) {
    throw std::runtime_error("Malformed header in: " + edges_path);
  }

  std::vector<std::pair<vid_t, vid_t>> raw_edges;
  raw_edges.reserve(m * 2);
  vid_t u = 0;
  vid_t v = 0;
  while (efile >> u >> v) {
    if (u != v) {
      raw_edges.emplace_back(u, v);
      raw_edges.emplace_back(v, u);
    }
  }

  std::sort(raw_edges.begin(), raw_edges.end());
  raw_edges.erase(std::unique(raw_edges.begin(), raw_edges.end()), raw_edges.end());

  Graph g;
  g.n = n;
  g.m = raw_edges.size();
  g.row_offsets.assign(n + 1, 0);
  g.col_indices.resize(g.m);

  for (const auto& e : raw_edges) {
    g.row_offsets[e.first + 1]++;
  }
  for (vid_t i = 0; i < n; ++i) {
    g.row_offsets[i + 1] += g.row_offsets[i];
  }

  std::vector<eid_t> cursor = g.row_offsets;
  for (const auto& e : raw_edges) {
    g.col_indices[cursor[e.first]++] = e.second;
  }

  std::ifstream expfile(expect_path);
  if (!expfile) {
    throw std::runtime_error("Cannot open expect file: " + expect_path);
  }

  std::string tag;
  vid_t src = 0;
  if (!(expfile >> tag >> src) || tag != "source") {
    throw std::runtime_error("Malformed expect header in: " + expect_path);
  }

  std::vector<int32_t> expected;
  expected.reserve(n);
  int32_t d = 0;
  while (expfile >> d) {
    expected.push_back(d);
  }

  return {g, src, expected};
}

std::string get_graph_file_path(const std::string& name) {
  if (name == "roadNet-CA") return "data/roadNet-CA.txt";
  if (name == "web-Google") return "data/web-Google.txt";
  if (name == "com-lj") return "data/com-lj.ungraph.txt";
  if (name == "com-orkut") return "data/com-orkut.ungraph.txt";
  if (name == "roadNet-PA") return "data/roadNet-PA.txt";
  if (name == "com-amazon") return "data/com-amazon.ungraph.txt";
  if (name == "com-dblp") return "data/com-dblp.ungraph.txt";
  if (name.rfind("rmat_", 0) == 0) return "data/" + name + ".txt";
  return "data/" + name + ".txt";
}

}  // namespace

int main(int argc, char** argv) {
  bool sanitizer_mode = false;
  std::string filter_graph = "";

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--sanitizer" || arg == "--quick") {
      sanitizer_mode = true;
    } else if (arg.rfind("--graph=", 0) == 0) {
      filter_graph = arg.substr(8);
    }
  }

  std::cout << "==========================================" << std::endl;
  std::cout << "Running M3 CPU BFS Test Suite" << std::endl;
  if (sanitizer_mode) {
    std::cout << "Mode: Sanitizer / Quick verification" << std::endl;
  }
  std::cout << "==========================================" << std::endl;

  bool all_passed = true;

  // Test 1: Managed check with cudaPointerGetAttributes
  {
    std::cout << "[Test 1] Managed Memory Attributes Check: ";
    Graph g;
    g.n = 10;
    g.m = 0;
    g.row_offsets.assign(11, 0);

    BfsState st = make_state(g, 0);
    cudaPointerAttributes attr_dist{};
    cudaPointerAttributes attr_frontier{};
    cudaPointerAttributes attr_next{};

    cudaError_t e1 = cudaPointerGetAttributes(&attr_dist, st.dist);
    cudaError_t e2 = cudaPointerGetAttributes(&attr_frontier, st.frontier);
    cudaError_t e3 = cudaPointerGetAttributes(&attr_next, st.next);

    bool ok1 = (e1 == cudaSuccess && attr_dist.type == cudaMemoryTypeManaged);
    bool ok2 = (e2 == cudaSuccess && attr_frontier.type == cudaMemoryTypeManaged);
    bool ok3 = (e3 == cudaSuccess && attr_next.type == cudaMemoryTypeManaged);

    free_state(st);

    if (ok1 && ok2 && ok3) {
      std::cout << "PASS (dist, frontier, next are cudaMemoryTypeManaged)" << std::endl;
    } else {
      std::cout << "FAIL (type codes: dist=" << attr_dist.type
                << ", frontier=" << attr_frontier.type
                << ", next=" << attr_next.type << ")" << std::endl;
      all_passed = false;
    }
  }

  // Test 2: Hand fixtures from M2 with threads = 1 and threads = 4
  {
    std::cout << "[Test 2] Hand fixtures (7) with threads = 1 and 4: ";
    const std::vector<std::string> hand_names = {
        "path5", "star6", "star6_leaf", "cycle6", "k4", "two_components", "isolated"};

    bool test2_ok = true;
    for (const auto& name : hand_names) {
      std::string edges_p = "tests/fixtures/hand/" + name + ".edges";
      std::string expect_p = "tests/fixtures/hand/" + name + ".expect";
      Fixture f = load_fixture(edges_p, expect_p);

      for (int t : {1, 4}) {
        auto got = bfs_cpu(f.g, f.source, t);
        Mismatch m = verify(f.expected, got);
        if (!m.ok) {
          std::cout << "\n  FAIL on " << name << " (t=" << t << ") at v=" << m.vertex
                    << " exp=" << m.expected << " got=" << m.got;
          test2_ok = false;
        }
      }
    }
    if (test2_ok) {
      std::cout << "PASS (all matched bfs_ref and expect)" << std::endl;
    } else {
      std::cout << " -> FAILED" << std::endl;
      all_passed = false;
    }
  }

  // Test 3: Isolated source edge case
  {
    std::cout << "[Test 3] Isolated source edge case: ";
    Fixture f = load_fixture("tests/fixtures/hand/isolated.edges",
                             "tests/fixtures/hand/isolated.expect");
    // Vertex 3 has degree 0
    auto dist = bfs_cpu(f.g, 3, 4);
    bool test3_ok = (dist[3] == 0 && dist[0] == -1 && dist[1] == -1 && dist[2] == -1);
    if (test3_ok) {
      std::cout << "PASS (dist[3]=0, all others -1)" << std::endl;
    } else {
      std::cout << "FAIL" << std::endl;
      all_passed = false;
    }
  }

  // Test 4: Range splitting test
  {
    std::cout << "[Test 4] Range splitting test (frontier >= 100): ";
    // Use rmat10 seed1
    Fixture f = load_fixture("tests/fixtures/rmat10/seed1.edges",
                             "tests/fixtures/rmat10/seed1.expect");

    // Expand level 0 to get level 1 frontier
    BfsState stA = make_state(f.g, 0);
    expand_level_cpu(f.g, stA, 0, stA.frontier_size, 4);
    std::swap(stA.frontier, stA.next);
    stA.frontier_size = stA.next_size;
    stA.next_size = 0;
    stA.level++;

    size_t fsize = stA.frontier_size;
    if (fsize < 100) {
      std::cout << "FAIL (frontier size " << fsize << " < 100)" << std::endl;
      all_passed = false;
    } else {
      // Create duplicate state stB
      BfsState stB = make_state(f.g, 0);
      for (size_t i = 0; i < f.g.n; ++i) stB.dist[i] = stA.dist[i];
      for (size_t i = 0; i < fsize; ++i) stB.frontier[i] = stA.frontier[i];
      stB.frontier_size = fsize;
      stB.level = stA.level;

      // Expand stA on full range [0, fsize)
      expand_level_cpu(f.g, stA, 0, fsize, 4);

      // Expand stB in two halves: [0, fsize/2) and [fsize/2, fsize)
      size_t mid = fsize / 2;
      expand_level_cpu(f.g, stB, 0, mid, 4);
      expand_level_cpu(f.g, stB, mid, fsize, 4);

      std::vector<vid_t> nextA(stA.next, stA.next + stA.next_size);
      std::vector<vid_t> nextB(stB.next, stB.next + stB.next_size);
      std::sort(nextA.begin(), nextA.end());
      std::sort(nextB.begin(), nextB.end());

      bool match = (nextA == nextB);
      free_state(stA);
      free_state(stB);

      if (match) {
        std::cout << "PASS (frontier size " << fsize
                  << ", both halves produced identical sorted next list of "
                  << nextA.size() << " vertices)" << std::endl;
      } else {
        std::cout << "FAIL (sizeA=" << nextA.size() << ", sizeB=" << nextB.size() << ")" << std::endl;
        all_passed = false;
      }
    }
  }

  // Test 5: All 20 NetworkX R-MAT10 fixtures for threads in {1, 2, 4, 8}
  {
    std::cout << "[Test 5] 20 NetworkX R-MAT10 fixtures for threads in {1, 2, 4, 8}: ";
    bool test5_ok = true;
    for (int k = 1; k <= 20; ++k) {
      std::string edges_p = "tests/fixtures/rmat10/seed" + std::to_string(k) + ".edges";
      std::string expect_p = "tests/fixtures/rmat10/seed" + std::to_string(k) + ".expect";
      Fixture f = load_fixture(edges_p, expect_p);

      for (int t : {1, 2, 4, 8}) {
        auto got = bfs_cpu(f.g, f.source, t);
        Mismatch m = verify(f.expected, got);
        if (!m.ok) {
          std::cout << "\n  FAIL seed " << k << " t=" << t << " at v=" << m.vertex
                    << " exp=" << m.expected << " got=" << m.got;
          test5_ok = false;
        }
      }
    }
    if (test5_ok) {
      std::cout << "PASS (all 20 seeds x 4 thread counts matched)" << std::endl;
    } else {
      std::cout << " -> FAILED" << std::endl;
      all_passed = false;
    }
  }

  // Test 6: Evaluation and Calibration graphs from results/graphs.csv
  {
    std::cout << "\n--- Evaluation & Calibration Graphs (results/graphs.csv) ---" << std::endl;
    std::ifstream csv_file("results/graphs.csv");
    if (!csv_file) {
      std::cerr << "Warning: Cannot open results/graphs.csv" << std::endl;
    } else {
      std::string line;
      // Skip header line
      std::getline(csv_file, line);

      std::vector<std::string> graph_names;
      while (std::getline(csv_file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string name;
        if (std::getline(ss, name, ',')) {
          graph_names.push_back(name);
        }
      }

      int tested_count = 0;
      int skipped_count = 0;

      for (const auto& name : graph_names) {
        if (!filter_graph.empty() && name != filter_graph) {
          continue;
        }

        std::string filepath = get_graph_file_path(name);
        if (!file_exists(filepath)) {
          std::cout << "[SKIPPED] " << name << " (" << filepath << " missing)" << std::endl;
          skipped_count++;
          continue;
        }

        // Under sanitizer mode, skip very large graphs (com-orkut, com-lj) to prevent out-of-memory
        if (sanitizer_mode && (name == "com-orkut" || name == "com-lj")) {
          std::cout << "[SKIPPED-SANITIZER] " << name << " (skipped under sanitizer to prevent OOM)" << std::endl;
          skipped_count++;
          continue;
        }

        std::cout << "Testing graph: " << name << " ... " << std::flush;
        GraphBuildStats stats;
        Graph g = load_snap_graph(filepath, stats);

        // Ground truth reference BFS
        auto ref = bfs_ref(g, 0);

        // Verify across thread counts {1, 2, 4, 8}
        bool graph_ok = true;
        for (int t : {1, 2, 4, 8}) {
          auto got = bfs_cpu(g, 0, t);
          Mismatch m = verify(ref, got);
          if (!m.ok) {
            std::cout << "\n  FAIL on " << name << " t=" << t << " at v=" << m.vertex
                      << " exp=" << m.expected << " got=" << m.got;
            graph_ok = false;
          }
        }

        if (graph_ok) {
          std::cout << "PASS (t={1,2,4,8} verified against bfs_ref)" << std::endl;
          tested_count++;
        } else {
          all_passed = false;
        }

        // Timing benchmark for rmat_s16_a0.45 or rmat_a0.45
        if (name == "rmat_s16_a0.45" || name == "rmat_a0.45") {
          std::cout << "  Timing for " << name << ":" << std::endl;
          for (int t : {1, 2, 4, 8}) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto got = bfs_cpu(g, 0, t);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::cout << "    threads=" << t << ": " << ms << " ms" << std::endl;
          }
        }

        // 10 repeats on web-Google (or rmat_s16_a0.45)
        if (name == "web-Google" || (sanitizer_mode && name == "rmat_s16_a0.45")) {
          std::cout << "  Running 10 repeat consistency check on " << name << " (threads=8): ";
          bool rep_ok = true;
          for (int r = 0; r < 10; ++r) {
            auto run_dist = bfs_cpu(g, 0, 8);
            Mismatch m = verify(ref, run_dist);
            if (!m.ok) {
              std::cout << "FAIL at run " << r << std::endl;
              rep_ok = false;
              break;
            }
          }
          if (rep_ok) {
            std::cout << "PASS (10/10 identical)" << std::endl;
          } else {
            all_passed = false;
          }
        }
      }

      std::cout << "\nGraphs tested: " << tested_count << ", Skipped: " << skipped_count << std::endl;
    }
  }

  std::cout << "\n==========================================" << std::endl;
  if (all_passed) {
    std::cout << "ALL M3 CPU BFS TESTS PASSED" << std::endl;
    std::cout << "==========================================" << std::endl;
    return 0;
  } else {
    std::cout << "SOME TESTS FAILED" << std::endl;
    std::cout << "==========================================" << std::endl;
    return 1;
  }
}
