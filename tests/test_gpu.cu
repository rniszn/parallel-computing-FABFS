// Sanitizer note: racecheck covers shared memory only.
// It does not check global-memory atomics, so do not rely on it for dist or counts.

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
#include "gpu/bfs_cuda.h"
#include "gpu/graph_gpu.h"
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
    if (arg == "--sanitizer") {
      sanitizer_mode = true;
    } else if (arg.rfind("--graph=", 0) == 0) {
      filter_graph = arg.substr(8);
    }
  }

  std::cout << "==========================================" << std::endl;
  std::cout << "Running M4 GPU BFS Test Suite" << std::endl;
  std::cout << "==========================================" << std::endl;

  bool all_passed = true;

  // Test 1: Managed memory pointer check
  {
    std::cout << "[Test 1] Managed Memory Attributes Check: ";
    Graph dummy;
    dummy.n = 10;
    dummy.m = 0;
    dummy.row_offsets.assign(11, 0);

    BfsState st = make_state(dummy, 0);

    cudaPointerAttributes attr_dist, attr_front, attr_next;
    cudaPointerGetAttributes(&attr_dist, st.dist);
    cudaPointerGetAttributes(&attr_front, st.frontier);
    cudaPointerGetAttributes(&attr_next, st.next);

    bool ok_dist = (attr_dist.type == cudaMemoryTypeManaged);
    bool ok_front = (attr_front.type == cudaMemoryTypeManaged);
    bool ok_next = (attr_next.type == cudaMemoryTypeManaged);

    free_state(st);

    if (ok_dist && ok_front && ok_next) {
      std::cout << "PASS (dist, frontier, next are cudaMemoryTypeManaged)" << std::endl;
    } else {
      std::cout << "FAIL (types: dist=" << attr_dist.type
                << ", front=" << attr_front.type
                << ", next=" << attr_next.type << ")" << std::endl;
      all_passed = false;
    }
  }

  // Test 2: Hand fixtures (7)
  {
    std::cout << "[Test 2] Hand fixtures (7): ";
    std::vector<std::string> hand_names = {
        "path5", "star6", "star6_leaf", "cycle6", "k4", "two_components", "isolated"};

    bool test2_ok = true;
    for (const auto& name : hand_names) {
      std::string edges_p = "tests/fixtures/hand/" + name + ".edges";
      std::string expect_p = "tests/fixtures/hand/" + name + ".expect";
      Fixture f = load_fixture(edges_p, expect_p);

      auto got = bfs_gpu(f.g, f.source);
      auto ref = bfs_ref(f.g, f.source);

      Mismatch m_expect = verify(f.expected, got);
      Mismatch m_ref = verify(ref, got);

      if (!m_expect.ok || !m_ref.ok) {
        std::cout << "\n  FAIL on fixture " << name;
        if (!m_expect.ok) {
          std::cout << " (mismatch with .expect at v=" << m_expect.vertex
                    << " exp=" << m_expect.expected << " got=" << m_expect.got << ")";
        }
        if (!m_ref.ok) {
          std::cout << " (mismatch with bfs_ref at v=" << m_ref.vertex
                    << " ref=" << m_ref.expected << " got=" << m_ref.got << ")";
        }
        test2_ok = false;
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
    auto dist = bfs_gpu(f.g, 3);
    bool test3_ok = (dist[3] == 0 && dist[0] == -1 && dist[1] == -1 && dist[2] == -1);
    if (test3_ok) {
      std::cout << "PASS (dist[3]=0, all others -1)" << std::endl;
    } else {
      std::cout << "FAIL" << std::endl;
      all_passed = false;
    }
  }

  // Test 4: Range splitting test (frontier >= 100)
  {
    std::cout << "[Test 4] Range splitting test (frontier >= 100): ";
    Fixture f = load_fixture("tests/fixtures/rmat10/seed1.edges",
                             "tests/fixtures/rmat10/seed1.expect");

    GpuGraph gg = upload_graph(f.g);

    // Expand level 0 on GPU to get level 1 frontier
    BfsState stA = make_state(f.g, 0);
    expand_level_gpu(gg, stA, 0, stA.frontier_size);
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
      expand_level_gpu(gg, stA, 0, fsize);

      // Expand stB in two calls: [0, fsize/2) and [fsize/2, fsize)
      size_t mid = fsize / 2;
      expand_level_gpu(gg, stB, 0, mid);
      std::vector<vid_t> nextB(stB.next, stB.next + stB.next_size);

      expand_level_gpu(gg, stB, mid, fsize);
      nextB.insert(nextB.end(), stB.next, stB.next + stB.next_size);

      std::vector<vid_t> nextA(stA.next, stA.next + stA.next_size);
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
    free_graph(gg);
  }

  // Test 5: Shared state CPU then GPU
  {
    std::cout << "[Test 5] Shared state interoperability (CPU level 0 -> GPU level 1): ";
    Fixture f = load_fixture("tests/fixtures/rmat10/seed1.edges",
                             "tests/fixtures/rmat10/seed1.expect");
    GpuGraph gg = upload_graph(f.g);
    BfsState st = make_state(f.g, f.source);

    // CPU level 0
    expand_level_cpu(f.g, st, 0, st.frontier_size, 4);
    std::swap(st.frontier, st.next);
    st.frontier_size = st.next_size;
    st.next_size = 0;
    st.level++;

    // GPU level 1
    if (st.frontier_size > 0) {
      expand_level_gpu(gg, st, 0, st.frontier_size);
      std::swap(st.frontier, st.next);
      st.frontier_size = st.next_size;
      st.next_size = 0;
      st.level++;
    }

    // Complete remaining levels on GPU
    while (st.frontier_size > 0) {
      expand_level_gpu(gg, st, 0, st.frontier_size);
      std::swap(st.frontier, st.next);
      st.frontier_size = st.next_size;
      st.next_size = 0;
      st.level++;
    }

    std::vector<int32_t> got(st.dist, st.dist + f.g.n);
    free_state(st);
    free_graph(gg);

    auto ref = bfs_ref(f.g, f.source);
    Mismatch m = verify(ref, got);
    if (m.ok) {
      std::cout << "PASS (verified against bfs_ref)" << std::endl;
    } else {
      std::cout << "FAIL at v=" << m.vertex << " exp=" << m.expected << " got=" << m.got << std::endl;
      all_passed = false;
    }
  }

  // Test 6: Shared state GPU then CPU
  {
    std::cout << "[Test 6] Shared state interoperability (GPU level 0 -> CPU level 1): ";
    Fixture f = load_fixture("tests/fixtures/rmat10/seed1.edges",
                             "tests/fixtures/rmat10/seed1.expect");
    GpuGraph gg = upload_graph(f.g);
    BfsState st = make_state(f.g, f.source);

    // GPU level 0
    expand_level_gpu(gg, st, 0, st.frontier_size);
    std::swap(st.frontier, st.next);
    st.frontier_size = st.next_size;
    st.next_size = 0;
    st.level++;

    // CPU level 1
    if (st.frontier_size > 0) {
      expand_level_cpu(f.g, st, 0, st.frontier_size, 4);
      std::swap(st.frontier, st.next);
      st.frontier_size = st.next_size;
      st.next_size = 0;
      st.level++;
    }

    // Complete remaining levels on CPU
    while (st.frontier_size > 0) {
      expand_level_cpu(f.g, st, 0, st.frontier_size, 4);
      std::swap(st.frontier, st.next);
      st.frontier_size = st.next_size;
      st.next_size = 0;
      st.level++;
    }

    std::vector<int32_t> got(st.dist, st.dist + f.g.n);
    free_state(st);
    free_graph(gg);

    auto ref = bfs_ref(f.g, f.source);
    Mismatch m = verify(ref, got);
    if (m.ok) {
      std::cout << "PASS (verified against bfs_ref)" << std::endl;
    } else {
      std::cout << "FAIL at v=" << m.vertex << " exp=" << m.expected << " got=" << m.got << std::endl;
      all_passed = false;
    }
  }

  // Test 7: 20 NetworkX R-MAT10 fixtures
  {
    std::cout << "[Test 7] 20 NetworkX R-MAT10 fixtures: ";
    bool test7_ok = true;
    for (int k = 1; k <= 20; ++k) {
      std::string edges_p = "tests/fixtures/rmat10/seed" + std::to_string(k) + ".edges";
      std::string expect_p = "tests/fixtures/rmat10/seed" + std::to_string(k) + ".expect";
      Fixture f = load_fixture(edges_p, expect_p);

      auto got = bfs_gpu(f.g, f.source);
      Mismatch m = verify(f.expected, got);
      if (!m.ok) {
        std::cout << "\n  FAIL seed " << k << " at v=" << m.vertex
                  << " exp=" << m.expected << " got=" << m.got;
        test7_ok = false;
      }
    }
    if (test7_ok) {
      std::cout << "PASS (all 20 seeds matched)" << std::endl;
    } else {
      std::cout << " -> FAILED" << std::endl;
      all_passed = false;
    }
  }

  // Test 8: Evaluation & Calibration graphs from results/graphs.csv
  {
    std::cout << "\n--- Evaluation & Calibration Graphs (results/graphs.csv) ---" << std::endl;
    std::ifstream csv_file("results/graphs.csv");
    if (!csv_file) {
      std::cerr << "Warning: Cannot open results/graphs.csv" << std::endl;
    } else {
      std::string line;
      std::getline(csv_file, line);  // Skip header

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

        if (sanitizer_mode && (name == "com-orkut" || name == "com-lj")) {
          std::cout << "[SKIPPED-SANITIZER] " << name << " (skipped under sanitizer to prevent OOM/timeout)" << std::endl;
          skipped_count++;
          continue;
        }

        std::cout << "Testing graph: " << name << " ... " << std::flush;
        GraphBuildStats stats;
        Graph g = load_snap_graph(filepath, stats);

        // Ground truth reference BFS
        auto ref = bfs_ref(g, 0);

        // GPU BFS
        auto got = bfs_gpu(g, 0);
        Mismatch m = verify(ref, got);
        if (!m.ok) {
          std::cout << "\n  FAIL on " << name << " at v=" << m.vertex
                    << " exp=" << m.expected << " got=" << m.got << std::endl;
          all_passed = false;
          continue;
        }

        std::cout << "PASS (verified against bfs_ref)" << std::endl;
        tested_count++;

        // 10 repeats on each graph (or web-Google)
        if (!sanitizer_mode || name == "web-Google" || name == "rmat_s16_a0.45") {
          std::cout << "  Running 10 repeat consistency check on " << name << ": " << std::flush;
          bool rep_ok = true;
          for (int r = 0; r < 10; ++r) {
            auto run_dist = bfs_gpu(g, 0);
            Mismatch mr = verify(ref, run_dist);
            if (!mr.ok) {
              std::cout << "FAIL at run " << r << " (diff at v=" << mr.vertex << ")" << std::endl;
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

        // Timing benchmark for rmat graphs
        if (name == "rmat_a0.45" || name == "rmat_s16_a0.45") {
          GpuGraph gg = upload_graph(g);
          auto t0 = std::chrono::high_resolution_clock::now();
          BfsState st = make_state(g, 0);
          while (st.frontier_size > 0) {
            expand_level_gpu(gg, st, 0, st.frontier_size);
            std::swap(st.frontier, st.next);
            st.frontier_size = st.next_size;
            st.next_size = 0;
            st.level++;
          }
          auto t1 = std::chrono::high_resolution_clock::now();
          double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
          std::cout << "  GPU BFS timing for " << name << ": " << ms << " ms" << std::endl;
          free_state(st);
          free_graph(gg);
        }

        // com-orkut level-by-level timing with cudaEvent
        if (name == "com-orkut" && !sanitizer_mode) {
          std::cout << "  Recording per-level timings for com-orkut..." << std::endl;
          GpuGraph gg = upload_graph(g);
          BfsState st = make_state(g, 0);

          std::ofstream ofs("results/gpu_levels_com-orkut.csv");
          ofs << "level,frontier_size,edges,ms\n";

          cudaEvent_t ev_start, ev_stop;
          cudaEventCreate(&ev_start);
          cudaEventCreate(&ev_stop);

          while (st.frontier_size > 0) {
            size_t cur_fsize = st.frontier_size;
            int32_t cur_level = st.level;

            cudaEventRecord(ev_start, 0);
            uint64_t edges = expand_level_gpu(gg, st, 0, cur_fsize);
            cudaEventRecord(ev_stop, 0);
            cudaEventSynchronize(ev_stop);

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, ev_start, ev_stop);

            ofs << cur_level << "," << cur_fsize << "," << edges << "," << ms << "\n";
            std::cout << "    level " << cur_level << ": frontier=" << cur_fsize
                      << " edges=" << edges << " " << ms << " ms" << std::endl;

            std::swap(st.frontier, st.next);
            st.frontier_size = st.next_size;
            st.next_size = 0;
            st.level++;
          }

          cudaEventDestroy(ev_start);
          cudaEventDestroy(ev_stop);
          free_state(st);
          free_graph(gg);
          ofs.close();
          std::cout << "  Saved per-level timings to results/gpu_levels_com-orkut.csv" << std::endl;
        }
      }

      std::cout << "\nGraphs tested: " << tested_count << ", Skipped: " << skipped_count << std::endl;
    }
  }

  std::cout << "\n==========================================" << std::endl;
  if (all_passed) {
    std::cout << "ALL M4 GPU BFS TESTS PASSED" << std::endl;
    std::cout << "==========================================" << std::endl;
    return 0;
  } else {
    std::cout << "SOME TESTS FAILED" << std::endl;
    std::cout << "==========================================" << std::endl;
    return 1;
  }
}
