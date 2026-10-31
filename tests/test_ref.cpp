#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "graph/types.h"
#include "ref/bfs_ref.h"

namespace {

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

  if (expected.size() != n) {
    throw std::runtime_error("Expected array size mismatch in " + expect_path +
                             ": got " + std::to_string(expected.size()) +
                             ", expected " + std::to_string(n));
  }

  return {g, src, expected};
}

}  // namespace

int main() {
  bool all_passed = true;

  std::cout << "Running M2 Reference BFS tests..." << std::endl;

  // Test 1: Hand-made fixtures match .expect exactly
  {
    std::cout << "[Test 1] Hand-made fixtures match .expect exactly: ";
    const std::vector<std::string> hand_names = {
        "path5", "star6", "star6_leaf", "cycle6", "k4", "two_components", "isolated"};

    bool test1_ok = true;
    for (const auto& name : hand_names) {
      std::string edges_p = "tests/fixtures/hand/" + name + ".edges";
      std::string expect_p = "tests/fixtures/hand/" + name + ".expect";
      Fixture f = load_fixture(edges_p, expect_p);
      auto dist = bfs_ref(f.g, f.source);
      Mismatch m = verify(f.expected, dist);
      if (!m.ok) {
        std::cout << "\n  FAIL on fixture " << name << " at vertex " << m.vertex
                  << ": expected " << m.expected << ", got " << m.got;
        test1_ok = false;
      }
    }
    if (test1_ok) {
      std::cout << "PASS (all 7 hand fixtures matched)" << std::endl;
    } else {
      std::cout << " -> FAILED" << std::endl;
      all_passed = false;
    }
  }

  // Test 2: The two-components fixture returns -1 for every vertex in the other component
  {
    std::cout << "[Test 2] two_components fixture unreachable component is -1: ";
    Fixture f = load_fixture("tests/fixtures/hand/two_components.edges",
                             "tests/fixtures/hand/two_components.expect");
    auto dist = bfs_ref(f.g, f.source);
    bool test2_ok = (dist[3] == -1 && dist[4] == -1 && dist[5] == -1 &&
                     dist[0] == 0 && dist[1] == 1 && dist[2] == 1);
    if (test2_ok) {
      std::cout << "PASS (vertices 3, 4, 5 are -1)" << std::endl;
    } else {
      std::cout << "FAIL" << std::endl;
      all_passed = false;
    }
  }

  // Test 3: The isolated-vertex fixture returns -1 for that vertex
  {
    std::cout << "[Test 3] isolated fixture returns -1 for vertex 3: ";
    Fixture f = load_fixture("tests/fixtures/hand/isolated.edges",
                             "tests/fixtures/hand/isolated.expect");
    auto dist = bfs_ref(f.g, f.source);
    bool test3_ok = (dist[3] == -1 && dist[0] == 0 && dist[1] == 1 && dist[2] == 2);
    if (test3_ok) {
      std::cout << "PASS (isolated vertex 3 is -1)" << std::endl;
    } else {
      std::cout << "FAIL" << std::endl;
      all_passed = false;
    }
  }

  // Test 4: Each of the 20 NetworkX fixtures matches bfs_ref with verify().ok == true
  {
    std::cout << "[Test 4] 20 NetworkX R-MAT (scale 10) fixtures match bfs_ref: ";
    bool test4_ok = true;
    for (int k = 1; k <= 20; ++k) {
      std::string edges_p = "tests/fixtures/rmat10/seed" + std::to_string(k) + ".edges";
      std::string expect_p = "tests/fixtures/rmat10/seed" + std::to_string(k) + ".expect";
      Fixture f = load_fixture(edges_p, expect_p);
      auto dist = bfs_ref(f.g, f.source);
      Mismatch m = verify(f.expected, dist);
      if (!m.ok) {
        std::cout << "\n  FAIL on seed " << k << " at vertex " << m.vertex
                  << ": expected " << m.expected << ", got " << m.got;
        test4_ok = false;
      }
    }
    if (test4_ok) {
      std::cout << "PASS (all 20 seeds matched)" << std::endl;
    } else {
      std::cout << " -> FAILED" << std::endl;
      all_passed = false;
    }
  }

  // Test 5: Deliberate error detection
  {
    std::cout << "[Test 5] Deliberate error detection with verify(): ";
    Fixture f = load_fixture("tests/fixtures/rmat10/seed1.edges",
                             "tests/fixtures/rmat10/seed1.expect");
    auto dist = bfs_ref(f.g, f.source);
    // Copy reference array and perturb vertex 5
    const vid_t target_vertex = 5;
    std::vector<int32_t> wrong = dist;
    int32_t original_val = wrong[target_vertex];
    wrong[target_vertex] += 1;

    Mismatch m = verify(dist, wrong);
    bool test5_ok = (!m.ok && m.vertex == target_vertex &&
                     m.expected == original_val && m.got == original_val + 1);
    if (test5_ok) {
      std::cout << "PASS (caught mismatch at vertex 5: expected "
                << m.expected << ", got " << m.got << ")" << std::endl;
    } else {
      std::cout << "FAIL (ok=" << m.ok << ", vertex=" << m.vertex
                << ", exp=" << m.expected << ", got=" << m.got << ")" << std::endl;
      all_passed = false;
    }
  }

  // Test 6: verify on arrays of different lengths
  {
    std::cout << "[Test 6] verify on arrays of different lengths: ";
    std::vector<int32_t> a = {0, 1, 2, 3};
    std::vector<int32_t> b = {0, 1, 2};
    Mismatch m = verify(a, b);
    bool test6_ok = (!m.ok && m.vertex == 0 && m.expected == -999 && m.got == -999);
    if (test6_ok) {
      std::cout << "PASS (ok=false, vertex=0, expected=-999, got=-999)" << std::endl;
    } else {
      std::cout << "FAIL" << std::endl;
      all_passed = false;
    }
  }

  // Test 7: bfs_ref with src >= n throws std::out_of_range
  {
    std::cout << "[Test 7] bfs_ref with src >= n throws std::out_of_range: ";
    Fixture f = load_fixture("tests/fixtures/hand/path5.edges",
                             "tests/fixtures/hand/path5.expect");
    bool caught = false;
    try {
      bfs_ref(f.g, f.g.n);
    } catch (const std::out_of_range& e) {
      caught = true;
    } catch (...) {
      caught = false;
    }

    if (caught) {
      std::cout << "PASS (std::out_of_range thrown and caught)" << std::endl;
    } else {
      std::cout << "FAIL (expected std::out_of_range)" << std::endl;
      all_passed = false;
    }
  }

  std::cout << "\n==========================================" << std::endl;
  if (all_passed) {
    std::cout << "ALL 7 TESTS PASSED" << std::endl;
    std::cout << "==========================================" << std::endl;
    return 0;
  } else {
    std::cout << "SOME TESTS FAILED" << std::endl;
    std::cout << "==========================================" << std::endl;
    return 1;
  }
}
