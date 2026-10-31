#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../src/graph/csr.h"
#include "../src/graph/loader.h"
#include "../src/graph/rmat.h"
#include "../src/graph/types.h"
#include "../src/graph/validate.h"

void test1_hand_built_5_vertex() {
  std::cout << "[Test 1] Hand-built 5-vertex graph... ";
  // 5 vertices: 0, 1, 2, 3, 4
  // Edges: (0, 1), (0, 3), (1, 2), (2, 3), (3, 4)
  // Expected degree of:
  // 0: {1, 3} -> deg 2
  // 1: {0, 2} -> deg 2
  // 2: {1, 3} -> deg 2
  // 3: {0, 2, 4} -> deg 3
  // 4: {3} -> deg 1
  // Total m = 10
  std::string tmp_file = "test_hand_built.txt";
  {
    std::ofstream ofs(tmp_file);
    ofs << "0 1\n0 3\n1 2\n2 3\n3 4\n";
  }

  GraphBuildStats stats;
  Graph g = build_csr_from_snap(tmp_file, stats);
  std::remove(tmp_file.c_str());

  assert(g.n == 5);
  assert(g.m == 10);

  // Exact hand-computed values
  const std::vector<eid_t> expected_row_offsets = {0, 2, 4, 6, 9, 10};
  const std::vector<vid_t> expected_col_indices = {1, 3, 0, 2, 1, 3, 0, 2, 4, 3};

  assert(g.row_offsets == expected_row_offsets);
  assert(g.col_indices == expected_col_indices);
  assert(validate_graph(g) == "ok");
  std::cout << "PASSED\n";
}

void test2_validator_rejections() {
  std::cout << "[Test 2] Validator rejection of corrupt graphs... ";
  // Valid base graph
  Graph base;
  base.n = 3;
  base.m = 2;
  base.row_offsets = {0, 1, 2, 2};
  base.col_indices = {1, 0};
  assert(validate_graph(base) == "ok");

  // Corrupted A: decreasing row_offsets
  Graph bad_row = base;
  bad_row.row_offsets = {0, 2, 1, 2};
  std::string err_a = validate_graph(bad_row);
  assert(err_a.find("decreasing") != std::string::npos);

  // Corrupted B: out of range neighbor
  Graph bad_col = base;
  bad_col.col_indices = {3, 0};  // 3 >= n (n=3)
  std::string err_b = validate_graph(bad_col);
  assert(err_b.find("out of range") != std::string::npos);

  // Corrupted C: self-loop
  Graph bad_loop = base;
  bad_loop.col_indices = {0, 0};  // 0 -> 0 is a self loop
  std::string err_c = validate_graph(bad_loop);
  assert(err_c.find("self-loop") != std::string::npos);

  // Corrupted D: one-way edge (asymmetric)
  Graph bad_asym;
  bad_asym.n = 3;
  bad_asym.m = 1;
  bad_asym.row_offsets = {0, 1, 1, 1};
  bad_asym.col_indices = {1};  // (0, 1) exists but (1, 0) missing
  std::string err_d = validate_graph(bad_asym);
  assert(err_d.find("asymmetric") != std::string::npos);

  std::cout << "PASSED\n";
}

void test3_snap_loader_formatting() {
  std::cout << "[Test 3] SNAP parser with comments and whitespace... ";
  std::string tmp_file = "test_snap_format.txt";
  {
    std::ofstream ofs(tmp_file);
    ofs << "# SNAP Format Test File\n"
        << "# Directed / Undirected comments\n"
        << "  0\t1\n"
        << "\n"
        << "1   2\n"
        << "# inter-comment\n"
        << "2 \t 3\n";
  }

  GraphBuildStats stats;
  Graph g = build_csr_from_snap(tmp_file, stats);
  std::remove(tmp_file.c_str());

  assert(g.n == 4);
  assert(g.m == 6);
  // Expected CSR for 0-1-2-3 path:
  // 0: {1}
  // 1: {0, 2}
  // 2: {1, 3}
  // 3: {2}
  const std::vector<eid_t> expected_row_offsets = {0, 1, 3, 5, 6};
  const std::vector<vid_t> expected_col_indices = {1, 0, 2, 1, 3, 2};

  assert(g.row_offsets == expected_row_offsets);
  assert(g.col_indices == expected_col_indices);
  assert(validate_graph(g) == "ok");
  std::cout << "PASSED\n";
}

void test4_rmat_determinism() {
  std::cout << "[Test 4] R-MAT generation determinism with fixed seed... ";
  std::string path1 = "test_rmat1.txt";
  std::string path2 = "test_rmat2.txt";

  GraphBuildStats s1, s2;
  Graph g1 = generate_or_load_rmat(path1, 10, 16, 0.45, 12345, s1);
  Graph g2 = generate_or_load_rmat(path2, 10, 16, 0.45, 12345, s2);

  std::remove(path1.c_str());
  std::remove(path2.c_str());

  assert(g1.n == g2.n);
  assert(g1.m == g2.m);
  assert(g1.row_offsets == g2.row_offsets);
  assert(g1.col_indices == g2.col_indices);
  assert(validate_graph(g1) == "ok");
  assert(validate_graph(g2) == "ok");
  std::cout << "PASSED\n";
}

void test5_directed_symmetrization() {
  std::cout << "[Test 5] Directed input symmetrization... ";
  std::string tmp_file = "test_directed.txt";
  {
    std::ofstream ofs(tmp_file);
    // Directed triangle cycle: 0->1, 1->2, 2->0
    ofs << "0 1\n1 2\n2 0\n";
  }

  GraphBuildStats stats;
  Graph g = build_csr_from_snap(tmp_file, stats);
  std::remove(tmp_file.c_str());

  assert(g.n == 3);
  assert(g.m == 6);
  // After symmetrization:
  // 0: {1, 2}
  // 1: {0, 2}
  // 2: {0, 1}
  const std::vector<eid_t> expected_row_offsets = {0, 2, 4, 6};
  const std::vector<vid_t> expected_col_indices = {1, 2, 0, 2, 0, 1};

  assert(g.row_offsets == expected_row_offsets);
  assert(g.col_indices == expected_col_indices);
  assert(validate_graph(g) == "ok");
  std::cout << "PASSED\n";
}

int main() {
  std::cout << "=== Running M1 Unit Tests ===\n";
  test1_hand_built_5_vertex();
  test2_validator_rejections();
  test3_snap_loader_formatting();
  test4_rmat_determinism();
  test5_directed_symmetrization();
  std::cout << "=== ALL M1 TESTS PASSED ===\n";
  return 0;
}
