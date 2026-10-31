#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct BenchmarkRun {
  std::string run_id;
  std::string policy;  // "Adaptive", "Pure-Static", "Pure-Dynamic", "Oracle-Replay"
  std::string graph;   // e.g. "roadNet-CA", "rmat_a0.25", etc.
  uint32_t round{1};   // 1..15
};

// Generates the standard 480-run evaluation manifest shuffled deterministically with seed
std::vector<BenchmarkRun> generate_evaluation_manifest(uint32_t seed = 12345, uint32_t num_rounds = 15);

// Generates a dry-run manifest for quick testing (e.g. 2 graphs, 1 round, 4 policies)
std::vector<BenchmarkRun> generate_dry_run_manifest(uint32_t seed = 12345);

// Checks if graph name matches any calibration graph
bool is_calibration_graph(const std::string& graph_name);

// Validates the manifest: verifies entry count, checks for duplicates, and enforces leakage guard
bool validate_manifest(const std::vector<BenchmarkRun>& runs, std::string* error_msg = nullptr, size_t expected_count = 480, bool is_dry_run = false);

// Writes manifest to JSON file
void save_manifest_json(const std::vector<BenchmarkRun>& runs, const std::string& filepath);

// Loads manifest from JSON file
std::vector<BenchmarkRun> load_manifest_json(const std::string& filepath);

// Computes SHA256 of manifest file or input parameters
std::string compute_sha256(const std::string& filepath_or_content, bool is_raw_content = false);

// Standard list of evaluation and calibration graphs
extern const std::vector<std::string> EVALUATION_GRAPHS;
extern const std::vector<std::string> CALIBRATION_GRAPHS;
extern const std::vector<std::string> POLICIES;
