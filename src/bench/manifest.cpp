#include "manifest.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>

const std::vector<std::string> EVALUATION_GRAPHS = {
    "roadNet-CA",
    "web-Google",
    "com-lj",
    "com-orkut",
    "rmat_a0.25",
    "rmat_a0.45",
    "rmat_a0.55",
    "rmat_a0.65"
};

const std::vector<std::string> CALIBRATION_GRAPHS = {
    "roadNet-PA",
    "com-amazon",
    "com-dblp",
    "rmat_s16_a0.45"
};

const std::vector<std::string> POLICIES = {
    "Adaptive",
    "Pure-Static",
    "Pure-Dynamic",
    "Oracle-Replay"
};

bool is_calibration_graph(const std::string& graph_name) {
  std::string lower = graph_name;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

  for (const auto& cg : CALIBRATION_GRAPHS) {
    std::string cg_lower = cg;
    std::transform(cg_lower.begin(), cg_lower.end(), cg_lower.begin(), ::tolower);
    if (lower == cg_lower || lower.find(cg_lower) != std::string::npos) {
      return true;
    }
  }

  // Also check common aliases
  if (lower.find("amazon") != std::string::npos ||
      lower.find("dblp") != std::string::npos ||
      lower.find("roadnet-pa") != std::string::npos ||
      lower.find("rmat_s16") != std::string::npos ||
      lower.find("scale_16") != std::string::npos) {
    return true;
  }
  return false;
}

std::vector<BenchmarkRun> generate_evaluation_manifest(uint32_t seed, uint32_t num_rounds) {
  std::vector<BenchmarkRun> runs;
  runs.reserve(EVALUATION_GRAPHS.size() * POLICIES.size() * num_rounds);

  uint32_t id = 1;
  for (const auto& g : EVALUATION_GRAPHS) {
    for (const auto& p : POLICIES) {
      for (uint32_t r = 1; r <= num_rounds; ++r) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "run_%03u", id++);
        runs.push_back({std::string(buf), p, g, r});
      }
    }
  }

  // Deterministic shuffle with the specified seed
  std::mt19937 rng(seed);
  std::shuffle(runs.begin(), runs.end(), rng);

  return runs;
}

std::vector<BenchmarkRun> generate_dry_run_manifest(uint32_t seed) {
  // Dry run: 1 round, 2 graphs (one small SNAP e.g. web-Google or roadNet-CA, one small synthetic R-MAT), all 4 policies
  // For the dry-run, we use 2 non-calibration graphs to test the full pipeline
  std::vector<std::string> dry_graphs = {"roadNet-CA", "rmat_s16_a0.45"}; // R-MAT s16 used explicitly for dry run
  std::vector<BenchmarkRun> runs;

  uint32_t id = 1;
  for (const auto& g : dry_graphs) {
    for (const auto& p : POLICIES) {
      char buf[32];
      std::snprintf(buf, sizeof(buf), "dry_run_%03u", id++);
      runs.push_back({std::string(buf), p, g, 1});
    }
  }

  std::mt19937 rng(seed);
  std::shuffle(runs.begin(), runs.end(), rng);
  return runs;
}

bool validate_manifest(const std::vector<BenchmarkRun>& runs, std::string* error_msg, size_t expected_count, bool is_dry_run) {
  if (expected_count > 0 && runs.size() != expected_count) {
    if (error_msg) *error_msg = "Manifest entry count mismatch: expected " + std::to_string(expected_count) + ", got " + std::to_string(runs.size());
    return false;
  }

  std::set<std::string> seen_ids;
  for (const auto& run : runs) {
    if (run.run_id.empty()) {
      if (error_msg) *error_msg = "Encountered empty run_id in manifest";
      return false;
    }
    if (seen_ids.count(run.run_id) > 0) {
      if (error_msg) *error_msg = "Duplicate run_id in manifest: " + run.run_id;
      return false;
    }
    seen_ids.insert(run.run_id);

    // Strict Leakage Guard: Check for calibration graphs in evaluation manifests
    if (!is_dry_run && is_calibration_graph(run.graph)) {
      if (error_msg) *error_msg = "LEAKAGE VIOLATION: Calibration graph detected in evaluation manifest: " + run.graph;
      return false;
    }
  }

  return true;
}

void save_manifest_json(const std::vector<BenchmarkRun>& runs, const std::string& filepath) {
  std::ofstream out(filepath);
  if (!out.is_open()) {
    throw std::runtime_error("Failed to open file for writing manifest JSON: " + filepath);
  }

  out << "[\n";
  for (size_t i = 0; i < runs.size(); ++i) {
    const auto& r = runs[i];
    out << "  {\n";
    out << "    \"run_id\": \"" << r.run_id << "\",\n";
    out << "    \"policy\": \"" << r.policy << "\",\n";
    out << "    \"graph\": \"" << r.graph << "\",\n";
    out << "    \"round\": " << r.round << "\n";
    out << "  }" << (i + 1 < runs.size() ? ",\n" : "\n");
  }
  out << "]\n";
}

std::vector<BenchmarkRun> load_manifest_json(const std::string& filepath) {
  std::ifstream in(filepath);
  if (!in.is_open()) {
    throw std::runtime_error("Failed to open file for reading manifest JSON: " + filepath);
  }

  std::vector<BenchmarkRun> runs;
  std::string line;
  BenchmarkRun current;
  bool in_entry = false;

  while (std::getline(in, line)) {
    if (line.find('{') != std::string::npos) {
      in_entry = true;
      current = BenchmarkRun{};
    } else if (line.find('}') != std::string::npos) {
      if (in_entry && !current.run_id.empty()) {
        runs.push_back(current);
      }
      in_entry = false;
    } else if (in_entry) {
      auto extract_string = [](const std::string& l, const std::string& key) -> std::string {
        size_t kpos = l.find("\"" + key + "\"");
        if (kpos == std::string::npos) return "";
        size_t colon = l.find(':', kpos);
        if (colon == std::string::npos) return "";
        size_t q1 = l.find('\"', colon);
        if (q1 == std::string::npos) return "";
        size_t q2 = l.find('\"', q1 + 1);
        if (q2 == std::string::npos) return "";
        return l.substr(q1 + 1, q2 - q1 - 1);
      };

      auto extract_uint = [](const std::string& l, const std::string& key) -> uint32_t {
        size_t kpos = l.find("\"" + key + "\"");
        if (kpos == std::string::npos) return 0;
        size_t colon = l.find(':', kpos);
        if (colon == std::string::npos) return 0;
        std::string sub = l.substr(colon + 1);
        return static_cast<uint32_t>(std::strtoul(sub.c_str(), nullptr, 10));
      };

      if (line.find("\"run_id\"") != std::string::npos) {
        current.run_id = extract_string(line, "run_id");
      } else if (line.find("\"policy\"") != std::string::npos) {
        current.policy = extract_string(line, "policy");
      } else if (line.find("\"graph\"") != std::string::npos) {
        current.graph = extract_string(line, "graph");
      } else if (line.find("\"round\"") != std::string::npos) {
        current.round = extract_uint(line, "round");
      }
    }
  }

  return runs;
}

std::string compute_sha256(const std::string& filepath_or_content, bool is_raw_content) {
  std::string cmd;
  if (is_raw_content) {
    // Escape single quotes for bash
    std::string escaped;
    for (char c : filepath_or_content) {
      if (c == '\'') escaped += "'\\''";
      else escaped += c;
    }
    cmd = "printf '%s' '" + escaped + "' | sha256sum 2>/dev/null";
  } else {
    cmd = "sha256sum \"" + filepath_or_content + "\" 2>/dev/null";
  }

  FILE* fp = popen(cmd.c_str(), "r");
  if (!fp) return "";

  char buf[128];
  std::string hash = "";
  if (std::fgets(buf, sizeof(buf), fp)) {
    hash = std::string(buf).substr(0, 64);
  }
  pclose(fp);
  return hash;
}
