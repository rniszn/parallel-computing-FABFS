#pragma once

#include <cstdint>
#include <set>
#include <string>
#include <vector>

struct CompletedRunRecord {
  std::string run_id;
  std::string status;  // "complete", "timeout"
  std::string graph;
  std::string policy;
  uint32_t round{0};
};

// Reads results/runs.jsonl and returns the set of completed run_ids (status == "complete" or "timeout")
std::set<std::string> load_completed_run_ids(const std::string& runs_jsonl_path);

// Appends a single completed run record to results/runs.jsonl and immediately fsyncs
void append_run_status(const std::string& runs_jsonl_path,
                       const std::string& run_id,
                       const std::string& status,
                       const std::string& graph = "",
                       const std::string& policy = "",
                       uint32_t round = 0);

// Scans the CSV log file and removes all rows belonging to uncompleted run_ids
// Returns number of orphaned/partial rows removed
size_t cleanup_partial_csv_rows(const std::string& csv_path,
                               const std::set<std::string>& completed_run_ids);

// Counts how many rows in the CSV belong to the given run_id
size_t count_csv_rows_for_run(const std::string& csv_path, const std::string& run_id);

// Counts total data rows (excluding header) in the CSV
size_t count_csv_total_rows(const std::string& csv_path);
