#include "resume.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#ifdef _WIN32
#include <io.h>
#define fsync _commit
#define fileno _fileno
#else
#include <unistd.h>
#endif

namespace {

std::string extract_json_field(const std::string& line, const std::string& field) {
  size_t kpos = line.find("\"" + field + "\"");
  if (kpos == std::string::npos) return "";
  size_t colon = line.find(':', kpos);
  if (colon == std::string::npos) return "";
  size_t q1 = line.find('\"', colon);
  if (q1 == std::string::npos) return "";
  size_t q2 = line.find('\"', q1 + 1);
  if (q2 == std::string::npos) return "";
  return line.substr(q1 + 1, q2 - q1 - 1);
}

}  // namespace

std::set<std::string> load_completed_run_ids(const std::string& runs_jsonl_path) {
  std::set<std::string> completed;
  std::ifstream in(runs_jsonl_path);
  if (!in.is_open()) {
    // If runs.jsonl does not exist yet, no runs have completed
    return completed;
  }

  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::string run_id = extract_json_field(line, "run_id");
    std::string status = extract_json_field(line, "status");

    // Both "complete" and "timeout" mark the run as finished (do not retry timeouts silently)
    if (!run_id.empty() && (status == "complete" || status == "timeout")) {
      completed.insert(run_id);
    }
  }

  return completed;
}

void append_run_status(const std::string& runs_jsonl_path,
                       const std::string& run_id,
                       const std::string& status,
                       const std::string& graph,
                       const std::string& policy,
                       uint32_t round) {
  FILE* fp = std::fopen(runs_jsonl_path.c_str(), "a");
  if (!fp) {
    throw std::runtime_error("Failed to open runs.jsonl for appending: " + runs_jsonl_path);
  }

  std::fprintf(fp,
               "{\"run_id\": \"%s\", \"status\": \"%s\", \"graph\": \"%s\", \"policy\": \"%s\", \"round\": %u}\n",
               run_id.c_str(), status.c_str(), graph.c_str(), policy.c_str(), round);

  std::fflush(fp);
  int fd = fileno(fp);
  if (fd != -1) {
    fsync(fd);
  }
  std::fclose(fp);
}

size_t cleanup_partial_csv_rows(const std::string& csv_path,
                               const std::set<std::string>& completed_run_ids) {
  std::ifstream in(csv_path);
  if (!in.is_open()) {
    return 0; // Nothing to clean
  }

  std::string header;
  if (!std::getline(in, header)) {
    in.close();
    return 0;
  }

  std::string tmp_path = csv_path + ".clean.tmp";
  std::ofstream out(tmp_path);
  if (!out.is_open()) {
    in.close();
    throw std::runtime_error("Failed to create temporary clean CSV: " + tmp_path);
  }

  out << header << "\n";

  size_t rows_removed = 0;
  std::string line;

  while (std::getline(in, line)) {
    if (line.empty()) continue;

    // Extract run_id from beginning of line (up to first comma)
    size_t comma = line.find(',');
    if (comma == std::string::npos) {
      // Malformed row, discard
      rows_removed++;
      continue;
    }

    std::string run_id = line.substr(0, comma);
    // Strip quotes if any
    if (run_id.size() >= 2 && run_id.front() == '"' && run_id.back() == '"') {
      run_id = run_id.substr(1, run_id.size() - 2);
    }

    if (completed_run_ids.count(run_id) > 0) {
      out << line << "\n";
    } else {
      // Incomplete or orphaned row from a crashed run
      rows_removed++;
    }
  }

  in.close();
  out.close();

  // Atomically replace original CSV with cleaned CSV
  std::remove(csv_path.c_str());
  std::rename(tmp_path.c_str(), csv_path.c_str());

  return rows_removed;
}

size_t count_csv_rows_for_run(const std::string& csv_path, const std::string& run_id) {
  std::ifstream in(csv_path);
  if (!in.is_open()) return 0;

  std::string line;
  if (!std::getline(in, line)) return 0; // Skip header

  size_t count = 0;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    size_t comma = line.find(',');
    if (comma != std::string::npos) {
      std::string id = line.substr(0, comma);
      if (id.size() >= 2 && id.front() == '"' && id.back() == '"') {
        id = id.substr(1, id.size() - 2);
      }
      if (id == run_id) {
        count++;
      }
    }
  }

  return count;
}

size_t count_csv_total_rows(const std::string& csv_path) {
  std::ifstream in(csv_path);
  if (!in.is_open()) return 0;

  std::string line;
  if (!std::getline(in, line)) return 0; // Skip header

  size_t count = 0;
  while (std::getline(in, line)) {
    if (!line.empty()) {
      count++;
    }
  }

  return count;
}
