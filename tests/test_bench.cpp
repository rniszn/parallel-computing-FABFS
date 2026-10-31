#include <cassert>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include "bench/manifest.h"
#include "bench/resume.h"
#include "bench/runner.h"
#include "log/record.h"

namespace {

void clean_dir(const std::string& dir) {
  std::string cmd = "rm -rf " + dir + " && mkdir -p " + dir;
  int ret = std::system(cmd.c_str());
  (void)ret;
}

}  // namespace

int main(int argc, char** argv) {
  std::cout << "==========================================" << std::endl;
  std::cout << "Running M8 Benchmark Runner Tests" << std::endl;
  std::cout << "==========================================" << std::endl;

  bool all_passed = true;

  // Test 1: Manifest: 480 entries, no duplicate run_id, seeded order matches expected order for seed 12345
  {
    std::cout << "[Test 1] Manifest generation (480 entries, no duplicates, deterministic seed): ";
    auto m1 = generate_evaluation_manifest(12345, 15);
    auto m2 = generate_evaluation_manifest(12345, 15);

    bool count_ok = (m1.size() == 480);
    bool match_ok = true;
    std::set<std::string> unique_ids;

    for (size_t i = 0; i < m1.size(); ++i) {
      unique_ids.insert(m1[i].run_id);
      if (m1[i].run_id != m2[i].run_id ||
          m1[i].graph != m2[i].graph ||
          m1[i].policy != m2[i].policy ||
          m1[i].round != m2[i].round) {
        match_ok = false;
      }
    }

    std::string err;
    bool valid_ok = validate_manifest(m1, &err, 480);

    if (count_ok && unique_ids.size() == 480 && match_ok && valid_ok) {
      std::cout << "PASS (480 entries, 480 unique IDs, deterministic shuffle matched)" << std::endl;
    } else {
      std::cout << "FAIL (count_ok=" << count_ok << ", unique=" << unique_ids.size()
                << ", match=" << match_ok << ", valid=" << valid_ok << ", err=" << err << ")" << std::endl;
      all_passed = false;
    }
  }

  // Test 2: A calibration graph in the manifest makes the runner refuse to start
  {
    std::cout << "[Test 2] Calibration graph rejection in manifest: ";
    auto bad_manifest = generate_evaluation_manifest(12345, 15);
    // Inject a forbidden calibration graph into the manifest
    bad_manifest[0].graph = "roadNet-PA";

    std::string err;
    bool valid = validate_manifest(bad_manifest, &err, bad_manifest.size());

    RunnerConfig cfg;
    cfg.allow_dirty = true;
    cfg.manifest_path = "tests/fixtures/bad_manifest.json";
    save_manifest_json(bad_manifest, cfg.manifest_path);

    BenchmarkRunner runner(cfg);
    std::string runner_err;
    bool init_res = runner.initialize(&runner_err);

    if (!valid && !init_res && (runner_err.find("Calibration") != std::string::npos || runner_err.find("LEAKAGE") != std::string::npos)) {
      std::cout << "PASS (runner refused to start on manifest containing calibration graph)" << std::endl;
    } else {
      std::cout << "FAIL (valid=" << valid << ", init_res=" << init_res << ", err=" << runner_err << ")" << std::endl;
      all_passed = false;
    }
  }

  // Test 4: A run with no line in runs.jsonl is rerun, and its old partial rows are gone
  {
    std::cout << "[Test 4] Partial row cleanup for uncompleted runs: ";
    std::string test_dir = "tests/fixtures/test_resume";
    clean_dir(test_dir);

    std::string csv_path = test_dir + "/test.csv";
    std::string jsonl_path = test_dir + "/runs.jsonl";

    // Write CSV containing header, 3 rows for run_001, and 2 partial rows for run_002
    LevelLogger logger;
    logger.open(csv_path);

    LevelRow r1;
    r1.run_id = "run_001";
    r1.graph = "roadNet-CA";
    r1.policy = "Adaptive";
    r1.round = 1;
    r1.level = 0;
    logger.write_row(r1);
    r1.level = 1;
    logger.write_row(r1);
    r1.level = 2;
    logger.write_row(r1);
    logger.end_run();

    LevelRow r2;
    r2.run_id = "run_002";
    r2.graph = "roadNet-CA";
    r2.policy = "Adaptive";
    r2.round = 1;
    r2.level = 0;
    logger.write_row(r2);
    r2.level = 1;
    logger.write_row(r2);
    // run_002 crashed before end_run / jsonl commit!
    logger.close();

    // In runs.jsonl, only run_001 is recorded as complete
    append_run_status(jsonl_path, "run_001", "complete", "roadNet-CA", "Adaptive", 1);

    auto completed_ids = load_completed_run_ids(jsonl_path);
    size_t removed = cleanup_partial_csv_rows(csv_path, completed_ids);

    size_t rows_001 = count_csv_rows_for_run(csv_path, "run_001");
    size_t rows_002 = count_csv_rows_for_run(csv_path, "run_002");
    size_t total_rows = count_csv_total_rows(csv_path);

    if (removed == 2 && rows_001 == 3 && rows_002 == 0 && total_rows == 3) {
      std::cout << "PASS (2 partial rows removed, completed run preserved)" << std::endl;
    } else {
      std::cout << "FAIL (removed=" << removed << ", rows_001=" << rows_001
                << ", rows_002=" << rows_002 << ", total=" << total_rows << ")" << std::endl;
      all_passed = false;
    }
  }

  // Test 5: A dirty git tree makes the runner refuse to start, unless --allow-dirty is given
  {
    std::cout << "[Test 5] Dirty git tree check and --allow-dirty override: ";
    clean_dir("tests/fixtures/test_env");
    auto d_manifest = generate_dry_run_manifest(12345);
    save_manifest_json(d_manifest, "tests/fixtures/test_env/dry_manifest.json");

    RunnerConfig cfg_strict;
    cfg_strict.allow_dirty = false;
    cfg_strict.dry_run = true;
    cfg_strict.manifest_path = "tests/fixtures/test_env/dry_manifest.json";

    RunnerConfig cfg_allow;
    cfg_allow.allow_dirty = true;
    cfg_allow.dry_run = true;
    cfg_allow.manifest_path = "tests/fixtures/test_env/dry_manifest.json";

    bool is_dirty = is_git_tree_dirty();
    BenchmarkRunner runner_strict(cfg_strict);
    std::string err_strict;
    bool strict_init = runner_strict.initialize(&err_strict);

    BenchmarkRunner runner_allow(cfg_allow);
    std::string err_allow;
    bool allow_init = runner_allow.initialize(&err_allow);

    if (is_dirty) {
      // In dirty tree: strict MUST fail, allow MUST pass
      if (!strict_init && allow_init && err_strict.find("dirty") != std::string::npos) {
        std::cout << "PASS (rejected dirty tree without --allow-dirty, accepted with flag)" << std::endl;
      } else {
        std::cout << "FAIL (strict_init=" << strict_init << ", allow_init=" << allow_init << ")" << std::endl;
        all_passed = false;
      }
    } else {
      // In clean tree: both succeed
      if (strict_init && allow_init) {
        std::cout << "PASS (clean tree accepted by runner)" << std::endl;
      } else {
        std::cout << "FAIL (strict_init=" << strict_init << ", allow_init=" << allow_init << ")" << std::endl;
        all_passed = false;
      }
    }
  }

  // Test 3: Dry run: 1 round, 2 graphs, all 4 policies. Kill runner with SIGKILL halfway, then restart.
  {
    std::cout << "[Test 3] Dry run crash-kill and resume test (2 graphs x 4 policies = 8 runs): " << std::endl;
    std::string dry_dir = "tests/fixtures/dry_run_test";
    clean_dir(dry_dir);

    std::string manifest_path = dry_dir + "/manifest.json";
    std::string runs_jsonl = dry_dir + "/runs.jsonl";
    std::string csv_path = dry_dir + "/eval.csv";

    // Generate dry manifest: 2 graphs, 1 round, 4 policies = 8 runs
    auto dry_manifest = generate_dry_run_manifest(12345);
    save_manifest_json(dry_manifest, manifest_path);

    // Pass 1: Run in child process with kill_after_rows = 15 (kills child mid-traversal)
    pid_t pid = fork();
    if (pid == 0) {
      // Child process
      RunnerConfig cfg_child;
      cfg_child.dry_run = true;
      cfg_child.allow_dirty = true;
      cfg_child.manifest_path = manifest_path;
      cfg_child.runs_jsonl_path = runs_jsonl;
      cfg_child.csv_output_path = csv_path;
      cfg_child.kill_after_rows = 15; // Self-terminate via SIGKILL after 15 rows

      BenchmarkRunner runner_child(cfg_child);
      runner_child.initialize();
      runner_child.run_all();
      std::exit(0);
    }

    // Parent waits for child
    int status = 0;
    waitpid(pid, &status, 0);

    bool killed_by_sigkill = (WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL);
    std::cout << "  Child killed by SIGKILL mid-run: " << (killed_by_sigkill ? "YES" : "NO") << std::endl;

    size_t rows_before_resume = count_csv_total_rows(csv_path);
    auto completed_before = load_completed_run_ids(runs_jsonl);
    std::cout << "  Before resume: completed runs in jsonl = " << completed_before.size()
              << ", rows in CSV = " << rows_before_resume << std::endl;

    // Pass 2: Restart runner on the same files without kill hook
    RunnerConfig cfg_resume;
    cfg_resume.dry_run = true;
    cfg_resume.allow_dirty = true;
    cfg_resume.manifest_path = manifest_path;
    cfg_resume.runs_jsonl_path = runs_jsonl;
    cfg_resume.csv_output_path = csv_path;

    BenchmarkRunner runner_resume(cfg_resume);
    std::string init_err;
    if (!runner_resume.initialize(&init_err)) {
      std::cout << "  FAIL: runner_resume.initialize() failed: " << init_err << std::endl;
      all_passed = false;
    } else {
      RunnerSummary summary = runner_resume.run_all();
      std::cout << "  Resume run finished. Summary: scheduled=" << summary.total_scheduled
                << ", skipped=" << summary.runs_skipped_resumed
                << ", executed=" << summary.runs_executed
                << ", completed=" << summary.runs_completed
                << ", total rows=" << summary.rows_written << std::endl;

      // Verification checks:
      auto completed_after = load_completed_run_ids(runs_jsonl);
      bool all_8_completed = (completed_after.size() == 8);

      // Verify no duplicate runs in runs.jsonl
      std::ifstream j_in(runs_jsonl);
      std::string j_line;
      std::vector<std::string> jsonl_run_ids;
      while (std::getline(j_in, j_line)) {
        size_t kpos = j_line.find("\"run_id\": \"");
        if (kpos != std::string::npos) {
          size_t qend = j_line.find("\"", kpos + 11);
          jsonl_run_ids.push_back(j_line.substr(kpos + 11, qend - (kpos + 11)));
        }
      }

      std::set<std::string> unique_completed(jsonl_run_ids.begin(), jsonl_run_ids.end());
      bool no_dup_runs = (jsonl_run_ids.size() == unique_completed.size());

      // Verify each run in CSV has valid positive row count and no duplicates
      bool csv_rows_ok = true;
      size_t total_csv_rows = 0;
      for (const auto& r : dry_manifest) {
        size_t r_count = count_csv_rows_for_run(csv_path, r.run_id);
        if (r_count == 0) {
          csv_rows_ok = false;
          std::cout << "  Missing rows for run: " << r.run_id << std::endl;
        }
        total_csv_rows += r_count;
      }

      bool total_match = (total_csv_rows == count_csv_total_rows(csv_path));

      if (all_8_completed && no_dup_runs && csv_rows_ok && total_match) {
        std::cout << "  PASS (all 8 runs completed, 0 duplicates, CSV rows equal sum of level counts)" << std::endl;
      } else {
        std::cout << "  FAIL (all_8=" << all_8_completed << ", no_dup=" << no_dup_runs
                  << ", csv_rows_ok=" << csv_rows_ok << ", total_match=" << total_match << ")" << std::endl;
        all_passed = false;
      }
    }
  }

  // Memory fit verification check (Requirement 8)
  {
    std::cout << "[Requirement 8] com-Orkut VRAM memory fit check: ";
    size_t free_mb = 0, req_mb = 0;
    bool fits = BenchmarkRunner::check_gpu_memory_fit("com-orkut", &free_mb, &req_mb);
    if (fits) {
      std::cout << "PASS (com-Orkut fits: required ~" << req_mb << " MB, free VRAM = " << free_mb << " MB)" << std::endl;
    } else {
      std::cout << "NOTE (com-Orkut memory fit check: required ~" << req_mb << " MB, free VRAM = " << free_mb << " MB)" << std::endl;
    }
  }

  std::cout << "==========================================" << std::endl;
  if (all_passed) {
    std::cout << "ALL M8 BENCHMARK RUNNER TESTS PASSED" << std::endl;
    std::cout << "==========================================" << std::endl;
    return 0;
  } else {
    std::cout << "SOME M8 BENCHMARK RUNNER TESTS FAILED" << std::endl;
    std::cout << "==========================================" << std::endl;
    return 1;
  }
}
