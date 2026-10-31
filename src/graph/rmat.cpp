#include "rmat.h"
#include <cstdio>
#include <random>
#include <stdexcept>
#include <sys/stat.h>
#include <vector>

namespace {

bool file_exists(const std::string& path) {
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
}

}  // namespace

void generate_rmat_file(const std::string& output_path,
                        uint32_t scale,
                        uint32_t edge_factor,
                        double a,
                        uint64_t seed) {
  FILE* fp = std::fopen(output_path.c_str(), "w");
  if (!fp) {
    throw std::runtime_error("Cannot create file: " + output_path);
  }

  std::vector<char> buffer(2 * 1024 * 1024);
  std::setvbuf(fp, buffer.data(), _IOFBF, buffer.size());

  double b = (1.0 - a) / 3.0;
  double c = b;
  double cum_a = a;
  double cum_ab = a + b;
  double cum_abc = a + b + c;

  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  uint64_t num_edges = static_cast<uint64_t>(edge_factor) * (1ULL << scale);

  for (uint64_t e = 0; e < num_edges; ++e) {
    uint32_t u = 0;
    uint32_t v = 0;
    for (int bit = static_cast<int>(scale) - 1; bit >= 0; --bit) {
      double p = dist(rng);
      if (p < cum_a) {
        // quadrant (0, 0)
      } else if (p < cum_ab) {
        // quadrant (0, 1)
        v |= (1U << bit);
      } else if (p < cum_abc) {
        // quadrant (1, 0)
        u |= (1U << bit);
      } else {
        // quadrant (1, 1)
        u |= (1U << bit);
        v |= (1U << bit);
      }
    }
    std::fprintf(fp, "%u\t%u\n", u, v);
  }

  std::fclose(fp);
}

Graph generate_or_load_rmat(const std::string& output_path,
                            uint32_t scale,
                            uint32_t edge_factor,
                            double a,
                            uint64_t seed,
                            GraphBuildStats& stats) {
  if (!file_exists(output_path)) {
    generate_rmat_file(output_path, scale, edge_factor, a, seed);
  }
  return build_csr_from_snap(output_path, stats);
}
