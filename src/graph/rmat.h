#pragma once

#include "csr.h"
#include "types.h"
#include <cstdint>
#include <string>

void generate_rmat_file(const std::string& output_path,
                        uint32_t scale,
                        uint32_t edge_factor,
                        double a,
                        uint64_t seed = 12345);

Graph generate_or_load_rmat(const std::string& output_path,
                            uint32_t scale,
                            uint32_t edge_factor,
                            double a,
                            uint64_t seed,
                            GraphBuildStats& stats);
