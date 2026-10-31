#pragma once

#include "csr.h"
#include "types.h"
#include <string>

Graph load_snap_graph(const std::string& filepath, GraphBuildStats& stats);
