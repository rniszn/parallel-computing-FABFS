#include "loader.h"

Graph load_snap_graph(const std::string& filepath, GraphBuildStats& stats) {
  return build_csr_from_snap(filepath, stats);
}
