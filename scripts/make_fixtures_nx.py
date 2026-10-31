#!/usr/bin/env python3
"""
Generate scale-10 R-MAT test fixtures using NetworkX for cross-checking M2 bfs_ref.
Seeds: 1 to 20
Scale: 10 (N = 1024), Edge factor: 16 (M = 16384)
Parameters: a = 0.45, b = c = d = (1 - a) / 3
"""

import os
import sys
import numpy as np
import networkx as nx

def main():
    print(f"make_fixtures_nx.py: numpy={np.__version__}, networkx={nx.__version__}")

    scale = 10
    n = 1 << scale  # 1024
    edge_factor = 16
    num_edges = edge_factor * n  # 16384
    a = 0.45
    b = (1.0 - a) / 3.0
    c = b
    cum_a = a
    cum_ab = a + b
    cum_abc = a + b + c

    out_dir = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "rmat10")
    os.makedirs(out_dir, exist_ok=True)

    for seed in range(1, 21):
        rng = np.random.default_rng(seed)

        # Generate edges
        edges_set = set()
        for _ in range(num_edges):
            u = 0
            v = 0
            for bit in range(scale - 1, -1, -1):
                p = rng.random()
                if p < cum_a:
                    pass
                elif p < cum_ab:
                    v |= (1 << bit)
                elif p < cum_abc:
                    u |= (1 << bit)
                else:
                    u |= (1 << bit)
                    v |= (1 << bit)

            # Skip self loops
            if u == v:
                continue

            # Canonicalize undirected edge
            if u > v:
                u, v = v, u
            edges_set.add((u, v))

        sorted_edges = sorted(edges_set)
        m = len(sorted_edges)

        # Build NetworkX graph with all n nodes
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(sorted_edges)

        # Compute shortest paths from source 0
        src = 0
        lengths = nx.single_source_shortest_path_length(G, src)

        # Expected distances for all vertices 0..n-1
        dist = [lengths.get(i, -1) for i in range(n)]

        # Write seed<k>.edges
        edges_path = os.path.join(out_dir, f"seed{seed}.edges")
        with open(edges_path, "w") as f:
            f.write(f"{n} {m}\n")
            for u, v in sorted_edges:
                f.write(f"{u} {v}\n")

        # Write seed<k>.expect
        expect_path = os.path.join(out_dir, f"seed{seed}.expect")
        with open(expect_path, "w") as f:
            f.write(f"source {src}\n")
            for d in dist:
                f.write(f"{d}\n")

        reachable = sum(1 for d in dist if d != -1)
        max_dist = max(dist)
        print(f"Seed {seed:2d}: vertices={n}, undirected_edges={m}, reachable={reachable}, max_dist={max_dist}")

    print("Generated 20 NetworkX R-MAT fixtures successfully.")

if __name__ == "__main__":
    main()
