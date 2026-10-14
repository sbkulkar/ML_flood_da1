# ============================================================
# network.py
#
# CLEAN DRAINAGE NETWORK CONSTRUCTION
#
# Based on the working Colab network logic.
#
# Responsibilities:
#   1. Build drainage endpoints
#   2. Snap endpoints within 15 m
#   3. Build physical drainage edges
#   4. Preserve elevation already attached to the data
#   5. Orient edges downhill
#   6. Diagnose components / terminals / headwaters
#
# IMPORTANT:
#   Elevation is NOT downloaded here.
#   Elevation must already exist in:
#
#       start_elevation
#       end_elevation
#
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class NetworkConfig:

    # Locked snapping tolerance from the working model.
    snap_tolerance_m: float = 15.0

    # Diagnostic search radius.
    candidate_distance_m: float = 50.0


# ============================================================
# UNION FIND
# ============================================================

class UnionFind:

    def __init__(self, n: int):

        self.parent = np.arange(n)

        self.rank = np.zeros(
            n,
            dtype=int
        )

    def find(self, x: int) -> int:

        while self.parent[x] != x:

            self.parent[x] = (
                self.parent[
                    self.parent[x]
                ]
            )

            x = self.parent[x]

        return x

    def union(self, a: int, b: int):

        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        if self.rank[root_a] < self.rank[root_b]:

            self.parent[root_a] = root_b

        elif self.rank[root_a] > self.rank[root_b]:

            self.parent[root_b] = root_a

        else:

            self.parent[root_b] = root_a

            self.rank[root_a] += 1


# ============================================================
# ENDPOINT TABLE
# ============================================================

def build_endpoint_table(
    drains: pd.DataFrame,
) -> pd.DataFrame:

    required = {
        "sx",
        "sy",
        "ex",
        "ey",
    }

    missing = required - set(
        drains.columns
    )

    if missing:

        raise ValueError(
            "Missing endpoint columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    records = []

    for segment_index, row in drains.iterrows():

        start_elevation = np.nan

        end_elevation = np.nan

        if "start_elevation" in drains.columns:

            value = row["start_elevation"]

            if pd.notna(value):

                start_elevation = float(value)

        if "end_elevation" in drains.columns:

            value = row["end_elevation"]

            if pd.notna(value):

                end_elevation = float(value)

        records.append(
            {
                "endpoint_id": len(records),
                "segment_index": segment_index,
                "endpoint_type": "START",
                "x": float(row["sx"]),
                "y": float(row["sy"]),
                "elevation": start_elevation,
            }
        )

        records.append(
            {
                "endpoint_id": len(records),
                "segment_index": segment_index,
                "endpoint_type": "END",
                "x": float(row["ex"]),
                "y": float(row["ey"]),
                "elevation": end_elevation,
            }
        )

    return pd.DataFrame(records)


# ============================================================
# ENDPOINT COMPATIBILITY
# ============================================================

def endpoints_are_compatible(
    endpoints: pd.DataFrame,
    a: int,
    b: int,
) -> bool:

    row_a = endpoints.iloc[a]

    row_b = endpoints.iloc[b]

    # Never connect the two ends of
    # the same drainage segment.

    if (
        row_a["segment_index"]
        ==
        row_b["segment_index"]
    ):

        return False

    return True


# ============================================================
# BUILD SNAPPED NODES
# ============================================================

def build_nodes(
    endpoints: pd.DataFrame,
    config: NetworkConfig,
):

    coordinates = endpoints[
        [
            "x",
            "y",
        ]
    ].to_numpy()

    tree = cKDTree(
        coordinates
    )

    candidate_pairs = tree.query_pairs(
        r=config.snap_tolerance_m
    )

    uf = UnionFind(
        len(endpoints)
    )

    accepted_pairs = []

    for a, b in candidate_pairs:

        if not endpoints_are_compatible(
            endpoints,
            a,
            b,
        ):

            continue

        uf.union(
            a,
            b
        )

        accepted_pairs.append(
            (
                a,
                b,
            )
        )

    roots = np.array(
        [
            uf.find(i)
            for i in range(
                len(endpoints)
            )
        ]
    )

    unique_roots, inverse = np.unique(
        roots,
        return_inverse=True
    )

    node_records = []

    for node_id, root in enumerate(
        unique_roots
    ):

        members = np.where(
            roots == root
        )[0]

        member_rows = endpoints.iloc[
            members
        ]

        elevations = (
            member_rows["elevation"]
            .dropna()
            .to_numpy()
        )

        if len(elevations) > 0:

            elevation = float(
                np.mean(
                    elevations
                )
            )

        else:

            elevation = np.nan

        node_records.append(
            {
                "node_id": node_id,

                "x": float(
                    member_rows["x"].mean()
                ),

                "y": float(
                    member_rows["y"].mean()
                ),

                "elevation": elevation,

                "endpoint_count": len(
                    members
                ),
            }
        )

    nodes = pd.DataFrame(
        node_records
    )

    endpoints = endpoints.copy()

    endpoints["node_id"] = inverse

    return (
        nodes,
        endpoints,
        accepted_pairs,
    )


# ============================================================
# BUILD PHYSICAL EDGES
# ============================================================

def build_edges(
    drains: pd.DataFrame,
    endpoints: pd.DataFrame,
):

    edge_records = []

    for segment_index, row in drains.iterrows():

        start = endpoints[
            (
                endpoints[
                    "segment_index"
                ]
                ==
                segment_index
            )
            &
            (
                endpoints[
                    "endpoint_type"
                ]
                ==
                "START"
            )
        ]

        end = endpoints[
            (
                endpoints[
                    "segment_index"
                ]
                ==
                segment_index
            )
            &
            (
                endpoints[
                    "endpoint_type"
                ]
                ==
                "END"
            )
        ]

        if start.empty or end.empty:

            continue

        start_node = int(
            start.iloc[0]["node_id"]
        )

        end_node = int(
            end.iloc[0]["node_id"]
        )

        if start_node == end_node:

            continue

        # ----------------------------------------------------
        # Physical drainage attributes
        # ----------------------------------------------------

        length = (
            float(row["swmm_length"])
            if "swmm_length" in drains.columns
            and pd.notna(row["swmm_length"])
            else float(row.get("length", 1.0))
        )

        width = (
            float(row["swmm_width"])
            if "swmm_width" in drains.columns
            and pd.notna(row["swmm_width"])
            else 1.0
        )

        depth = (
            float(row["swmm_depth"])
            if "swmm_depth" in drains.columns
            and pd.notna(row["swmm_depth"])
            else 1.0
        )

        roughness = (
            float(row["roughness"])
            if "roughness" in drains.columns
            and pd.notna(row["roughness"])
            else 0.015
        )

        edge_records.append(
            {
                "segment_index":
                    segment_index,

                "from_node":
                    start_node,

                "to_node":
                    end_node,

                "length":
                    length,

                "width":
                    width,

                "depth":
                    depth,

                "roughness":
                    roughness,

                "source":
                    "GCC_API",
            }
        )

    return pd.DataFrame(
        edge_records
    )


# ============================================================
# REMOVE DUPLICATE NODE PAIRS
# ============================================================

def remove_duplicate_edges(
    edges: pd.DataFrame,
):

    if edges.empty:

        return (
            edges.copy(),
            pd.DataFrame()
        )

    edges = edges.copy()

    edges["pair_key"] = edges.apply(
        lambda r: tuple(
            sorted(
                (
                    int(r["from_node"]),
                    int(r["to_node"])
                )
            )
        ),
        axis=1,
    )

    duplicate_mask = (
        edges.duplicated(
            "pair_key",
            keep="first"
        )
    )

    duplicates = edges[
        duplicate_mask
    ].copy()

    unique_edges = edges[
        ~duplicate_mask
    ].copy()

    unique_edges = unique_edges.drop(
        columns=["pair_key"]
    )

    duplicates = duplicates.drop(
        columns=["pair_key"]
    )

    return (
        unique_edges.reset_index(
            drop=True
        ),
        duplicates.reset_index(
            drop=True
        )
    )


# ============================================================
# ORIENT BY ELEVATION
# ============================================================

def orient_edges_downhill(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
):

    if edges.empty:

        return edges.copy()

    elevations = dict(
        zip(
            nodes["node_id"].astype(int),
            nodes["elevation"].astype(float),
        )
    )

    invalid_nodes = []

    for node_id, elevation in elevations.items():

        if not np.isfinite(elevation):

            invalid_nodes.append(
                node_id
            )

    if invalid_nodes:

        raise ValueError(
            "Cannot orient network because "
            f"{len(invalid_nodes)} nodes have "
            "invalid or missing elevation."
        )

    oriented = edges.copy()

    uphill_edges = 0

    for idx in oriented.index:

        a = int(
            oriented.at[
                idx,
                "from_node"
            ]
        )

        b = int(
            oriented.at[
                idx,
                "to_node"
            ]
        )

        za = elevations[a]

        zb = elevations[b]

        if za > zb:

            # Already downhill.
            continue

        elif zb > za:

            # Reverse edge.
            oriented.at[
                idx,
                "from_node"
            ] = b

            oriented.at[
                idx,
                "to_node"
            ] = a

        else:

            # Flat edge.
            # Keep original physical direction.
            pass

    return oriented


# ============================================================
# GRAPH
# ============================================================

def build_graph(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
):

    G = nx.DiGraph()

    for _, row in nodes.iterrows():

        G.add_node(
            int(row["node_id"]),
            elevation=float(
                row["elevation"]
            ),
            x=float(
                row["x"]
            ),
            y=float(
                row["y"]
            ),
        )

    for _, row in edges.iterrows():

        G.add_edge(
            int(row["from_node"]),
            int(row["to_node"]),
            length=float(
                row["length"]
            ),
            width=float(
                row["width"]
            ),
            depth=float(
                row["depth"]
            ),
            roughness=float(
                row["roughness"]
            ),
            segment_index=int(
                row["segment_index"]
            ),
        )

    return G


# ============================================================
# COMPONENT DIAGNOSTICS
# ============================================================

def component_diagnostics(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
):

    G = nx.Graph()

    for node_id in nodes[
        "node_id"
    ]:

        G.add_node(
            int(node_id)
        )

    for _, row in edges.iterrows():

        G.add_edge(
            int(row["from_node"]),
            int(row["to_node"])
        )

    components = []

    for component_id, component in enumerate(
        sorted(
            nx.connected_components(G),
            key=len,
            reverse=True
        ),
        start=1,
    ):

        component_nodes = list(
            component
        )

        component_edges = G.subgraph(
            component_nodes
        ).number_of_edges()

        elevations = nodes[
            nodes["node_id"].isin(
                component_nodes
            )
        ]["elevation"].dropna()

        if len(elevations):

            min_elevation = float(
                elevations.min()
            )

            max_elevation = float(
                elevations.max()
            )

        else:

            min_elevation = np.nan

            max_elevation = np.nan

        components.append(
            {
                "component_id":
                    component_id,

                "nodes":
                    len(component_nodes),

                "edges":
                    component_edges,

                "min_elevation":
                    min_elevation,

                "max_elevation":
                    max_elevation,
            }
        )

    return pd.DataFrame(
        components
    )


# ============================================================
# NETWORK VALIDATION
# ============================================================

def validate_network(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
):

    G = nx.DiGraph()

    for node_id in nodes[
        "node_id"
    ]:

        G.add_node(
            int(node_id)
        )

    for _, row in edges.iterrows():

        G.add_edge(
            int(row["from_node"]),
            int(row["to_node"])
        )

    weak_components = list(
        nx.weakly_connected_components(G)
    )

    strong_components = list(
        nx.strongly_connected_components(G)
    )

    terminals = [
        n
        for n in G.nodes
        if G.out_degree(n) == 0
    ]

    headwaters = [
        n
        for n in G.nodes
        if G.in_degree(n) == 0
    ]

    reachable = set()

    for terminal in terminals:

        reachable.update(
            nx.ancestors(
                G,
                terminal
            )
        )

        reachable.add(
            terminal
        )

    uphill_edges = 0

    elevation = dict(
        zip(
            nodes["node_id"].astype(int),
            nodes["elevation"].astype(float),
        )
    )

    for u, v in G.edges:

        if elevation[v] > elevation[u]:

            uphill_edges += 1

    reachability_pct = (
        100.0
        * len(reachable)
        / len(nodes)
        if len(nodes)
        else 0.0
    )

    return {
        "nodes":
            len(nodes),

        "edges":
            len(edges),

        "weak_components":
            len(weak_components),

        "strong_components":
            len(strong_components),

        "terminals":
            len(terminals),

        "headwaters":
            len(headwaters),

        "nodes_reaching_terminal":
            len(reachable),

        "terminal_reachability_pct":
            reachability_pct,

        "uphill_edges":
            uphill_edges,
    }


# ============================================================
# MAIN CONSTRUCTOR
# ============================================================

def construct_network(
    drains: pd.DataFrame,
    config: NetworkConfig | None = None,
):

    if config is None:

        config = NetworkConfig()

    print()
    print("=" * 70)
    print("NETWORK CONSTRUCTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Check elevation BEFORE doing anything else.
    # --------------------------------------------------------

    required_elevation = {
        "start_elevation",
        "end_elevation",
    }

    missing_elevation_columns = (
        required_elevation
        - set(drains.columns)
    )

    if missing_elevation_columns:

        raise ValueError(
            "Elevation has not been attached to "
            "the drainage dataframe.\n"
            "Missing columns: "
            + ", ".join(
                sorted(
                    missing_elevation_columns
                )
            )
        )

    # --------------------------------------------------------
    # Endpoints
    # --------------------------------------------------------

    endpoints = build_endpoint_table(
        drains
    )

    print(
        f"Endpoints: {len(endpoints)}"
    )

    # --------------------------------------------------------
    # Nodes
    # --------------------------------------------------------

    nodes, endpoints, accepted_pairs = (
        build_nodes(
            endpoints,
            config
        )
    )

    print(
        "Accepted endpoint unions: "
        f"{len(accepted_pairs)}"
    )

    print(
        f"Network nodes: {len(nodes)}"
    )

    # --------------------------------------------------------
    # Physical edges
    # --------------------------------------------------------

    physical_edges = build_edges(
        drains,
        endpoints
    )

    print(
        "Physical drainage edges: "
        f"{len(physical_edges)}"
    )

    # --------------------------------------------------------
    # Duplicate physical connections
    # --------------------------------------------------------

    edges, duplicates = (
        remove_duplicate_edges(
            physical_edges
        )
    )

    if len(duplicates):

        print(
            "Duplicate node-pair groups: "
            f"{len(duplicates)}"
        )

    print(
        "Physical graph: "
        f"{len(nodes)} nodes / "
        f"{len(edges)} edges"
    )

    # --------------------------------------------------------
    # Elevation validation
    # --------------------------------------------------------

    valid_elevations = (
        nodes["elevation"]
        .apply(
            lambda x:
            np.isfinite(x)
        )
        .sum()
    )

    print()
    print(
        "Elevation check:"
    )

    print(
        f"Valid elevations: "
        f"{valid_elevations}/{len(nodes)}"
    )

    if valid_elevations != len(nodes):

        missing = nodes[
            ~nodes["elevation"].apply(
                np.isfinite
            )
        ]["node_id"].tolist()

        raise ValueError(
            "Cannot orient network because "
            f"{len(missing)} nodes have "
            "invalid or missing elevation."
        )

    # --------------------------------------------------------
    # Orient downhill
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "FLOW DIRECTION"
    )

    print(
        "=" * 70
    )

    edges = orient_edges_downhill(
        nodes,
        edges
    )

    # --------------------------------------------------------
    # Graph
    # --------------------------------------------------------

    G = build_graph(
        nodes,
        edges
    )

    # --------------------------------------------------------
    # Components
    # --------------------------------------------------------

    components = component_diagnostics(
        nodes,
        edges
    )

    print(
        f"Connected components: "
        f"{len(components)}"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validation = validate_network(
        nodes,
        edges
    )

    print()
    print(
        "=" * 70
    )

    print(
        "NETWORK VALIDATION"
    )

    print(
        "=" * 70
    )

    for key, value in validation.items():

        print(
            f"{key}: {value}"
        )

    return {
        "nodes": nodes.reset_index(
            drop=True
        ),

        "edges": edges.reset_index(
            drop=True
        ),

        "endpoints": endpoints.reset_index(
            drop=True
        ),

        "components": components.reset_index(
            drop=True
        ),

        "duplicates": duplicates.reset_index(
            drop=True
        ),

        "validation": validation,

        "graph": G,
    }