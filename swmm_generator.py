# ============================================================
# swmm_generator.py
#
# VALIDATED DRAINAGE NETWORK -> SWMM
#
# Input:
#   data/network/<REGION>/nodes.csv
#   data/network/<REGION>/edges.csv
#
# Output:
#   data/swmm/<REGION>/<REGION>_flood_model.inp
#
# This file converts the already validated drainage network
# into a SWMM input file.
#
# IMPORTANT:
#   - Coordinates are read from x,y in nodes.csv
#   - Coordinates are projected EPSG:32644
#   - Natural terminal is converted into SWMM OUTFALL_01
#   - Rainfall and prototype subcatchments are included
# ============================================================


from pathlib import Path
import math
import sys

import pandas as pd

from config import (
    REGION,
    STUDY_AREA_NAME,
)


# ============================================================
# REGION
# ============================================================

# Usage:
#
#   python swmm_generator.py mumbai_hindmata
#
# If no region is supplied, REGION from config.py is used.

if len(sys.argv) > 1:
    ACTIVE_REGION = sys.argv[1]
else:
    ACTIVE_REGION = REGION


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

NETWORK_DIR = (
    PROJECT_DIR
    / "data"
    / "network"
    / ACTIVE_REGION
)

SWMM_DIR = (
    PROJECT_DIR
    / "data"
    / "swmm"
    / ACTIVE_REGION
)

NODES_FILE = (
    NETWORK_DIR
    / "nodes.csv"
)

EDGES_FILE = (
    NETWORK_DIR
    / "edges.csv"
)

OUTPUT_FILE = (
    SWMM_DIR
    / f"{ACTIVE_REGION}_flood_model.inp"
)


# ============================================================
# DEFAULT HYDRAULIC PARAMETERS
# ============================================================

DEFAULT_WIDTH = 1.0
DEFAULT_DEPTH = 1.0
DEFAULT_ROUGHNESS = 0.015

DEFAULT_MAX_DEPTH = 3.0
DEFAULT_INITIAL_DEPTH = 0.0
DEFAULT_SURCHARGE_DEPTH = 1.0

DEFAULT_OUTFALL_STAGE = 0.0

# ============================================================
# RAINFALL / RUNOFF DEFAULTS
# ============================================================

RAINGAGE_NAME = "RG_VELLORE"

# Initial historical-event test rainfall.
# This is NOT labelled as a return-period storm.
RAINFALL_TIMESERIES_NAME = "Vellore_Historical_Test"

RAINFALL_FORMAT = "INTENSITY"
RAINFALL_INTERVAL = "00:15"

# Placeholder only until the justified rainfall time series
# is prepared from the selected rainfall source.
DEFAULT_RAINFALL_INTENSITY = 0.0

DEFAULT_SUBCATCHMENT_AREA = 1.0
DEFAULT_IMPERVIOUSNESS = 50.0
DEFAULT_MANNING_N_PERV = 0.15
DEFAULT_MANNING_N_IMPERV = 0.015
DEFAULT_DEPRESSION_PERV = 0.05
DEFAULT_DEPRESSION_IMPERV = 0.02
DEFAULT_INFILTRATION_RATE = 3.0


# ============================================================
# SWMM OUTFALL NAME
# ============================================================

OUTFALL_NAME = "OUTFALL_01"


# ============================================================
# HELPERS
# ============================================================

def build_raingage():
    """
    Build the SWMM rain gage definition.
    """
    return [
        "[RAINGAGES]",
        ";;Name           Format       Interval       SCF      Source",
        f"{RAINGAGE_NAME:<15} {RAINFALL_FORMAT:<12} {RAINFALL_INTERVAL:<14} 1.0      TIMESERIES {RAINFALL_TIMESERIES_NAME}",
        "",
    ]

def build_timeseries():
    """
    Build a clearly labelled synthetic test rainfall time series.
    Values are rainfall intensity in mm/hour at 15-minute intervals.
    """
    return [
        "[TIMESERIES]",
        ";;Name                 Date       Time       Value",
        f"{RAINFALL_TIMESERIES_NAME:<22} 01/01/2025 00:00      0.0",
        f"{RAINFALL_TIMESERIES_NAME:<22} 01/01/2025 00:15      5.0",
        f"{RAINFALL_TIMESERIES_NAME:<22} 01/01/2025 00:30     10.0",
        f"{RAINFALL_TIMESERIES_NAME:<22} 01/01/2025 00:45     20.0",
        f"{RAINFALL_TIMESERIES_NAME:<22} 01/01/2025 01:00     30.0",
        f"{RAINFALL_TIMESERIES_NAME:<22} 01/01/2025 01:15     40.0",
        f"{RAINFALL_TIMESERIES_NAME:<22} 01/01/2025 01:30     30.0",
        f"{RAINFALL_TIMESERIES_NAME:<22} 01/01/2025 01:45     20.0",
        f"{RAINFALL_TIMESERIES_NAME:<22} 01/01/2025 02:00     10.0",
        f"{RAINFALL_TIMESERIES_NAME:<22} 01/01/2025 02:15      5.0",
        f"{RAINFALL_TIMESERIES_NAME:<22} 01/01/2025 02:30      0.0",
        "",
    ]




def clean_number(value, default=0.0):
    """
    Safely convert a value to a finite float.
    """

    try:
        value = float(value)

        if math.isfinite(value):
            return value

    except (
        TypeError,
        ValueError,
    ):
        pass

    return default


# ============================================================
# SWMM NODE NAME
# ============================================================

def swmm_node_name(node_id):
    """
    Convert numeric node ID to SWMM node name.

    Example:
        0   -> J_0000
        431 -> J_0431
    """

    return f"J_{int(node_id):04d}"


# ============================================================
# SWMM CONDUIT NAME
# ============================================================

def swmm_conduit_name(index):
    """
    Convert edge index to SWMM conduit name.
    """

    return f"C_{int(index):04d}"


# ============================================================
# LOAD NETWORK
# ============================================================

def load_network():

    print()
    print("=" * 70)
    print("LOADING VALIDATED NETWORK")
    print("=" * 70)

    print()
    print(f"Region: {ACTIVE_REGION}")

    print()
    print("Nodes file:")
    print(NODES_FILE)

    print()
    print("Edges file:")
    print(EDGES_FILE)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not NODES_FILE.exists():

        raise FileNotFoundError(
            f"\nNodes file not found:\n"
            f"{NODES_FILE}\n\n"
            f"Run pipeline.py {ACTIVE_REGION} first."
        )

    if not EDGES_FILE.exists():

        raise FileNotFoundError(
            f"\nEdges file not found:\n"
            f"{EDGES_FILE}\n\n"
            f"Run pipeline.py {ACTIVE_REGION} first."
        )

    # --------------------------------------------------------
    # Read CSV files
    # --------------------------------------------------------

    nodes = pd.read_csv(
        NODES_FILE
    )

    edges = pd.read_csv(
        EDGES_FILE
    )

    print()
    print(f"Nodes loaded: {len(nodes)}")
    print(f"Edges loaded: {len(edges)}")

    # --------------------------------------------------------
    # Required node columns
    # --------------------------------------------------------

    required_node_columns = {
        "node_id",
        "x",
        "y",
        "elevation",
    }

    missing_nodes = (
        required_node_columns
        - set(nodes.columns)
    )

    if missing_nodes:

        raise ValueError(
            "Missing required node columns: "
            + ", ".join(
                sorted(missing_nodes)
            )
        )

    # --------------------------------------------------------
    # Required edge columns
    # --------------------------------------------------------

    required_edge_columns = {
        "from_node",
        "to_node",
        "length",
    }

    missing_edges = (
        required_edge_columns
        - set(edges.columns)
    )

    if missing_edges:

        raise ValueError(
            "Missing required edge columns: "
            + ", ".join(
                sorted(missing_edges)
            )
        )

    return nodes, edges


# ============================================================
# VALIDATE NETWORK
# ============================================================

def validate_network(
    nodes,
    edges,
):

    print()
    print("=" * 70)
    print("SWMM NETWORK VALIDATION")
    print("=" * 70)

    # ========================================================
    # NODE IDs
    # ========================================================

    node_ids = set(
        pd.to_numeric(
            nodes["node_id"],
            errors="coerce"
        )
        .dropna()
        .astype(int)
        .tolist()
    )

    print()
    print(
        f"Node IDs found: {len(node_ids)}"
    )

    # ========================================================
    # NODE IDs VALIDITY
    # ========================================================

    invalid_node_ids = (
        pd.to_numeric(
            nodes["node_id"],
            errors="coerce"
        ).isna()
    )

    if invalid_node_ids.any():

        raise ValueError(
            "Some nodes have invalid node_id values."
        )

    # ========================================================
    # ELEVATIONS
    # ========================================================

    elevations = pd.to_numeric(
        nodes["elevation"],
        errors="coerce",
    )

    missing_elevations = int(
        elevations.isna().sum()
    )

    print(
        f"Missing node elevations: "
        f"{missing_elevations}"
    )

    if missing_elevations > 0:

        raise ValueError(
            "SWMM cannot be generated because "
            "some nodes have missing elevations."
        )

    # ========================================================
    # X COORDINATES
    # ========================================================

    x_values = pd.to_numeric(
        nodes["x"],
        errors="coerce",
    )

    invalid_x = (
        x_values.isna()
        | ~x_values.map(math.isfinite)
    )

    invalid_x_count = int(
        invalid_x.sum()
    )

    print(
        f"Invalid X coordinates: "
        f"{invalid_x_count}"
    )

    if invalid_x_count > 0:

        raise ValueError(
            "Some nodes have invalid x coordinates."
        )

    # ========================================================
    # Y COORDINATES
    # ========================================================

    y_values = pd.to_numeric(
        nodes["y"],
        errors="coerce",
    )

    invalid_y = (
        y_values.isna()
        | ~y_values.map(math.isfinite)
    )

    invalid_y_count = int(
        invalid_y.sum()
    )

    print(
        f"Invalid Y coordinates: "
        f"{invalid_y_count}"
    )

    if invalid_y_count > 0:

        raise ValueError(
            "Some nodes have invalid y coordinates."
        )

    # ========================================================
    # EDGE ENDPOINTS
    # ========================================================

    invalid_from = []
    invalid_to = []

    for index, edge in edges.iterrows():

        try:
            from_node = int(
                edge["from_node"]
            )

            to_node = int(
                edge["to_node"]
            )

        except (
            TypeError,
            ValueError,
        ):

            raise ValueError(
                f"Invalid node reference in edge "
                f"at row {index}."
            )

        if from_node not in node_ids:

            invalid_from.append(
                index
            )

        if to_node not in node_ids:

            invalid_to.append(
                index
            )

    if invalid_from:

        raise ValueError(
            f"{len(invalid_from)} edges reference "
            "nonexistent from_node IDs."
        )

    if invalid_to:

        raise ValueError(
            f"{len(invalid_to)} edges reference "
            "nonexistent to_node IDs."
        )

    # ========================================================
    # EDGE LENGTH
    # ========================================================

    lengths = pd.to_numeric(
        edges["length"],
        errors="coerce",
    )

    invalid_lengths = (
        (~lengths.notna())
        | (lengths <= 0)
    )

    invalid_length_count = int(
        invalid_lengths.sum()
    )

    print(
        f"Invalid edge lengths: "
        f"{invalid_length_count}"
    )

    if invalid_length_count > 0:

        raise ValueError(
            "SWMM cannot be generated because "
            "some conduit lengths are invalid."
        )

    # ========================================================
    # VALIDATION COMPLETE
    # ========================================================

    print()
    print(
        "Network is valid for SWMM generation."
    )


# ============================================================
# FIND NATURAL TERMINAL NODES
# ============================================================

def find_terminal_nodes(
    nodes,
    edges,
):

    # Nodes appearing as upstream/from nodes
    from_nodes = set(
        edges["from_node"]
        .astype(int)
        .tolist()
    )

    # Nodes appearing as downstream/to nodes
    to_nodes = set(
        edges["to_node"]
        .astype(int)
        .tolist()
    )

    # A natural terminal is a node that appears
    # downstream but never appears upstream.
    terminals = sorted(
        to_nodes - from_nodes
    )

    if not terminals:

        raise ValueError(
            "No natural terminal node was found."
        )

    return terminals


# ============================================================
# FIND LOWEST TERMINAL
# ============================================================

def find_lowest_terminal(
    nodes,
    terminals,
):

    terminal_elevations = []

    for node_id in terminals:

        matches = nodes.loc[
            nodes["node_id"].astype(int)
            == int(node_id)
        ]

        if len(matches) == 0:
            continue

        elevation = clean_number(
            matches.iloc[0]["elevation"]
        )

        terminal_elevations.append(
            (
                int(node_id),
                elevation,
            )
        )

    if not terminal_elevations:

        raise ValueError(
            "Could not determine terminal elevation."
        )

    # Lowest terminal becomes the outfall
    terminal, elevation = min(
        terminal_elevations,
        key=lambda item: item[1]
    )

    return terminal, elevation



# ============================================================
# BUILD SUBCATCHMENTS
# ============================================================

def build_subcatchments(
    nodes,
    terminal_node,
):

    lines = []

    lines.append("[SUBCATCHMENTS]")
    lines.append(
        ";;Name           Rain Gage       Outlet         Area     "
        "%Imperv   Width    %Slope   CurbLen   SnowPack"
    )
    lines.append(
        ";;-------------- --------------- -------------- -------- "
        "-------- -------- -------- --------- --------"
    )

    subcatchment_number = 1

    for _, node in nodes.iterrows():

        node_id = int(node["node_id"])

        # The natural terminal is represented as an outfall,
        # so it does not receive a subcatchment directly.
        if node_id == int(terminal_node):
            continue

        node_name = swmm_node_name(node_id)

        subcatchment_name = f"S_{subcatchment_number:04d}"

        lines.append(
            f"{subcatchment_name:<16}"
            f"{RAINGAGE_NAME:<16}"
            f"{node_name:<15}"
            f"{DEFAULT_SUBCATCHMENT_AREA:>8.3f}"
            f"{DEFAULT_IMPERVIOUSNESS:>10.2f}"
            f"{100.0:>9.2f}"
            f"{1.0:>9.2f}"
            f"{0.0:>10.2f}"
            f"{'':>10}"
        )

        subcatchment_number += 1

    lines.append("")

    return lines


# ============================================================
# BUILD SUBAREAS
# ============================================================

def build_subareas():

    lines = []

    lines.append("[SUBAREAS]")
    lines.append(
        ";;Subcatchment   N-Imperv   N-Perv    S-Imperv   "
        "S-Perv    PctZero   RouteTo    PctRouted"
    )
    lines.append(
        ";;-------------- ---------- --------- ---------- "
        "--------- --------- --------- ---------"
    )

    # One common prototype parameter set is used initially.
    # These values will later be replaced/tuned using Vellore
    # land-use and hydrologic information.
    for i in range(1, 46):
        subcatchment_name = f"S_{i:04d}"

        lines.append(
            f"{subcatchment_name:<16}"
            f"{DEFAULT_MANNING_N_IMPERV:>11.3f}"
            f"{DEFAULT_MANNING_N_PERV:>10.3f}"
            f"{DEFAULT_DEPRESSION_IMPERV:>11.3f}"
            f"{DEFAULT_DEPRESSION_PERV:>10.3f}"
            f"{0.0:>10.2f}"
            f"{'OUTLET':>10}"
            f"{100.0:>10.2f}"
        )

    lines.append("")

    return lines


# ============================================================
# BUILD JUNCTIONS
# ============================================================

def build_junctions(
    nodes,
    terminal_node,
):

    lines = []

    lines.append(
        "[JUNCTIONS]"
    )

    lines.append(
        ";;Name           Elevation   MaxDepth   "
        "InitDepth   SurDepth   Aponded"
    )

    lines.append(
        ";;-------------- ----------- ---------- "
        "----------- ---------- ----------"
    )

    for _, node in nodes.iterrows():

        node_id = int(
            node["node_id"]
        )

        # ----------------------------------------------------
        # Do NOT put the natural terminal into JUNCTIONS.
        # It will become OUTFALL_01.
        # ----------------------------------------------------

        if node_id == int(
            terminal_node
        ):
            continue

        name = swmm_node_name(
            node_id
        )

        elevation = clean_number(
            node["elevation"]
        )

        lines.append(
            f"{name:<16}"
            f"{elevation:>11.3f}"
            f"{DEFAULT_MAX_DEPTH:>11.3f}"
            f"{DEFAULT_INITIAL_DEPTH:>12.3f}"
            f"{DEFAULT_SURCHARGE_DEPTH:>11.3f}"
            f"{0.0:>11.3f}"
        )

    lines.append("")

    return lines


# ============================================================
# BUILD OUTFALL
# ============================================================

def build_outfall(
    nodes,
    edges,
):

    terminals = find_terminal_nodes(
        nodes,
        edges,
    )

    terminal, terminal_elevation = (
        find_lowest_terminal(
            nodes,
            terminals,
        )
    )

    print()
    print(
        "Natural terminal nodes:"
    )

    for node_id in terminals:

        matches = nodes.loc[
            nodes["node_id"].astype(int)
            == int(node_id)
        ]

        if len(matches) == 0:
            continue

        elevation = clean_number(
            matches.iloc[0]["elevation"]
        )

        print(
            f"  {swmm_node_name(node_id)}"
            f" -> {elevation:.3f} m"
        )

    print()

    print(
        f"Selected natural terminal: "
        f"{swmm_node_name(terminal)}"
    )

    print(
        f"Selected SWMM outfall: "
        f"{OUTFALL_NAME}"
    )

    print(
        f"Outfall elevation: "
        f"{terminal_elevation:.3f} m"
    )

    # --------------------------------------------------------
    # Create SWMM outfall
    # --------------------------------------------------------

    lines = []

    lines.append(
        "[OUTFALLS]"
    )

    lines.append(
        ";;Name           Elevation   Type       "
        "Stage Data   Gated"
    )

    lines.append(
        ";;-------------- ----------- ---------- "
        "------------ ------"
    )

    lines.append(
        f"{OUTFALL_NAME:<16}"
        f"{terminal_elevation:>11.3f} "
        f"{'FREE':<10}"
        f"{'':<13}"
        f"{'NO':>6}"
    )

    lines.append("")

    return (
        lines,
        terminal,
        terminal_elevation,
    )


# ============================================================
# BUILD CONDUITS
# ============================================================

def build_conduits(
    edges,
    terminal_node,
):

    lines = []

    lines.append(
        "[CONDUITS]"
    )

    lines.append(
        ";;Name           From Node       To Node         "
        "Length       Roughness   InOffset   OutOffset  "
        "InitFlow   MaxFlow"
    )

    lines.append(
        ";;-------------- --------------- --------------- "
        "------------ ------------ ---------- ---------- "
        "---------- ----------"
    )

    for index, edge in edges.iterrows():

        conduit_name = swmm_conduit_name(
            index
        )

        # ----------------------------------------------------
        # FROM NODE
        # ----------------------------------------------------

        from_node_id = int(
            edge["from_node"]
        )

        from_node = swmm_node_name(
            from_node_id
        )

        # ----------------------------------------------------
        # TO NODE
        # ----------------------------------------------------

        to_node_id = int(
            edge["to_node"]
        )

        # If this conduit terminates at the natural terminal,
        # connect it directly to OUTFALL_01.
        if to_node_id == int(
            terminal_node
        ):

            to_node = OUTFALL_NAME

        else:

            to_node = swmm_node_name(
                to_node_id
            )

        # ----------------------------------------------------
        # LENGTH
        # ----------------------------------------------------

        length = clean_number(
            edge["length"]
        )

        # ----------------------------------------------------
        # ROUGHNESS
        # ----------------------------------------------------

        roughness = clean_number(
            edge.get(
                "roughness",
                DEFAULT_ROUGHNESS,
            ),
            DEFAULT_ROUGHNESS,
        )

        # ----------------------------------------------------
        # CONDUIT
        # ----------------------------------------------------

        lines.append(
            f"{conduit_name:<16}"
            f"{from_node:<16}"
            f"{to_node:<16}"
            f"{length:>12.3f}"
            f"{roughness:>13.5f}"
            f"{0.0:>11.3f}"
            f"{0.0:>11.3f}"
            f"{0.0:>11.3f}"
            f"{0.0:>11.3f}"
        )

    lines.append("")

    return lines


# ============================================================
# BUILD XSECTIONS
# ============================================================

def build_xsections(edges):

    lines = []

    lines.append(
        "[XSECTIONS]"
    )

    lines.append(
        ";;Link           Shape        Geom1       "
        "Geom2       Geom3       Geom4       Barrels"
    )

    lines.append(
        ";;-------------- ------------ ----------- "
        "----------- ----------- ----------- --------"
    )

    for index, edge in edges.iterrows():

        conduit_name = swmm_conduit_name(
            index
        )

        # ----------------------------------------------------
        # Width
        # ----------------------------------------------------

        width = clean_number(
            edge.get(
                "width",
                DEFAULT_WIDTH,
            ),
            DEFAULT_WIDTH,
        )

        # ----------------------------------------------------
        # Depth
        # ----------------------------------------------------

        depth = clean_number(
            edge.get(
                "depth",
                DEFAULT_DEPTH,
            ),
            DEFAULT_DEPTH,
        )

        # ----------------------------------------------------
        # Prevent invalid zero dimensions
        # ----------------------------------------------------

        if width <= 0:
            width = DEFAULT_WIDTH

        if depth <= 0:
            depth = DEFAULT_DEPTH

        # ----------------------------------------------------
        # Rectangular open channel
        # ----------------------------------------------------

        lines.append(
            f"{conduit_name:<16}"
            f"{'RECT_OPEN':<13}"
            f"{width:>11.3f}"
            f"{depth:>12.3f}"
            f"{0.0:>12.3f}"
            f"{0.0:>12.3f}"
            f"{1:>9d}"
        )

    lines.append("")

    return lines


# ============================================================
# BUILD MAP DIMENSIONS
# ============================================================

def build_map_dimensions(
    nodes,
):

    x_values = pd.to_numeric(
        nodes["x"],
        errors="coerce",
    )

    y_values = pd.to_numeric(
        nodes["y"],
        errors="coerce",
    )

    min_x = float(
        x_values.min()
    )

    max_x = float(
        x_values.max()
    )

    min_y = float(
        y_values.min()
    )

    max_y = float(
        y_values.max()
    )

    # --------------------------------------------------------
    # Add small margin around network
    # --------------------------------------------------------

    x_margin = max(
        (max_x - min_x) * 0.02,
        10.0,
    )

    y_margin = max(
        (max_y - min_y) * 0.02,
        10.0,
    )

    min_x -= x_margin
    max_x += x_margin

    min_y -= y_margin
    max_y += y_margin

    return (
        f"DIMENSIONS "
        f"{min_x:.3f} "
        f"{min_y:.3f} "
        f"{max_x:.3f} "
        f"{max_y:.3f}"
    )


# ============================================================
# BUILD COORDINATES
# ============================================================

def build_coordinates(
    nodes,
):

    lines = []

    lines.append(
        "[COORDINATES]"
    )

    lines.append(
        ";;Node           X-Coord        Y-Coord"
    )

    lines.append(
        ";;-------------- -------------- --------------"
    )

    for _, node in nodes.iterrows():

        node_id = int(
            node["node_id"]
        )

        node_name = swmm_node_name(
            node_id
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Your nodes.csv contains:
        #
        # x = EPSG:32644 X coordinate
        # y = EPSG:32644 Y coordinate
        #
        # DO NOT use longitude/latitude here.
        # ----------------------------------------------------

        x = clean_number(
            node["x"]
        )

        y = clean_number(
            node["y"]
        )

        # ----------------------------------------------------
        # Natural terminal is represented by OUTFALL_01
        # ----------------------------------------------------

        if node_id == CURRENT_TERMINAL_NODE:

            node_name = OUTFALL_NAME

        lines.append(
            f"{node_name:<16}"
            f"{x:>14.3f}"
            f"{y:>14.3f}"
        )

    lines.append("")

    return lines


# ============================================================
# BUILD LOSSES
# ============================================================

def build_losses():

    return [
        "[LOSSES]",
        ";;Link           Inlet      Outlet     Average    Flap Gate",
        ";;-------------- ---------- ---------- ---------- ----------",
        "",
    ]


# ============================================================
# BUILD SWMM FILE
# ============================================================

def build_swmm_file(
    nodes,
    edges,
):

    global CURRENT_TERMINAL_NODE

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    SWMM_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Determine natural terminal
    # --------------------------------------------------------

    terminals = find_terminal_nodes(
        nodes,
        edges,
    )

    terminal_node, terminal_elevation = (
        find_lowest_terminal(
            nodes,
            terminals,
        )
    )

    CURRENT_TERMINAL_NODE = int(
        terminal_node
    )

    # --------------------------------------------------------
    # Build outfall
    # --------------------------------------------------------

    (
        outfall_lines,
        terminal_node,
        terminal_elevation,
    ) = build_outfall(
        nodes,
        edges,
    )

    # --------------------------------------------------------
    # Start SWMM file
    # --------------------------------------------------------

    lines = []

    # ========================================================
    # TITLE
    # ========================================================

    lines.extend(
        [
            "[TITLE]",
            ";;",
            f";; {STUDY_AREA_NAME}",
            f";; Region: {ACTIVE_REGION}",
            ";; Drainage network generated from validated GIS/OSM data",
            ";; SRTM elevation attached to drainage nodes",
            ";; Coordinate system: EPSG:32644",
            ";;",
            "",
        ]
    )

    # ========================================================
    # OPTIONS
    # ========================================================

    lines.extend(
        [
            "[OPTIONS]",
            ";;Option             Value",
            "FLOW_UNITS           CMS",
            "INFILTRATION         HORTON",
            "FLOW_ROUTING         DYNWAVE",
            "LINK_OFFSETS         DEPTH",
            "MIN_SLOPE             0",  
            "ALLOW_PONDING         YES",
            "SKIP_STEADY_STATE     NO",
            "",
            "START_DATE           01/01/2025",
            "START_TIME           00:00:00",
            "END_DATE             01/02/2025",
            "END_TIME             00:00:00",
            "",
        ]
    )

    # ========================================================
    # RAIN GAGE
    # ========================================================

    lines.extend(
        build_raingage()
    )


    # ========================================================
    # JUNCTIONS
    # ========================================================

    lines.extend(
        build_junctions(
            nodes,
            terminal_node,
        )
    )
    # ========================================================
    # SUBCATCHMENTS
    # ========================================================

    lines.extend(
        build_subcatchments(
            nodes,
            terminal_node,
        )
    )

    # ========================================================
    # SUBAREAS
    # ========================================================

    lines.extend(
        build_subareas()
    )

    # ========================================================
    # OUTFALL
    # ========================================================

    lines.extend(
        outfall_lines
    )

    # ========================================================
    # CONDUITS
    # ========================================================

    lines.extend(
        build_conduits(
            edges,
            terminal_node,
        )
    )

    # ========================================================
    # XSECTIONS
    # ========================================================

    lines.extend(
        build_xsections(
            edges
        )
    )

    # ========================================================
    # LOSSES
    # ========================================================

    lines.extend(
        build_losses()
    )
    
    # ========================================================
    # RAINFALL TIME SERIES
    # ========================================================

    lines.extend(
        build_timeseries()
    )

    # ========================================================
    # REPORT
    # ========================================================

    lines.extend(
        [
            "[REPORT]",
            "INPUT      NO",
            "CONTROLS   NO",
            "SUBCATCHMENTS ALL",
            "NODES ALL",
            "LINKS ALL",
            "",
        ]
    )

    # ========================================================
    # MAP
    # ========================================================

    lines.extend(
        [
            "[MAP]",
            build_map_dimensions(
                nodes
            ),
            "",
        ]
    )

    # ========================================================
    # COORDINATES
    # ========================================================

    lines.extend(
        build_coordinates(
            nodes
        )
    )

    # ========================================================
    # SYMBOLS
    # ========================================================

    lines.extend(
        [
            "[SYMBOLS]",
            ";;Gage           X-Coord      Y-Coord",
            ";;-------------- ------------ ------------",
            "",
        ]
    )

    # ========================================================
    # TAGS
    # ========================================================

    lines.extend(
        [
            "[TAGS]",
            ";;Category       Name",
            "",
        ]
    )

    # ========================================================
    # WRITE FILE
    # ========================================================

    OUTPUT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    # ========================================================
    # OUTPUT INFORMATION
    # ========================================================

    print()
    print("=" * 70)
    print("SWMM INPUT CREATED")
    print("=" * 70)

    print()
    print("File:")
    print(OUTPUT_FILE)

    print()
    print(f"Nodes: {len(nodes)}")
    print(f"Conduits: {len(edges)}")

    print(
        f"Natural terminal: "
        f"{swmm_node_name(terminal_node)}"
    )

    print(
        f"SWMM outfall: "
        f"{OUTFALL_NAME}"
    )

    print(
        f"Outfall elevation: "
        f"{terminal_elevation:.3f} m"
    )

    print()
    print(
        "Coordinate system: EPSG:32644"
    )

    print(
        "Coordinates loaded from nodes.csv: x,y"
    )

    return OUTPUT_FILE


# ============================================================
# VALIDATE SWMM FILE STRUCTURE
# ============================================================

def validate_swmm_file(
    filepath,
):

    print()
    print("=" * 70)
    print("VALIDATING GENERATED SWMM FILE")
    print("=" * 70)

    text = Path(
        filepath
    ).read_text(
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Required sections
    # --------------------------------------------------------

    required_sections = [
        "[TITLE]",
        "[OPTIONS]",
        "[RAINGAGES]",
        "[TIMESERIES]",
        "[SUBCATCHMENTS]",
        "[SUBAREAS]",
          "[JUNCTIONS]",
        "[OUTFALLS]",
        "[CONDUITS]",
        "[XSECTIONS]",
        "[LOSSES]",
        "[REPORT]",
        "[MAP]",
        "[COORDINATES]",
    ]

    missing = [
        section
        for section in required_sections
        if section not in text
    ]

    if missing:

        raise ValueError(
            "Generated SWMM file is missing sections: "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # Make sure OUTFALL_01 exists
    # --------------------------------------------------------

    if OUTFALL_NAME not in text:

        raise ValueError(
            "OUTFALL_01 was not found in the generated SWMM file."
        )

    # --------------------------------------------------------
    # Check coordinates are not all zero
    # --------------------------------------------------------

    coordinate_section = text.split(
        "[COORDINATES]",
        1
    )[1]

    coordinate_section = coordinate_section.split(
        "[SYMBOLS]",
        1
    )[0]

    coordinate_lines = []

    for line in coordinate_section.splitlines():

        stripped = line.strip()

        if not stripped:
            continue

        if stripped.startswith(";"):
            continue

        parts = stripped.split()

        if len(parts) >= 3:

            try:

                x = float(parts[1])
                y = float(parts[2])

                coordinate_lines.append(
                    (x, y)
                )

            except ValueError:
                continue

    if not coordinate_lines:

        raise ValueError(
            "No valid SWMM coordinates were generated."
        )

    non_zero_coordinates = [
        (x, y)
        for x, y in coordinate_lines
        if abs(x) > 0.001
        or abs(y) > 0.001
    ]

    if not non_zero_coordinates:

        raise ValueError(
            "All SWMM coordinates are zero. "
            "Check nodes.csv x,y values."
        )

    # --------------------------------------------------------
    # Check outfall connectivity
    # --------------------------------------------------------

    conduit_section = text.split(
        "[CONDUITS]",
        1
    )[1]

    conduit_section = conduit_section.split(
        "[XSECTIONS]",
        1
    )[0]

    outfall_connections = 0

    for line in conduit_section.splitlines():

        stripped = line.strip()

        if not stripped:
            continue

        if stripped.startswith(";"):
            continue

        parts = stripped.split()

        if len(parts) >= 3:

            from_node = parts[1]
            to_node = parts[2]

            if (
                from_node == OUTFALL_NAME
                or to_node == OUTFALL_NAME
            ):

                outfall_connections += 1

    if outfall_connections == 0:

        raise ValueError(
            "OUTFALL_01 exists but is not connected "
            "to any conduit."
        )

    # --------------------------------------------------------
    # Validation passed
    # --------------------------------------------------------

    print()
    print(
        "SWMM file structure validation: PASSED"
    )

    print(
        f"Valid coordinate points: "
        f"{len(coordinate_lines)}"
    )

    print(
        f"Non-zero coordinate points: "
        f"{len(non_zero_coordinates)}"
    )

    print(
        f"Outfall-connected conduits: "
        f"{outfall_connections}"
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("#" * 70)
    print("# DRAINAGE NETWORK -> SWMM")
    print("#" * 70)

    print()
    print(
        f"Active region: "
        f"{ACTIVE_REGION}"
    )

    print(
        f"Study area: "
        f"{STUDY_AREA_NAME}"
    )

    print()

    # ========================================================
    # 1. LOAD NETWORK
    # ========================================================

    nodes, edges = load_network()

    # ========================================================
    # 2. VALIDATE NETWORK
    # ========================================================

    validate_network(
        nodes,
        edges,
    )

    # ========================================================
    # 3. BUILD SWMM
    # ========================================================

    filepath = build_swmm_file(
        nodes,
        edges,
    )

    # ========================================================
    # 4. VALIDATE SWMM
    # ========================================================

    validate_swmm_file(
        filepath
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("#" * 70)
    print("# SWMM FILE READY")
    print("#" * 70)

    print()
    print(
        f"Run this file in SWMM:\n"
        f"{filepath}"
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()