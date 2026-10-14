# ============================================================
# swmm_mumbai.py
# ============================================================
# MUMBAI HINDMATA BASIN -> SWMM
#
# This file is Mumbai-specific.
#
# It:
#   1. Loads the validated Mumbai drainage network
#   2. Creates SWMM junctions and conduits
#   3. Creates a Mumbai rainfall time series
#   4. Creates a rain gauge
#   5. Creates subcatchments
#   6. Adds Horton infiltration
#   7. Connects the natural terminal to OUTFALL_01
#   8. Uses the actual x,y coordinates from nodes.csv
#
# IMPORTANT:
# Rainfall parameters below are a TEST STORM.
# They are not calibrated Mumbai rainfall observations.
# ============================================================


from pathlib import Path
import math

import pandas as pd

from config import STUDY_AREA_NAME


# ============================================================
# MUMBAI CONFIGURATION
# ============================================================

REGION = "mumbai_hindmata"

PROJECT_DIR = Path(__file__).resolve().parent

NETWORK_DIR = (
    PROJECT_DIR
    / "data"
    / "network"
    / REGION
)

SWMM_DIR = (
    PROJECT_DIR
    / "data"
    / "swmm"
    / REGION
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
    / f"{REGION}_flood_model.inp"
)


# ============================================================
# SWMM SETTINGS
# ============================================================

OUTFALL_NAME = "OUTFALL_01"

RAIN_GAGE_NAME = "RG_MUMBAI"

RAIN_TIMESERIES_NAME = "MUMBAI_RAIN"


# ============================================================
# HYDRAULIC DEFAULTS
# ============================================================

DEFAULT_WIDTH = 1.0
DEFAULT_DEPTH = 1.0
DEFAULT_ROUGHNESS = 0.015

DEFAULT_MAX_DEPTH = 3.0
DEFAULT_INITIAL_DEPTH = 0.0
DEFAULT_SURCHARGE_DEPTH = 1.0


# ============================================================
# HYDROLOGY SETTINGS
# ============================================================

# Temporary test assumptions.
# These should be calibrated later using real Mumbai data.

PERCENT_IMPERVIOUS = 70.0

MANNING_N_IMPERVIOUS = 0.015
MANNING_N_PERVIOUS = 0.30

DEPRESSION_STORAGE_IMPERVIOUS = 1.5
DEPRESSION_STORAGE_PERVIOUS = 3.0

PERCENT_ZERO_IMPERVIOUS = 25.0

# Horton infiltration parameters
HORTON_MAX_RATE = 75.0
HORTON_MIN_RATE = 5.0
HORTON_DECAY = 4.0
HORTON_DRY_TIME = 7.0
HORTON_MAX_INFIL = 0.0


# ============================================================
# SYNTHETIC MUMBAI TEST STORM
# ============================================================
#
# Rainfall intensity = mm/hour
#
# 15-minute interval
#
# This represents a heavy urban rainfall event for testing
# whether the SWMM model actually produces runoff and flooding.
#
# It is NOT an official observed Mumbai rainfall event.
# ============================================================

RAIN_SERIES = [
    ("00:00", 0.0),
    ("00:15", 5.0),
    ("00:30", 10.0),
    ("00:45", 20.0),
    ("01:00", 40.0),
    ("01:15", 70.0),
    ("01:30", 100.0),
    ("01:45", 120.0),
    ("02:00", 110.0),
    ("02:15", 90.0),
    ("02:30", 70.0),
    ("02:45", 50.0),
    ("03:00", 30.0),
    ("03:15", 15.0),
    ("03:30", 5.0),
    ("03:45", 0.0),
    ("04:00", 0.0),
    ("05:00", 0.0),
    ("06:00", 0.0),
]


# ============================================================
# HELPERS
# ============================================================

def clean_number(value, default=0.0):
    """Safely convert a value to a finite float."""

    try:
        value = float(value)

        if math.isfinite(value):
            return value

    except (TypeError, ValueError):
        pass

    return default


def swmm_node_name(node_id):
    """Convert numeric node ID to SWMM junction name."""

    return f"J_{int(node_id):04d}"


def swmm_conduit_name(index):
    """Convert edge row index to SWMM conduit name."""

    return f"C_{int(index):04d}"


def swmm_subcatchment_name(node_id):
    """Create a subcatchment name associated with a junction."""

    return f"S_{int(node_id):04d}"


# ============================================================
# LOAD NETWORK
# ============================================================

def load_network():

    print()
    print("=" * 70)
    print("LOADING MUMBAI NETWORK")
    print("=" * 70)

    print()
    print("Nodes file:")
    print(NODES_FILE)

    print()
    print("Edges file:")
    print(EDGES_FILE)

    if not NODES_FILE.exists():

        raise FileNotFoundError(
            f"\nMumbai nodes file not found:\n{NODES_FILE}"
        )

    if not EDGES_FILE.exists():

        raise FileNotFoundError(
            f"\nMumbai edges file not found:\n{EDGES_FILE}"
        )

    nodes = pd.read_csv(NODES_FILE)

    edges = pd.read_csv(EDGES_FILE)

    print()
    print(f"Nodes loaded: {len(nodes)}")
    print(f"Edges loaded: {len(edges)}")

    required_node_columns = {
        "node_id",
        "x",
        "y",
        "elevation",
    }

    required_edge_columns = {
        "from_node",
        "to_node",
        "length",
    }

    missing_nodes = (
        required_node_columns
        - set(nodes.columns)
    )

    missing_edges = (
        required_edge_columns
        - set(edges.columns)
    )

    if missing_nodes:

        raise ValueError(
            "Missing node columns: "
            + ", ".join(sorted(missing_nodes))
        )

    if missing_edges:

        raise ValueError(
            "Missing edge columns: "
            + ", ".join(sorted(missing_edges))
        )

    return nodes, edges


# ============================================================
# VALIDATE NETWORK
# ============================================================

def validate_network(nodes, edges):

    print()
    print("=" * 70)
    print("VALIDATING MUMBAI NETWORK")
    print("=" * 70)

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
    print(f"Node IDs found: {len(node_ids)}")

    # --------------------------------------------------------
    # Elevation
    # --------------------------------------------------------

    elevations = pd.to_numeric(
        nodes["elevation"],
        errors="coerce"
    )

    invalid_elevations = (
        elevations.isna()
        | ~elevations.map(math.isfinite)
    )

    if invalid_elevations.any():

        raise ValueError(
            "Some Mumbai nodes have invalid elevations."
        )

    print(
        f"Invalid elevations: "
        f"{int(invalid_elevations.sum())}"
    )

    # --------------------------------------------------------
    # X coordinates
    # --------------------------------------------------------

    x_values = pd.to_numeric(
        nodes["x"],
        errors="coerce"
    )

    invalid_x = (
        x_values.isna()
        | ~x_values.map(math.isfinite)
    )

    if invalid_x.any():

        raise ValueError(
            "Some Mumbai nodes have invalid X coordinates."
        )

    print(
        f"Invalid X coordinates: "
        f"{int(invalid_x.sum())}"
    )

    # --------------------------------------------------------
    # Y coordinates
    # --------------------------------------------------------

    y_values = pd.to_numeric(
        nodes["y"],
        errors="coerce"
    )

    invalid_y = (
        y_values.isna()
        | ~y_values.map(math.isfinite)
    )

    if invalid_y.any():

        raise ValueError(
            "Some Mumbai nodes have invalid Y coordinates."
        )

    print(
        f"Invalid Y coordinates: "
        f"{int(invalid_y.sum())}"
    )

    # --------------------------------------------------------
    # Edge endpoints
    # --------------------------------------------------------

    invalid_from = 0
    invalid_to = 0

    for _, edge in edges.iterrows():

        from_node = int(edge["from_node"])
        to_node = int(edge["to_node"])

        if from_node not in node_ids:
            invalid_from += 1

        if to_node not in node_ids:
            invalid_to += 1

    if invalid_from > 0:

        raise ValueError(
            f"{invalid_from} edges have invalid from_node IDs."
        )

    if invalid_to > 0:

        raise ValueError(
            f"{invalid_to} edges have invalid to_node IDs."
        )

    # --------------------------------------------------------
    # Lengths
    # --------------------------------------------------------

    lengths = pd.to_numeric(
        edges["length"],
        errors="coerce"
    )

    invalid_lengths = (
        lengths.isna()
        | (lengths <= 0)
    )

    if invalid_lengths.any():

        raise ValueError(
            "Some Mumbai conduit lengths are invalid."
        )

    print(
        f"Invalid conduit lengths: "
        f"{int(invalid_lengths.sum())}"
    )

    print()
    print("Mumbai network validation PASSED.")


# ============================================================
# FIND NATURAL TERMINALS
# ============================================================

def find_terminal_nodes(nodes, edges):

    from_nodes = set(
        edges["from_node"]
        .astype(int)
        .tolist()
    )

    to_nodes = set(
        edges["to_node"]
        .astype(int)
        .tolist()
    )

    terminals = sorted(
        to_nodes - from_nodes
    )

    if not terminals:

        raise ValueError(
            "No natural terminal node found."
        )

    return terminals


# ============================================================
# FIND LOWEST TERMINAL
# ============================================================

def find_lowest_terminal(nodes, terminals):

    terminal_data = []

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

        terminal_data.append(
            (
                int(node_id),
                elevation
            )
        )

    if not terminal_data:

        raise ValueError(
            "Could not determine terminal elevation."
        )

    return min(
        terminal_data,
        key=lambda x: x[1]
    )


# ============================================================
# BUILD JUNCTIONS
# ============================================================

def build_junctions(nodes, terminal_node):

    lines = []

    lines.append("[JUNCTIONS]")

    lines.append(
        ";;Name           Elevation   MaxDepth   "
        "InitDepth   SurDepth   Aponded"
    )

    lines.append(
        ";;-------------- ----------- ---------- "
        "----------- ---------- ----------"
    )

    for _, node in nodes.iterrows():

        node_id = int(node["node_id"])

        # The natural terminal becomes the SWMM outfall.
        if node_id == int(terminal_node):
            continue

        name = swmm_node_name(node_id)

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

def build_outfall(nodes, edges):

    terminals = find_terminal_nodes(
        nodes,
        edges
    )

    terminal_node, terminal_elevation = (
        find_lowest_terminal(
            nodes,
            terminals
        )
    )

    print()
    print("Natural terminal nodes:")

    for node_id in terminals:

        matches = nodes.loc[
            nodes["node_id"].astype(int)
            == int(node_id)
        ]

        elevation = clean_number(
            matches.iloc[0]["elevation"]
        )

        print(
            f"  {swmm_node_name(node_id)} "
            f"-> {elevation:.3f} m"
        )

    print()
    print(
        f"Selected natural terminal: "
        f"{swmm_node_name(terminal_node)}"
    )

    print(
        f"SWMM outfall: {OUTFALL_NAME}"
    )

    print(
        f"Outfall elevation: "
        f"{terminal_elevation:.3f} m"
    )

    lines = []

    lines.append("[OUTFALLS]")

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

    return lines, terminal_node


# ============================================================
# BUILD CONDUITS
# ============================================================

def build_conduits(edges, terminal_node):

    lines = []

    lines.append("[CONDUITS]")

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

        conduit_name = swmm_conduit_name(index)

        from_node_id = int(
            edge["from_node"]
        )

        to_node_id = int(
            edge["to_node"]
        )

        from_node = swmm_node_name(
            from_node_id
        )

        # ----------------------------------------------------
        # Connect natural terminal directly to OUTFALL_01
        # ----------------------------------------------------

        if to_node_id == int(terminal_node):

            to_node = OUTFALL_NAME

        else:

            to_node = swmm_node_name(
                to_node_id
            )

        length = clean_number(
            edge["length"]
        )

        roughness = clean_number(
            edge.get(
                "roughness",
                DEFAULT_ROUGHNESS
            ),
            DEFAULT_ROUGHNESS
        )

        width = clean_number(
            edge.get(
                "width",
                DEFAULT_WIDTH
            ),
            DEFAULT_WIDTH
        )

        depth = clean_number(
            edge.get(
                "depth",
                DEFAULT_DEPTH
            ),
            DEFAULT_DEPTH
        )

        if roughness <= 0:
            roughness = DEFAULT_ROUGHNESS

        if width <= 0:
            width = DEFAULT_WIDTH

        if depth <= 0:
            depth = DEFAULT_DEPTH

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

    lines.append("[XSECTIONS]")

    lines.append(
        ";;Link           Shape        Geom1       "
        "Geom2       Geom3       Geom4       Barrels"
    )

    lines.append(
        ";;-------------- ------------ ----------- "
        "----------- ----------- ----------- --------"
    )

    for index, edge in edges.iterrows():

        conduit_name = swmm_conduit_name(index)

        width = clean_number(
            edge.get(
                "width",
                DEFAULT_WIDTH
            ),
            DEFAULT_WIDTH
        )

        depth = clean_number(
            edge.get(
                "depth",
                DEFAULT_DEPTH
            ),
            DEFAULT_DEPTH
        )

        if width <= 0:
            width = DEFAULT_WIDTH

        if depth <= 0:
            depth = DEFAULT_DEPTH

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
# BUILD RAIN GAGE
# ============================================================

def build_raingage():

    lines = []

    lines.append("[RAINGAGES]")

    lines.append(
        ";;Name           Format    Interval SCF      Source"
    )

    lines.append(
        ";;-------------- --------- -------- -------- ----------------"
    )

    lines.append(
        f"{RAIN_GAGE_NAME:<16}"
        f"{'INTENSITY':<10}"
        f"{'0:15':<9}"
        f"{1.0:<9.3f}"
        f"TIMESERIES {RAIN_TIMESERIES_NAME}"
    )

    lines.append("")

    return lines


# ============================================================
# BUILD RAINFALL TIME SERIES
# ============================================================

def build_timeseries():

    lines = []

    lines.append("[TIMESERIES]")

    lines.append(
        ";;Name           Date        Time        Value"
    )

    lines.append(
        ";;-------------- ---------- ---------- ----------"
    )

    for time_value, rainfall in RAIN_SERIES:

        lines.append(
            f"{RAIN_TIMESERIES_NAME:<16}"
            f"{'01/01/2025':<11}"
            f"{time_value:<11}"
            f"{rainfall:>10.3f}"
        )

    lines.append("")

    return lines


# ============================================================
# CALCULATE CATCHMENT AREA
# ============================================================

def calculate_total_area(nodes):

    x = pd.to_numeric(
        nodes["x"],
        errors="coerce"
    )

    y = pd.to_numeric(
        nodes["y"],
        errors="coerce"
    )

    min_x = float(x.min())
    max_x = float(x.max())

    min_y = float(y.min())
    max_y = float(y.max())

    width = max_x - min_x
    height = max_y - min_y

    # Bounding-box area in square metres.
    total_area_m2 = width * height

    return total_area_m2


# ============================================================
# BUILD SUBCATCHMENTS
# ============================================================

def build_subcatchments(
    nodes,
    terminal_node
):

    active_nodes = nodes[
        nodes["node_id"].astype(int)
        != int(terminal_node)
    ].copy()

    total_area_m2 = calculate_total_area(
        active_nodes
    )

    if total_area_m2 <= 0:

        raise ValueError(
            "Calculated Mumbai catchment area is invalid."
        )

    number_of_subcatchments = len(
        active_nodes
    )

    area_per_subcatchment_m2 = (
        total_area_m2
        / number_of_subcatchments
    )

    # SWMM metric subcatchment area = hectares.
    area_per_subcatchment_ha = (
        area_per_subcatchment_m2
        / 10000.0
    )

    print()
    print(
        f"Approximate drainage bounding-box area: "
        f"{total_area_m2 / 1_000_000:.3f} km2"
    )

    print(
        f"Subcatchments created: "
        f"{number_of_subcatchments}"
    )

    print(
        f"Approximate area per subcatchment: "
        f"{area_per_subcatchment_ha:.6f} ha"
    )

    lines = []

    lines.append("[SUBCATCHMENTS]")

    lines.append(
        ";;Name           Rain Gage       Outlet          "
        "Area       %Imperv    Width      %Slope     CurbLen"
    )

    lines.append(
        ";;-------------- --------------- --------------- "
        "---------- ---------- ---------- ---------- ----------"
    )

    for _, node in active_nodes.iterrows():

        node_id = int(
            node["node_id"]
        )

        subcatchment = swmm_subcatchment_name(
            node_id
        )

        outlet = swmm_node_name(
            node_id
        )

        # Characteristic width approximation.
        width_m = math.sqrt(
            area_per_subcatchment_m2
        )

        if width_m <= 0:
            width_m = 10.0

        # Temporary representative slope.
        slope_percent = 1.0

        lines.append(
            f"{subcatchment:<16}"
            f"{RAIN_GAGE_NAME:<16}"
            f"{outlet:<16}"
            f"{area_per_subcatchment_ha:>10.5f}"
            f"{PERCENT_IMPERVIOUS:>11.2f}"
            f"{width_m:>11.2f}"
            f"{slope_percent:>11.2f}"
            f"{0.0:>11.2f}"
        )

    lines.append("")

    return lines, len(active_nodes)


# ============================================================
# BUILD SUBAREAS
# ============================================================

def build_subareas(
    nodes,
    terminal_node
):

    active_nodes = nodes[
        nodes["node_id"].astype(int)
        != int(terminal_node)
    ]

    lines = []

    lines.append("[SUBAREAS]")

    lines.append(
        ";;Name           N-Imperv   N-Perv     "
        "S-Imperv   S-Perv    PctZero   RouteTo"
    )

    lines.append(
        ";;-------------- ---------- ---------- "
        "---------- ---------- ---------- ----------"
    )

    for _, node in active_nodes.iterrows():

        node_id = int(
            node["node_id"]
        )

        subcatchment = swmm_subcatchment_name(
            node_id
        )

        # IMPORTANT:
        # RouteTo in [SUBAREAS] is OUTLET,
        # not a junction name.
        lines.append(
            f"{subcatchment:<16}"
            f"{MANNING_N_IMPERVIOUS:>11.3f}"
            f"{MANNING_N_PERVIOUS:>11.3f}"
            f"{DEPRESSION_STORAGE_IMPERVIOUS:>11.3f}"
            f"{DEPRESSION_STORAGE_PERVIOUS:>11.3f}"
            f"{PERCENT_ZERO_IMPERVIOUS:>11.2f}"
            f"{'OUTLET':>11}"
        )

    lines.append("")

    return lines


# ============================================================
# BUILD HORTON INFILTRATION
# ============================================================

def build_infiltration(
    nodes,
    terminal_node
):

    active_nodes = nodes[
        nodes["node_id"].astype(int)
        != int(terminal_node)
    ]

    lines = []

    lines.append("[INFILTRATION]")

    lines.append(
        ";;Subcatchment   MaxRate    MinRate    Decay     "
        "DryTime    MaxInf"
    )

    lines.append(
        ";;-------------- ---------- ---------- ---------- "
        "---------- ----------"
    )

    for _, node in active_nodes.iterrows():

        node_id = int(
            node["node_id"]
        )

        subcatchment = swmm_subcatchment_name(
            node_id
        )

        lines.append(
            f"{subcatchment:<16}"
            f"{HORTON_MAX_RATE:>11.3f}"
            f"{HORTON_MIN_RATE:>11.3f}"
            f"{HORTON_DECAY:>11.3f}"
            f"{HORTON_DRY_TIME:>11.3f}"
            f"{HORTON_MAX_INFIL:>11.3f}"
        )

    lines.append("")

    return lines


# ============================================================
# BUILD MAP
# ============================================================

def build_map_dimensions(nodes):

    x = pd.to_numeric(
        nodes["x"],
        errors="coerce"
    )

    y = pd.to_numeric(
        nodes["y"],
        errors="coerce"
    )

    min_x = float(x.min())
    max_x = float(x.max())

    min_y = float(y.min())
    max_y = float(y.max())

    x_margin = max(
        (max_x - min_x) * 0.02,
        10.0
    )

    y_margin = max(
        (max_y - min_y) * 0.02,
        10.0
    )

    return (
        f"DIMENSIONS "
        f"{min_x - x_margin:.3f} "
        f"{min_y - y_margin:.3f} "
        f"{max_x + x_margin:.3f} "
        f"{max_y + y_margin:.3f}"
    )


# ============================================================
# BUILD COORDINATES
# ============================================================

def build_coordinates(nodes):

    lines = []

    lines.append("[COORDINATES]")

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

        x = clean_number(
            node["x"]
        )

        y = clean_number(
            node["y"]
        )

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
        ""
    ]


# ============================================================
# BUILD SWMM FILE
# ============================================================

def build_swmm_file(
    nodes,
    edges
):

    SWMM_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find natural terminal
    # --------------------------------------------------------

    terminals = find_terminal_nodes(
        nodes,
        edges
    )

    terminal_node, terminal_elevation = (
        find_lowest_terminal(
            nodes,
            terminals
        )
    )

    # --------------------------------------------------------
    # Build outfall
    # --------------------------------------------------------

    outfall_lines, terminal_node = (
        build_outfall(
            nodes,
            edges
        )
    )

    # --------------------------------------------------------
    # Start file
    # --------------------------------------------------------

    lines = []

    # ========================================================
    # TITLE
    # ========================================================

    lines.extend([
        "[TITLE]",
        ";;",
        f";; {STUDY_AREA_NAME}",
        ";; Region: mumbai_hindmata",
        ";; Mumbai Hindmata Basin SWMM model",
        ";; Drainage network from validated GIS/OSM data",
        ";; Elevations attached to drainage nodes",
        ";; Coordinate system: EPSG:32643",
        ";; Rainfall: synthetic heavy-rain test storm",
        ";;",
        ""
    ])

    # ========================================================
    # OPTIONS
    # ========================================================

    lines.extend([
        "[OPTIONS]",
        ";;Option             Value",
        "FLOW_UNITS           CMS",
        "INFILTRATION         HORTON",
        "FLOW_ROUTING         DYNWAVE",
        "LINK_OFFSETS         DEPTH",
        "MIN_SLOPE            0",
        "ALLOW_PONDING        YES",
        "SKIP_STEADY_STATE    NO",
        "",
        "START_DATE           01/01/2025",
        "START_TIME           00:00:00",
        "END_DATE             01/01/2025",
        "END_TIME             06:00:00",
        "",
        "REPORT_START_DATE    01/01/2025",
        "REPORT_START_TIME    00:00:00",
        "WET_STEP             00:05:00",
        "DRY_STEP             01:00:00",
        "REPORT_STEP          00:05:00",
        "ROUTING_STEP         00:00:30",
        ""
    ])

    # ========================================================
    # RAINGAGES
    # ========================================================

    lines.extend(
        build_raingage()
    )

    # ========================================================
    # TIMESERIES
    # ========================================================

    lines.extend(
        build_timeseries()
    )

    # ========================================================
    # JUNCTIONS
    # ========================================================

    lines.extend(
        build_junctions(
            nodes,
            terminal_node
        )
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
            terminal_node
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
    # SUBCATCHMENTS
    # ========================================================

    subcatchment_lines, subcatchment_count = (
        build_subcatchments(
            nodes,
            terminal_node
        )
    )

    lines.extend(
        subcatchment_lines
    )

    # ========================================================
    # SUBAREAS
    # ========================================================

    lines.extend(
        build_subareas(
            nodes,
            terminal_node
        )
    )

    # ========================================================
    # INFILTRATION
    # ========================================================

    lines.extend(
        build_infiltration(
            nodes,
            terminal_node
        )
    )

    # ========================================================
    # LOSSES
    # ========================================================

    lines.extend(
        build_losses()
    )

    # ========================================================
    # REPORT
    # ========================================================

    lines.extend([
        "[REPORT]",
        "INPUT        NO",
        "CONTROLS     NO",
        "SUBCATCHMENTS ALL",
        "NODES        ALL",
        "LINKS        ALL",
        ""
    ])

    # ========================================================
    # MAP
    # ========================================================

    lines.extend([
        "[MAP]",
        build_map_dimensions(nodes),
        ""
    ])

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

    lines.extend([
        "[SYMBOLS]",
        ";;Gage           X-Coord      Y-Coord",
        ";;-------------- ------------ ------------",
        f"{RAIN_GAGE_NAME:<16}"
        f"{float(nodes['x'].mean()):>13.3f}"
        f"{float(nodes['y'].mean()):>13.3f}",
        ""
    ])

    # ========================================================
    # TAGS
    # ========================================================

    lines.extend([
        "[TAGS]",
        ";;Category       Name",
        ""
    ])

    # ========================================================
    # WRITE
    # ========================================================

    OUTPUT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print()
    print("=" * 70)
    print("MUMBAI SWMM INPUT CREATED")
    print("=" * 70)

    print()
    print("File:")
    print(OUTPUT_FILE)

    print()
    print(f"Nodes: {len(nodes)}")
    print(f"Conduits: {len(edges)}")
    print(f"Subcatchments: {subcatchment_count}")

    print()
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
    print("Rain gauge:")
    print(RAIN_GAGE_NAME)

    print()
    print("Rainfall time series:")
    print(RAIN_TIMESERIES_NAME)

    print()
    print("Coordinate system: EPSG:32643")
    print("Coordinates: nodes.csv x,y")

    return OUTPUT_FILE


# ============================================================
# VALIDATE SWMM FILE
# ============================================================

def validate_swmm_file(filepath):

    print()
    print("=" * 70)
    print("VALIDATING MUMBAI SWMM FILE")
    print("=" * 70)

    text = Path(filepath).read_text(
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
        "[JUNCTIONS]",
        "[OUTFALLS]",
        "[CONDUITS]",
        "[XSECTIONS]",
        "[SUBCATCHMENTS]",
        "[SUBAREAS]",
        "[INFILTRATION]",
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
            "Missing SWMM sections: "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # Check rain gauge
    # --------------------------------------------------------

    if RAIN_GAGE_NAME not in text:

        raise ValueError(
            "Mumbai rain gauge was not generated."
        )

    # --------------------------------------------------------
    # Check rainfall time series
    # --------------------------------------------------------

    if RAIN_TIMESERIES_NAME not in text:

        raise ValueError(
            "Mumbai rainfall time series was not generated."
        )

    # --------------------------------------------------------
    # Check outfall
    # --------------------------------------------------------

    if OUTFALL_NAME not in text:

        raise ValueError(
            "OUTFALL_01 was not generated."
        )

    # --------------------------------------------------------
    # Check coordinates
    # --------------------------------------------------------

    coordinate_section = text.split(
        "[COORDINATES]",
        1
    )[1]

    coordinate_section = coordinate_section.split(
        "[SYMBOLS]",
        1
    )[0]

    coordinates = []

    for line in coordinate_section.splitlines():

        stripped = line.strip()

        if not stripped:
            continue

        if stripped.startswith(";"):
            continue

        parts = stripped.split()

        if len(parts) < 3:
            continue

        try:

            x = float(parts[1])
            y = float(parts[2])

            coordinates.append(
                (x, y)
            )

        except ValueError:
            continue

    if not coordinates:

        raise ValueError(
            "No valid coordinates found."
        )

    non_zero = [
        (x, y)
        for x, y in coordinates
        if abs(x) > 0.001
        or abs(y) > 0.001
    ]

    if not non_zero:

        raise ValueError(
            "All coordinates are zero."
        )

    # --------------------------------------------------------
    # Check outfall connection
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
            "OUTFALL_01 is not connected to a conduit."
        )

    # --------------------------------------------------------
    # Count subcatchments
    # --------------------------------------------------------

    subcatchment_section = text.split(
        "[SUBCATCHMENTS]",
        1
    )[1]

    subcatchment_section = subcatchment_section.split(
        "[SUBAREAS]",
        1
    )[0]

    subcatchment_count = 0

    for line in subcatchment_section.splitlines():

        stripped = line.strip()

        if not stripped:
            continue

        if stripped.startswith(";"):
            continue

        subcatchment_count += 1

    if subcatchment_count == 0:

        raise ValueError(
            "No subcatchments were generated."
        )

    # --------------------------------------------------------
    # Validation passed
    # --------------------------------------------------------

    print()
    print("SWMM structure validation: PASSED")

    print(
        f"Valid coordinate points: "
        f"{len(coordinates)}"
    )

    print(
        f"Non-zero coordinate points: "
        f"{len(non_zero)}"
    )

    print(
        f"Outfall-connected conduits: "
        f"{outfall_connections}"
    )

    print(
        f"Subcatchments: "
        f"{subcatchment_count}"
    )

    print(
        f"Rainfall records: "
        f"{len(RAIN_SERIES)}"
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("#" * 70)
    print("# MUMBAI HINDMATA SWMM MODEL")
    print("#" * 70)

    print()
    print("Region:")
    print(REGION)

    print()
    print("Study area:")
    print(STUDY_AREA_NAME)

    print()
    print("Rainfall:")
    print("Synthetic heavy-rain test storm")

    # --------------------------------------------------------
    # 1. Load
    # --------------------------------------------------------

    nodes, edges = load_network()

    # --------------------------------------------------------
    # 2. Validate
    # --------------------------------------------------------

    validate_network(
        nodes,
        edges
    )

    # --------------------------------------------------------
    # 3. Build
    # --------------------------------------------------------

    filepath = build_swmm_file(
        nodes,
        edges
    )

    # --------------------------------------------------------
    # 4. Validate generated SWMM
    # --------------------------------------------------------

    validate_swmm_file(
        filepath
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("#" * 70)
    print("# MUMBAI SWMM FILE READY")
    print("#" * 70)

    print()
    print(
        "SWMM file:"
    )

    print(
        filepath
    )

    print()
    print(
        "NEXT STEP:"
    )

    print(
        "Open the generated .inp file in EPA SWMM "
        "and run the simulation."
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()