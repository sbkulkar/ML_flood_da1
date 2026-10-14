# ============================================================
# pipeline.py
#
# MULTI-REGION DRAINAGE NETWORK PIPELINE
#
# Supports:
#   - Velachery
#   - Adyar Coastal
#   - Mumbai Mithi
#   - Mumbai Hindmata
#
# ============================================================

from pathlib import Path
import json
import sys

import geopandas as gpd
import pandas as pd
import requests

from shapely.geometry import LineString


# ============================================================
# CONFIGURATION
# ============================================================

from config import (
    PROJECT_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    NETWORK_DIR,

    DRAINAGE_API_URL,
    OSM_OVERPASS_URL,

    REGION,
    CURRENT_REGION,

    STUDY_AREA_NAME,
    STUDY_AREA_DESCRIPTION,
    OUTFALL_TARGET,
    BBOX,

    SOURCE_CRS,
    PROJECTED_CRS,

    DRAINAGE_SOURCE,

    SNAP_TOLERANCE_M,

    DEFAULT_WIDTH_M,
    DEFAULT_DEPTH_M,
    DEFAULT_ROUGHNESS,

    DEM_DIR,
)


# ============================================================
# ELEVATION
# ============================================================

from elevation import attach_elevations


# ============================================================
# NETWORK
# ============================================================

from network import (
    NetworkConfig,
    construct_network,
)


# ============================================================
# DIRECTORIES
# ============================================================

PROJECT_DIR = Path(PROJECT_DIR)
RAW_DIR = Path(RAW_DIR)
PROCESSED_DIR = Path(PROCESSED_DIR)
NETWORK_DIR = Path(NETWORK_DIR)
DEM_DIR = Path(DEM_DIR)

REGION_RAW_DIR = RAW_DIR / REGION
REGION_PROCESSED_DIR = PROCESSED_DIR / REGION

REGION_RAW_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REGION_PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

NETWORK_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DEM_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DISPLAY CONFIGURATION
# ============================================================

def print_configuration():

    print()
    print("=" * 70)
    print("ACTIVE STUDY REGION")
    print("=" * 70)

    print()
    print("Region:", REGION)

    print(
        "City:",
        CURRENT_REGION.get("city", "unknown")
    )

    print(
        "Study area:",
        STUDY_AREA_NAME
    )

    print(
        "Description:",
        STUDY_AREA_DESCRIPTION
    )

    print(
        "Outfall target:",
        OUTFALL_TARGET
    )

    print(
        "Bounding box:",
        BBOX
    )

    print()
    print(
        "Drainage source:",
        DRAINAGE_SOURCE
    )

    if DRAINAGE_SOURCE == "gcc_api":

        print(
            "Source URL:",
            DRAINAGE_API_URL
        )

    elif DRAINAGE_SOURCE == "osm_overpass":

        print(
            "Source URL:",
            OSM_OVERPASS_URL
        )

    print()
    print(
        "Source CRS:",
        SOURCE_CRS
    )

    print(
        "Projected CRS:",
        PROJECTED_CRS
    )

    print(
        "DEM directory:",
        DEM_DIR
    )

    print()
    print("=" * 70)


# ============================================================
# CHENNAI GCC API
# ============================================================

def download_gcc_drainage():

    print()
    print("=" * 70)
    print("CHENNAI GCC DRAINAGE ACQUISITION")
    print("=" * 70)

    min_lon = BBOX[0]
    min_lat = BBOX[1]
    max_lon = BBOX[2]
    max_lat = BBOX[3]

    geometry = (
        f"{min_lon},{min_lat},"
        f"{max_lon},{max_lat}"
    )

    params = {
        "where": "1=1",
        "geometry": geometry,
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": 4326,
        "f": "geojson",
    }

    print()
    print("GCC API:")
    print(DRAINAGE_API_URL)

    print()
    print("Bounding box:")
    print(BBOX)

    response = requests.get(
        DRAINAGE_API_URL,
        params=params,
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    if "features" not in data:

        raise ValueError(
            "GCC API response does not contain features."
        )

    gdf = gpd.GeoDataFrame.from_features(
        data["features"],
        crs=SOURCE_CRS,
    )

    print()
    print(
        f"GCC features downloaded: "
        f"{len(gdf)}"
    )

    return gdf


# ============================================================
# MUMBAI OSM OVERPASS
# ============================================================

def download_osm_drainage():

    print()
    print("=" * 70)
    print("MUMBAI OSM DRAINAGE ACQUISITION")
    print("=" * 70)

    min_lon = BBOX[0]
    min_lat = BBOX[1]
    max_lon = BBOX[2]
    max_lat = BBOX[3]

    # Overpass bbox order:
    #
    # south, west, north, east
    #

    bbox = (
        f"{min_lat},"
        f"{min_lon},"
        f"{max_lat},"
        f"{max_lon}"
    )

    query = f"""
    [out:json][timeout:180];

    (
        way["waterway"="drain"]({bbox});
        way["waterway"="ditch"]({bbox});
        way["waterway"="canal"]({bbox});
        way["waterway"="stream"]({bbox});
    );

    out geom;
    """

    print()
    print("Overpass API:")
    print(OSM_OVERPASS_URL)

    print()
    print("Bounding box:")
    print(BBOX)

    # --------------------------------------------------------
    # HTTP HEADERS
    # --------------------------------------------------------

    headers = {
        "User-Agent": (
            "MumbaiFloodDigitalTwin/1.0 "
            "(OpenStreetMap data acquisition)"
        ),
        "Accept": "application/json",
        "Content-Type": (
            "application/x-www-form-urlencoded"
        ),
        "Referer": (
            "https://www.openstreetmap.org/"
        ),
    }

    try:

        response = requests.post(
            OSM_OVERPASS_URL,
            data={
                "data": query
            },
            headers=headers,
            timeout=240,
        )

        print()
        print(
            "Overpass HTTP status:",
            response.status_code
        )

        response.raise_for_status()

    except requests.exceptions.HTTPError as e:

        print()
        print(
            "ERROR: Overpass API request failed."
        )

        print(
            "HTTP status:",
            response.status_code
        )

        print(
            "Response:",
            response.text[:1000]
        )

        raise RuntimeError(
            "Overpass API returned an HTTP error. "
            f"Status: {response.status_code}"
        ) from e

    except requests.exceptions.RequestException as e:

        raise RuntimeError(
            "Could not connect to Overpass API."
        ) from e

    # --------------------------------------------------------
    # PARSE RESPONSE
    # --------------------------------------------------------

    data = response.json()

    elements = data.get(
        "elements",
        []
    )

    print()
    print(
        f"OSM elements downloaded: "
        f"{len(elements)}"
    )

    records = []

    # --------------------------------------------------------
    # CONVERT OSM WAYS
    # --------------------------------------------------------

    for element in elements:

        geometry = element.get(
            "geometry"
        )

        if not geometry:
            continue

        coordinates = []

        for point in geometry:

            lat = point.get("lat")
            lon = point.get("lon")

            if (
                lat is None
                or lon is None
            ):
                continue

            coordinates.append(
                (
                    float(lon),
                    float(lat),
                )
            )

        if len(coordinates) < 2:
            continue

        line = LineString(
            coordinates
        )

        tags = element.get(
            "tags",
            {}
        )

        records.append(
            {
                "osm_id":
                    element.get("id"),

                "waterway":
                    tags.get("waterway"),

                "name":
                    tags.get("name"),

                "geometry":
                    line,
            }
        )

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    if not records:

        raise ValueError(
            "No OSM drainage/waterway "
            "features were found for "
            f"region '{REGION}'."
        )

    gdf = gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs=SOURCE_CRS,
    )

    print()
    print(
        f"OSM drainage lines created: "
        f"{len(gdf)}"
    )

    return gdf


# ============================================================
# SAVE RAW DATA
# ============================================================

def save_raw_drainage(gdf):

    raw_file = (
        REGION_RAW_DIR
        / "drainage_raw.geojson"
    )

    geographic = gdf.to_crs(
        SOURCE_CRS
    )

    geographic.to_file(
        raw_file,
        driver="GeoJSON",
    )

    print()
    print(
        "Raw drainage saved:"
    )

    print(
        raw_file
    )

    return raw_file


# ============================================================
# STEP 1
# LOAD RAW DRAINAGE
# ============================================================

def load_raw_drainage():

    print()
    print("=" * 70)
    print("STEP 1: LOAD RAW DRAINAGE")
    print("=" * 70)

    # --------------------------------------------------------
    # SELECT SOURCE ACCORDING TO REGION
    # --------------------------------------------------------

    if DRAINAGE_SOURCE == "gcc_api":

        gdf = download_gcc_drainage()

    elif DRAINAGE_SOURCE == "osm_overpass":

        gdf = download_osm_drainage()

    else:

        raise ValueError(
            "Unsupported drainage source: "
            f"{DRAINAGE_SOURCE}"
        )

    if gdf.empty:

        raise ValueError(
            f"No drainage features found "
            f"for region '{REGION}'."
        )

    print()
    print(
        f"Raw features loaded: "
        f"{len(gdf)}"
    )

    save_raw_drainage(
        gdf
    )

    return gdf


# ============================================================
# STEP 2
# CLEAN DRAINAGE
# ============================================================

def clean_drainage(gdf):

    print()
    print("=" * 70)
    print("STEP 2: CLEAN DRAINAGE")
    print("=" * 70)

    gdf = gdf.copy()

    # --------------------------------------------------------
    # CRS
    # --------------------------------------------------------

    if gdf.crs is None:

        gdf = gdf.set_crs(
            SOURCE_CRS
        )

    else:

        gdf = gdf.to_crs(
            SOURCE_CRS
        )

    # --------------------------------------------------------
    # REMOVE MISSING GEOMETRY
    # --------------------------------------------------------

    before = len(gdf)

    gdf = gdf[
        gdf.geometry.notna()
    ].copy()

    print(
        "Removed missing geometry:",
        before - len(gdf)
    )

    # --------------------------------------------------------
    # REMOVE EMPTY GEOMETRY
    # --------------------------------------------------------

    before = len(gdf)

    gdf = gdf[
        ~gdf.geometry.is_empty
    ].copy()

    print(
        "Removed empty geometry:",
        before - len(gdf)
    )

    # --------------------------------------------------------
    # KEEP LINE GEOMETRIES
    # --------------------------------------------------------

    before = len(gdf)

    gdf = gdf[
        gdf.geometry.geom_type.isin(
            [
                "LineString",
                "MultiLineString",
            ]
        )
    ].copy()

    print(
        "Removed non-line geometry:",
        before - len(gdf)
    )

    # --------------------------------------------------------
    # REMOVE ZERO-LENGTH GEOMETRY
    # --------------------------------------------------------

    projected = gdf.to_crs(
        PROJECTED_CRS
    )

    before = len(gdf)

    gdf = gdf[
        projected.geometry.length > 0
    ].copy()

    print(
        "Removed zero-length geometry:",
        before - len(gdf)
    )

    # --------------------------------------------------------
    # RESET INDEX
    # --------------------------------------------------------

    gdf = gdf.reset_index(
        drop=True
    )

    if gdf.empty:

        raise ValueError(
            "No drainage segments remain "
            "after cleaning."
        )

    print()
    print(
        f"Cleaned drainage segments: "
        f"{len(gdf)}"
    )

    return gdf


# ============================================================
# STEP 3
# COORDINATE PREPARATION
# ============================================================

def prepare_coordinates(gdf):

    print()
    print("=" * 70)
    print("STEP 3: COORDINATE PREPARATION")
    print("=" * 70)

    projected = gdf.to_crs(
        PROJECTED_CRS
    ).copy()

    sx = []
    sy = []
    ex = []
    ey = []

    for geometry in projected.geometry:

        if geometry.geom_type == "MultiLineString":

            geometry = max(
                geometry.geoms,
                key=lambda x: x.length,
            )

        coords = list(
            geometry.coords
        )

        start = coords[0]
        end = coords[-1]

        sx.append(
            float(start[0])
        )

        sy.append(
            float(start[1])
        )

        ex.append(
            float(end[0])
        )

        ey.append(
            float(end[1])
        )

    projected["sx"] = sx
    projected["sy"] = sy

    projected["ex"] = ex
    projected["ey"] = ey

    # --------------------------------------------------------
    # SWMM LENGTH
    # --------------------------------------------------------

    projected["swmm_length"] = (
        projected.geometry.length
    )

    # --------------------------------------------------------
    # WIDTH
    # --------------------------------------------------------

    if "swmm_width" not in projected.columns:

        projected["swmm_width"] = (
            DEFAULT_WIDTH_M
        )

    else:

        projected["swmm_width"] = (
            pd.to_numeric(
                projected["swmm_width"],
                errors="coerce",
            )
            .fillna(
                DEFAULT_WIDTH_M
            )
        )

    # --------------------------------------------------------
    # DEPTH
    # --------------------------------------------------------

    if "swmm_depth" not in projected.columns:

        projected["swmm_depth"] = (
            DEFAULT_DEPTH_M
        )

    else:

        projected["swmm_depth"] = (
            pd.to_numeric(
                projected["swmm_depth"],
                errors="coerce",
            )
            .fillna(
                DEFAULT_DEPTH_M
            )
        )

    # --------------------------------------------------------
    # ROUGHNESS
    # --------------------------------------------------------

    if "roughness" not in projected.columns:

        projected["roughness"] = (
            DEFAULT_ROUGHNESS
        )

    else:

        projected["roughness"] = (
            pd.to_numeric(
                projected["roughness"],
                errors="coerce",
            )
            .fillna(
                DEFAULT_ROUGHNESS
            )
        )

    print()
    print(
        "Projected CRS:",
        PROJECTED_CRS
    )

    print(
        "Endpoint coordinates prepared."
    )

    print(
        f"Coordinate system prepared "
        f"for {len(projected)} segments."
    )

    return projected


# ============================================================
# STEP 4
# SAVE CLEAN DRAINAGE
# ============================================================

def save_clean_drainage(gdf):

    output = (
        REGION_PROCESSED_DIR
        / "drainage_clean.geojson"
    )

    geographic = gdf.to_crs(
        SOURCE_CRS
    )

    geographic.to_file(
        output,
        driver="GeoJSON",
    )

    print()
    print(
        "Clean drainage saved:"
    )

    print(
        output
    )

    return output


# ============================================================
# STEP 5
# ELEVATION
# ============================================================

def add_elevation(gdf):

    print()
    print("=" * 70)
    print("STEP 5: ELEVATION")
    print("=" * 70)

    print()
    print(
        "Using DEM directory:"
    )

    print(
        DEM_DIR
    )

    # IMPORTANT:
    # Region-specific DEM directory
    gdf = attach_elevations(
        gdf,
        dem_dir=DEM_DIR,
    )

    required = {
        "start_elevation",
        "end_elevation",
    }

    missing = (
        required
        - set(gdf.columns)
    )

    if missing:

        raise ValueError(
            "Elevation stage did not "
            "produce required columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    start_valid = (
        pd.to_numeric(
            gdf["start_elevation"],
            errors="coerce",
        )
        .notna()
        .sum()
    )

    end_valid = (
        pd.to_numeric(
            gdf["end_elevation"],
            errors="coerce",
        )
        .notna()
        .sum()
    )

    print()

    print(
        f"Valid start elevations: "
        f"{start_valid}/{len(gdf)}"
    )

    print(
        f"Valid end elevations: "
        f"{end_valid}/{len(gdf)}"
    )

    if (
        start_valid != len(gdf)
        or
        end_valid != len(gdf)
    ):

        raise ValueError(
            "Elevation stage produced "
            "missing endpoint elevations."
        )

    return gdf


# ============================================================
# STEP 6
# BUILD NETWORK
# ============================================================

def build_drainage_network(gdf):

    print()
    print("=" * 70)
    print("STEP 6: NETWORK CONSTRUCTION")
    print("=" * 70)

    network_config = NetworkConfig(
        snap_tolerance_m=
            SNAP_TOLERANCE_M
    )

    result = construct_network(
        gdf,
        config=network_config,
    )

    return result


# ============================================================
# STEP 7
# SAVE NETWORK TABLES
# ============================================================

def save_network_tables(result):

    print()
    print("=" * 70)
    print("STEP 7: SAVE NETWORK DATA")
    print("=" * 70)

    nodes = result["nodes"]
    edges = result["edges"]
    endpoints = result["endpoints"]
    components = result["components"]
    duplicates = result["duplicates"]

    nodes_file = (
        NETWORK_DIR
        / "nodes.csv"
    )

    edges_file = (
        NETWORK_DIR
        / "edges.csv"
    )

    endpoints_file = (
        NETWORK_DIR
        / "endpoints.csv"
    )

    components_file = (
        NETWORK_DIR
        / "components.csv"
    )

    duplicates_file = (
        NETWORK_DIR
        / "duplicate_edges.csv"
    )

    nodes.to_csv(
        nodes_file,
        index=False,
    )

    edges.to_csv(
        edges_file,
        index=False,
    )

    endpoints.to_csv(
        endpoints_file,
        index=False,
    )

    components.to_csv(
        components_file,
        index=False,
    )

    duplicates.to_csv(
        duplicates_file,
        index=False,
    )

    print()
    print("Nodes:", nodes_file)
    print("Edges:", edges_file)
    print("Endpoints:", endpoints_file)
    print("Components:", components_file)
    print("Duplicates:", duplicates_file)


# ============================================================
# STEP 8
# VALIDATION REPORT
# ============================================================

def save_validation_report(result):

    report_file = (
        NETWORK_DIR
        / "network_validation.json"
    )

    validation = result[
        "validation"
    ]

    validation["region"] = REGION

    validation["study_area"] = (
        STUDY_AREA_NAME
    )

    validation["outfall_target"] = (
        OUTFALL_TARGET
    )

    validation["drainage_source"] = (
        DRAINAGE_SOURCE
    )

    validation["projected_crs"] = (
        str(PROJECTED_CRS)
    )

    with open(
        report_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            validation,
            f,
            indent=2,
        )

    print()
    print(
        "Validation report saved:"
    )

    print(
        report_file
    )


# ============================================================
# STEP 9
# NETWORK SUMMARY
# ============================================================

def print_network_summary(result):

    validation = result[
        "validation"
    ]

    components = result[
        "components"
    ]

    print()
    print("=" * 70)
    print("NETWORK SUMMARY")
    print("=" * 70)

    print()
    print(
        "Region:",
        REGION
    )

    print(
        "Study area:",
        STUDY_AREA_NAME
    )

    print(
        "Drainage source:",
        DRAINAGE_SOURCE
    )

    print()
    print(
        f"Nodes: "
        f"{validation['nodes']}"
    )

    print(
        f"Edges: "
        f"{validation['edges']}"
    )

    print(
        f"Weak components: "
        f"{validation['weak_components']}"
    )

    print(
        f"Strong components: "
        f"{validation['strong_components']}"
    )

    print(
        f"Natural terminals: "
        f"{validation['terminals']}"
    )

    print(
        f"Headwaters: "
        f"{validation['headwaters']}"
    )

    print(
        f"Terminal reachability: "
        f"{validation['terminal_reachability_pct']:.2f}%"
    )

    print(
        f"Uphill edges: "
        f"{validation['uphill_edges']}"
    )

    print()
    print(
        "Largest connected components:"
    )

    if not components.empty:

        print(
            components
            .head(10)
            .to_string(
                index=False
            )
        )


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline():

    print()

    print("#" * 70)

    print(
        f"# {REGION.upper()} DRAINAGE NETWORK PIPELINE"
    )

    print("#" * 70)

    # --------------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------------

    print_configuration()

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    gdf = load_raw_drainage()

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    gdf = clean_drainage(
        gdf
    )

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    gdf = prepare_coordinates(
        gdf
    )

    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    save_clean_drainage(
        gdf
    )

    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    gdf = add_elevation(
        gdf
    )

    # --------------------------------------------------------
    # STEP 6
    # --------------------------------------------------------

    result = build_drainage_network(
        gdf
    )

    # --------------------------------------------------------
    # STEP 7
    # --------------------------------------------------------

    save_network_tables(
        result
    )

    # --------------------------------------------------------
    # STEP 8
    # --------------------------------------------------------

    save_validation_report(
        result
    )

    # --------------------------------------------------------
    # STEP 9
    # --------------------------------------------------------

    print_network_summary(
        result
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print()

    print("#" * 70)

    print(
        "# NETWORK BUILD COMPLETE"
    )

    print("#" * 70)

    print()

    print(
        "Region:",
        REGION
    )

    print(
        "Study area:",
        STUDY_AREA_NAME
    )

    print(
        "Drainage source:",
        DRAINAGE_SOURCE
    )

    print()

    print(
        "SWMM generation is the next stage."
    )

    print(
        "RL integration comes after SWMM validation."
    )

    print()

    return result


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_pipeline()