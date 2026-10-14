# ============================================================
# elevation.py
#
# SRTM ELEVATION
#
# Responsibilities:
#   - determine required SRTM tiles
#   - download HGT files when required
#   - read SRTM elevation
#   - attach elevation to drainage endpoints
#
# Input:
#   Projected drainage GeoDataFrame containing:
#       sx, sy, ex, ey
#
# Output:
#   Same GeoDataFrame with:
#       start_elevation
#       end_elevation
#
# ============================================================

from __future__ import annotations

import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# ============================================================
# CONSTANTS
# ============================================================

SRTM_URL = (
    "https://terrain.ardupilot.org/SRTM3/Eurasia/"
)

SRTM_SIZE = 1201


# ============================================================
# TILE NAME
# ============================================================

def get_tile_name(
    latitude: float,
    longitude: float,
) -> str:
    """
    Return the SRTM tile containing a coordinate.

    Examples:
        12.99, 80.21 -> N12E080
        19.01, 72.83 -> N19E072
    """

    latitude = float(latitude)
    longitude = float(longitude)

    lat_degree = math.floor(latitude)
    lon_degree = math.floor(longitude)

    lat_prefix = "N" if lat_degree >= 0 else "S"
    lon_prefix = "E" if lon_degree >= 0 else "W"

    return (
        f"{lat_prefix}{abs(lat_degree):02d}"
        f"{lon_prefix}{abs(lon_degree):03d}"
    )


# ============================================================
# DOWNLOAD TILE
# ============================================================

def download_srtm_tile(
    tile_name: str,
    dem_dir: Path,
) -> Path:

    dem_dir = Path(dem_dir)

    dem_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    hgt_file = (
        dem_dir
        / f"{tile_name}.hgt"
    )

    zip_file = (
        dem_dir
        / f"{tile_name}.hgt.zip"
    )

    # --------------------------------------------------------
    # Already extracted
    # --------------------------------------------------------

    if hgt_file.exists():

        print(
            f"HGT already exists: {hgt_file}"
        )

        return hgt_file

    # --------------------------------------------------------
    # Download ZIP
    # --------------------------------------------------------

    url = (
        SRTM_URL
        + f"{tile_name}.hgt.zip"
    )

    print()
    print(
        f"Downloading SRTM tile: {tile_name}"
    )

    print(
        f"URL: {url}"
    )

    response = requests.get(
        url,
        timeout=120,
    )

    response.raise_for_status()

    zip_file.write_bytes(
        response.content
    )

    print(
        f"Downloaded: {zip_file}"
    )

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    print(
        "Extracting DEM..."
    )

    with zipfile.ZipFile(
        zip_file,
        "r",
    ) as archive:

        archive.extractall(
            dem_dir
        )

    if not hgt_file.exists():

        candidates = list(
            dem_dir.glob(
                f"{tile_name}*.hgt"
            )
        )

        if not candidates:

            raise FileNotFoundError(
                f"Could not find HGT file for {tile_name}"
            )

        hgt_file = candidates[0]

    print(
        f"HGT extracted: {hgt_file}"
    )

    return hgt_file


# ============================================================
# LOAD HGT
# ============================================================

def load_hgt(
    hgt_file: Path,
) -> np.ndarray:

    """
    Load an SRTM HGT file.

    SRTM3:
        1201 x 1201
        signed 16-bit integer
        big-endian
    """

    data = np.fromfile(
        hgt_file,
        dtype=">i2",
    )

    expected = (
        SRTM_SIZE
        * SRTM_SIZE
    )

    if len(data) != expected:

        raise ValueError(
            f"Invalid HGT size for {hgt_file}: "
            f"{len(data)} values, "
            f"expected {expected}"
        )

    return data.reshape(
        (
            SRTM_SIZE,
            SRTM_SIZE,
        )
    )


# ============================================================
# SAMPLE HGT
# ============================================================

def sample_hgt(
    data: np.ndarray,
    latitude: float,
    longitude: float,
) -> float:

    """
    Bilinearly sample an SRTM tile.

    HGT:
        first row = northern edge
        first column = western edge
    """

    lat_floor = math.floor(
        latitude
    )

    lon_floor = math.floor(
        longitude
    )

    lat_fraction = (
        latitude
        - lat_floor
    )

    lon_fraction = (
        longitude
        - lon_floor
    )

    # --------------------------------------------------------
    # Convert geographic coordinate
    # into HGT row / column
    # --------------------------------------------------------

    row_float = (
        (1.0 - lat_fraction)
        * (SRTM_SIZE - 1)
    )

    col_float = (
        lon_fraction
        * (SRTM_SIZE - 1)
    )

    row_float = np.clip(
        row_float,
        0,
        SRTM_SIZE - 1,
    )

    col_float = np.clip(
        col_float,
        0,
        SRTM_SIZE - 1,
    )

    r0 = int(
        math.floor(row_float)
    )

    c0 = int(
        math.floor(col_float)
    )

    r1 = min(
        r0 + 1,
        SRTM_SIZE - 1,
    )

    c1 = min(
        c0 + 1,
        SRTM_SIZE - 1,
    )

    dr = (
        row_float
        - r0
    )

    dc = (
        col_float
        - c0
    )

    q11 = float(
        data[r0, c0]
    )

    q12 = float(
        data[r0, c1]
    )

    q21 = float(
        data[r1, c0]
    )

    q22 = float(
        data[r1, c1]
    )

    # --------------------------------------------------------
    # Handle SRTM void value
    # --------------------------------------------------------

    values = np.array(
        [
            q11,
            q12,
            q21,
            q22,
        ]
    )

    valid = (
        values
        != -32768
    )

    if not valid.any():

        return np.nan

    if not valid.all():

        return float(
            np.mean(
                values[valid]
            )
        )

    # --------------------------------------------------------
    # Bilinear interpolation
    # --------------------------------------------------------

    value = (
        q11
        * (1 - dr)
        * (1 - dc)

        + q12
        * (1 - dr)
        * dc

        + q21
        * dr
        * (1 - dc)

        + q22
        * dr
        * dc
    )

    return float(value)


# ============================================================
# GET ELEVATION
# ============================================================

def get_elevation(
    latitude: float,
    longitude: float,
    dem_dir: Path,
    tile_cache: dict,
) -> float:

    tile_name = get_tile_name(
        latitude,
        longitude,
    )

    if tile_name not in tile_cache:

        hgt_file = download_srtm_tile(
            tile_name,
            dem_dir,
        )

        print(
            f"Loading DEM: {hgt_file.name}"
        )

        tile_cache[tile_name] = load_hgt(
            hgt_file
        )

    data = tile_cache[
        tile_name
    ]

    return sample_hgt(
        data,
        latitude,
        longitude,
    )


# ============================================================
# ATTACH ELEVATIONS
# ============================================================

def attach_elevations(
    gdf,
    dem_dir=None,
):

    """
    Attach real SRTM elevations to drainage endpoints.

    Expected columns:
        sx
        sy
        ex
        ey

    IMPORTANT:
        The GeoDataFrame CRS is used dynamically.

        Chennai:
            EPSG:32644

        Mumbai:
            EPSG:32643

    Therefore there is NO hardcoded Chennai CRS here.
    """

    gdf = gdf.copy()

    # --------------------------------------------------------
    # Resolve DEM directory
    # --------------------------------------------------------

    if dem_dir is None:

        dem_dir = (
            Path.cwd()
            / "data"
            / "dem"
        )

    dem_dir = Path(
        dem_dir
    )

    dem_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Validate coordinates
    # --------------------------------------------------------

    required = {
        "sx",
        "sy",
        "ex",
        "ey",
    }

    missing = (
        required
        - set(gdf.columns)
    )

    if missing:

        raise ValueError(
            "Missing coordinate columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    # --------------------------------------------------------
    # Validate CRS
    # --------------------------------------------------------

    if gdf.crs is None:

        raise ValueError(
            "Drainage GeoDataFrame has no CRS. "
            "Set the correct projected CRS "
            "before attaching elevations."
        )

    source_crs = gdf.crs

    print()
    print(
        f"Elevation coordinate conversion: "
        f"{source_crs} -> EPSG:4326"
    )

    # --------------------------------------------------------
    # Convert projected coordinates
    # to geographic coordinates
    # --------------------------------------------------------

    from pyproj import Transformer

    transformer = Transformer.from_crs(
        source_crs,
        "EPSG:4326",
        always_xy=True,
    )

    start_lon = []
    start_lat = []

    end_lon = []
    end_lat = []

    for _, row in gdf.iterrows():

        lon, lat = transformer.transform(
            float(row["sx"]),
            float(row["sy"]),
        )

        start_lon.append(
            lon
        )

        start_lat.append(
            lat
        )

        lon, lat = transformer.transform(
            float(row["ex"]),
            float(row["ey"]),
        )

        end_lon.append(
            lon
        )

        end_lat.append(
            lat
        )

    # --------------------------------------------------------
    # Coordinate sanity check
    # --------------------------------------------------------

    all_lons = (
        start_lon
        + end_lon
    )

    all_lats = (
        start_lat
        + end_lat
    )

    if all_lons and all_lats:

        print(
            f"Geographic extent used for SRTM: "
            f"lon {min(all_lons):.6f} "
            f"to {max(all_lons):.6f}, "
            f"lat {min(all_lats):.6f} "
            f"to {max(all_lats):.6f}"
        )

        if not all(
            -180.0 <= lon <= 180.0
            for lon in all_lons
        ):

            raise ValueError(
                "Invalid longitude produced "
                "during CRS conversion."
            )

        if not all(
            -90.0 <= lat <= 90.0
            for lat in all_lats
        ):

            raise ValueError(
                "Invalid latitude produced "
                "during CRS conversion."
            )

    # --------------------------------------------------------
    # Required tiles
    # --------------------------------------------------------

    tile_names = set()

    for lat, lon in zip(
        start_lat,
        start_lon,
    ):

        tile_names.add(
            get_tile_name(
                lat,
                lon,
            )
        )

    for lat, lon in zip(
        end_lat,
        end_lon,
    ):

        tile_names.add(
            get_tile_name(
                lat,
                lon,
            )
        )

    print()
    print(
        "=" * 70
    )

    print(
        "REAL SRTM ELEVATION"
    )

    print(
        "=" * 70
    )

    print(
        "Required SRTM tiles:"
    )

    for tile in sorted(
        tile_names
    ):

        print(
            f"  - {tile}"
        )

    # --------------------------------------------------------
    # Load tiles
    # --------------------------------------------------------

    tile_cache = {}

    for tile in sorted(
        tile_names
    ):

        hgt_file = download_srtm_tile(
            tile,
            dem_dir,
        )

        print(
            f"Loading DEM: {hgt_file.name}"
        )

        tile_cache[tile] = load_hgt(
            hgt_file
        )

    # --------------------------------------------------------
    # Sample elevations
    # --------------------------------------------------------

    start_elevations = []
    end_elevations = []

    for lat, lon in zip(
        start_lat,
        start_lon,
    ):

        elevation = get_elevation(
            lat,
            lon,
            dem_dir,
            tile_cache,
        )

        start_elevations.append(
            elevation
        )

    for lat, lon in zip(
        end_lat,
        end_lon,
    ):

        elevation = get_elevation(
            lat,
            lon,
            dem_dir,
            tile_cache,
        )

        end_elevations.append(
            elevation
        )

    # --------------------------------------------------------
    # Attach
    # --------------------------------------------------------

    gdf[
        "start_elevation"
    ] = start_elevations

    gdf[
        "end_elevation"
    ] = end_elevations

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    all_elevations = np.concatenate(
        [
            np.asarray(
                start_elevations,
                dtype=float,
            ),

            np.asarray(
                end_elevations,
                dtype=float,
            ),
        ]
    )

    valid = np.isfinite(
        all_elevations
    )

    print()
    print(
        "Elevation statistics:"
    )

    print(
        f"Endpoint elevations: "
        f"{len(all_elevations)}"
    )

    print(
        f"Valid elevations: "
        f"{valid.sum()}"
    )

    print(
        f"Missing elevations: "
        f"{(~valid).sum()}"
    )

    if valid.any():

        print(
            f"Minimum: "
            f"{all_elevations[valid].min():.2f} m"
        )

        print(
            f"Maximum: "
            f"{all_elevations[valid].max():.2f} m"
        )

        print(
            f"Mean: "
            f"{all_elevations[valid].mean():.2f} m"
        )

    return gdf


# ============================================================
# COMPATIBILITY ALIAS
# ============================================================

def get_node_elevations(
    nodes,
    dem_dir,
):

    """
    Compatibility helper for code that already has
    a node table.

    Node table must contain:
        longitude
        latitude
    """

    nodes = nodes.copy()

    tile_cache = {}

    for _, row in nodes.iterrows():

        elevation = get_elevation(
            float(row["latitude"]),
            float(row["longitude"]),
            Path(dem_dir),
            tile_cache,
        )

        if "elevation" not in nodes.columns:
            nodes["elevation"] = np.nan

        nodes.loc[
            _,
            "elevation"
        ] = elevation

    return nodes