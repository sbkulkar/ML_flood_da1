# ============================================================
# config.py
# Central project configuration
# Supports multiple study regions (Chennai + Mumbai)
# ============================================================

from pathlib import Path
import sys


# ============================================================
# PROJECT
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent


# ============================================================
# BASE DATA DIRECTORIES (shared, not region-specific)
# ============================================================

DATA_DIR = PROJECT_DIR / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SWMM_DIR = DATA_DIR / "swmm"
RL_DIR = DATA_DIR / "rl"

# NOTE: NETWORK_DIR and DEM_DIR are defined further down, AFTER
# the active REGION is resolved. Each region gets its own
# subfolder (e.g. data/network/velachery/, data/network/
# mumbai_mithi/) so that nodes.csv/edges.csv and cached DEM
# tiles from different cities/regions can never collide or
# silently overwrite one another. SWMM_DIR and RL_DIR stay
# shared because their filenames already embed the region
# name (via MODEL_NAME / RL_MODEL_NAME / PPO_MODEL_NAME).


# ============================================================
# DRAINAGE NETWORK SOURCE APIS
# ============================================================
#
# Chennai has a public GCC GIS drainage layer.
# Mumbai has no equivalent public BMC API, so Mumbai regions
# fall back to OpenStreetMap's waterway/drain tags via the
# Overpass API. This is far less complete than an official
# drainage survey -- treat generated networks as an
# approximation and manually verify against known drains
# wherever possible.

DRAINAGE_API_URL = (
    "https://gisgcc.chennaicorporation.gov.in/"
    "server/rest/services/GCCDepts/"
    "GCC_COLLABORATION_LAYER/MapServer/8/query"
)

OSM_OVERPASS_URL = (
    "https://overpass-api.de/api/interpreter"
)


# ============================================================
# REGION DEFINITIONS
# ============================================================

REGIONS = {

    # --------------------------------------------------------
    # VELACHERY
    # --------------------------------------------------------

    "velachery": {

        "name": "Velachery & Pallikaranai Basin",

        "description": (
            "Critical flood hotspot draining "
            "towards Pallikaranai Marshland"
        ),

        "outfall_target": (
            "Pallikaranai Marsh / Buckingham Canal Link"
        ),

        "bbox": [
            80.205,
            12.960,
            80.238,
            12.992,
        ],

        "city": "chennai",

        "drainage_source": "gcc_api",

        "model_name":
            "velachery_flood_model",

        "rl_model_name":
            "velachery_rl_model",

        "ppo_model_name":
            "ppo_velachery_flood_control",
    },


    # --------------------------------------------------------
    # ADYAR
    # --------------------------------------------------------

    "adyar_coastal": {

        "name":
            "Adyar Coastal Basin",

        "description": (
            "Coastal drainage region around the "
            "Adyar River and southern Chennai coast"
        ),

        "outfall_target": (
            "Adyar River / Coastal Outfall"
        ),

        "bbox": [
            80.245,
            12.970,
            80.285,
            13.015,
        ],

        "city": "chennai",

        "drainage_source": "gcc_api",

        "model_name":
            "adyar_coastal_flood_model",

        "rl_model_name":
            "adyar_coastal_rl_model",

        "ppo_model_name":
            "ppo_adyar_coastal_flood_control",
    },


    # --------------------------------------------------------
    # MUMBAI -- MITHI RIVER BASIN
    # --------------------------------------------------------
    #
    # NOTE: this bbox is a starting placeholder covering the
    # Mithi River corridor (a known chronic flood zone from
    # Powai/Vihar towards Mahim Creek). Verify and replace
    # with your actual study area bounding box before relying
    # on generated results.

    "mumbai_mithi": {

        "name":
            "Mithi River Basin (Mumbai)",

        "description": (
            "Chronic urban flood corridor along the Mithi "
            "River, draining from Powai/Vihar towards "
            "Mahim Creek"
        ),

        "outfall_target": (
            "Mahim Creek / Arabian Sea"
        ),

        "bbox": [
            72.850,
            19.050,
            72.900,
            19.120,
        ],

        "city": "mumbai",

        "drainage_source": "osm_overpass",

        "model_name":
            "mumbai_mithi_flood_model",

        "rl_model_name":
            "mumbai_mithi_rl_model",

        "ppo_model_name":
            "ppo_mumbai_mithi_flood_control",
    },

    # --------------------------------------------------------
    # MUMBAI -- HINDMATA / PAREL-DADAR
    # --------------------------------------------------------
    #
    # One of Mumbai's most well-known chronic waterlogging
    # spots, in a historically low-lying reclaimed area.
    # Verify/refine this bbox against your actual study area.

    "mumbai_hindmata": {

        "name":
            "Hindmata Basin (Parel-Dadar, Mumbai)",

        "description": (
            "Chronic waterlogging hotspot in the low-lying "
            "Parel-Dadar reclaimed area near Hindmata Junction"
        ),

        "outfall_target": (
            "Mumbai Storm Water Drain Network / Arabian Sea"
        ),

        "bbox": [
            72.825,
            19.000,
            72.850,
            19.020,
        ],

        "city": "mumbai",

        "drainage_source": "osm_overpass",

        "model_name":
            "mumbai_hindmata_flood_model",

        "rl_model_name":
            "mumbai_hindmata_rl_model",

        "ppo_model_name":
            "ppo_mumbai_hindmata_flood_control",
    },


    # --------------------------------------------------------
    # VELLORE
    # --------------------------------------------------------

    "vellore": {

        "name":
            "Vellore Urban Drainage Basin",

        "description": (
            "Urban drainage study region in Vellore, "
            "Tamil Nadu"
        ),

        "outfall_target":
            "Local drainage outfall / Palar River system",

        "bbox": [
           79.100,
           12.890,
           79.170,
           12.950
        ],

        "city": "vellore",

        "drainage_source":
            "osm_overpass",

        "model_name":
            "vellore_flood_model",

        "rl_model_name":
            "vellore_rl_model",

        "ppo_model_name":
        "ppo_vellore_flood_control",

        "control_node":
        "J_0033",

         "pump_id":
          "P1",

        "outfall_id":
        "J_0003",
},
    }


# ============================================================
# SELECT REGION
# ============================================================
#
# Usage:
#
# python pipeline.py velachery
# python pipeline.py adyar_coastal
# python pipeline.py mumbai_mithi
# python pipeline.py vellore
#
# ============================================================

if len(sys.argv) < 2:

    print()
    print("=" * 70)
    print("ERROR: NO REGION SELECTED")
    print("=" * 70)

    print()
    print("Available regions:")

    for region_name in REGIONS:
        print(f"  - {region_name}")

    print()
    print("Usage:")
    print("  python pipeline.py velachery")
    print("  python pipeline.py adyar_coastal")
    print("  python pipeline.py mumbai_mithi")
    print("  python pipeline.py mumbai_hindmata")
    print("  python pipeline.py vellore")
    print()

    raise ValueError(
        "No region selected."
    )

REGION = sys.argv[1].lower()

# ============================================================
# VALIDATE REGION
# ============================================================

if REGION not in REGIONS:

    print()
    print("=" * 70)
    print("ERROR: UNKNOWN REGION")
    print("=" * 70)

    print()
    print("Requested region:")
    print(REGION)

    print()
    print("Available regions:")

    for region_name in REGIONS:
        print(
            f" - {region_name}"
        )

    print()

    raise ValueError(
        f"Unknown REGION: {REGION}"
    )


# ============================================================
# ACTIVE REGION CONFIGURATION
# ============================================================

CURRENT_REGION = REGIONS[REGION]


# ============================================================
# REGION-SPECIFIC DATA DIRECTORIES
# ============================================================
#
# nodes.csv/edges.csv and cached DEM tiles are kept in a
# per-region subfolder so that Chennai and Mumbai (or any two
# regions) can never overwrite each other's network or
# elevation data by running scripts back-to-back for
# different regions.

NETWORK_DIR = DATA_DIR / "network" / REGION
DEM_DIR = DATA_DIR / "dem" / REGION


# ============================================================
# ACTIVE STUDY AREA
# ============================================================

STUDY_AREA_NAME = (
    CURRENT_REGION["name"]
)

STUDY_AREA_DESCRIPTION = (
    CURRENT_REGION["description"]
)

OUTFALL_TARGET = (
    CURRENT_REGION["outfall_target"]
)

BBOX = (
    CURRENT_REGION["bbox"]
)

CITY = (
    CURRENT_REGION.get("city", "chennai")
)

DRAINAGE_SOURCE = (
    CURRENT_REGION.get("drainage_source", "gcc_api")
)


# ============================================================
# ACTIVE MODEL NAMES
# ============================================================

MODEL_NAME = (
    CURRENT_REGION["model_name"]
)

RL_MODEL_NAME = (
    CURRENT_REGION["rl_model_name"]
)

PPO_MODEL_NAME = (
    CURRENT_REGION["ppo_model_name"]
)


# ============================================================
# COORDINATE SYSTEMS
# ============================================================

SOURCE_CRS = "EPSG:4326"

# UTM zone depends on city:
# Chennai -> UTM 44N (EPSG:32644)
# Mumbai -> UTM 43N (EPSG:32643)
PROJECTED_CRS = (
    "EPSG:32643" if CITY == "mumbai" else "EPSG:32644"
)


# ============================================================
# NETWORK
# ============================================================

SNAP_TOLERANCE_M = 15.0


# ============================================================
# HYDRAULIC DEFAULTS
# ============================================================

DEFAULT_WIDTH_M = 1.0

DEFAULT_DEPTH_M = 1.0

DEFAULT_ROUGHNESS = 0.015


# ============================================================
# CREATE DIRECTORIES
# ============================================================

for directory in [

    DATA_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    NETWORK_DIR,
    DEM_DIR,
    SWMM_DIR,
    RL_DIR,

]:

    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# CONFIGURATION SUMMARY
# ============================================================

print()
print("=" * 70)
print("FLOOD CONTROL CONFIGURATION")
print("=" * 70)

print()
print("Active region:")
print(
    REGION
)

print()
print("City:")
print(
    CITY
)

print()
print("Study area:")
print(
    STUDY_AREA_NAME
)

print()
print("Bounding box:")
print(
    BBOX
)

print()
print("Projected CRS:")
print(
    PROJECTED_CRS
)

print()
print("SWMM model:")
print(
    MODEL_NAME
)

print()
print("RL model:")
print(
    RL_MODEL_NAME
)

print()
print("PPO model:")
print(
    PPO_MODEL_NAME
)

print()
print("=" * 70)