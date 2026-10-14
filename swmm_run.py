"""
============================================================
SWMM SIMULATION
============================================================

Purpose:
    Run the validated SWMM model and save simulation outputs.

Pipeline position:

    API
      ↓
    Clean drainage
      ↓
    Elevation
      ↓
    Network
      ↓
    SWMM .inp
      ↓
    THIS FILE
      ↓
    RL ENVIRONMENT
"""

from pathlib import Path

from pyswmm import Simulation
from config import REGION


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

SWMM_DIR = PROJECT_DIR / "data" / "swmm"

INPUT_FILE = SWMM_DIR / f"{REGION}_flood_model.inp"

REPORT_FILE = SWMM_DIR / f"{REGION}_flood_model.rpt"

OUTPUT_FILE = SWMM_DIR / f"{REGION}_flood_model.out"


# ============================================================
# MAIN
# ============================================================

def run_swmm():

    print()
    print("#" * 70)
    print("# SWMM SIMULATION")
    print("#" * 70)

    # --------------------------------------------------------
    # CHECK INPUT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 1: CHECK SWMM INPUT")
    print("=" * 70)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"SWMM input file not found:\n{INPUT_FILE}"
        )

    print("SWMM input:")
    print(INPUT_FILE)

    # --------------------------------------------------------
    # REMOVE OLD OUTPUTS
    # --------------------------------------------------------

    for file in [REPORT_FILE, OUTPUT_FILE]:

        if file.exists():

            file.unlink()

    # --------------------------------------------------------
    # RUN SWMM
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 2: RUN SWMM")
    print("=" * 70)

    print()
    print("Starting SWMM simulation...")

    try:

        with Simulation(
            str(INPUT_FILE)
        ) as sim:

            step_count = 0

            for _ in sim:

                step_count += 1

                if step_count % 100 == 0:

                    print(
                        f"Simulation steps completed: "
                        f"{step_count}"
                    )

    except Exception as e:

        print()
        print("SWMM SIMULATION FAILED")
        print()
        print(str(e))

        raise

    # --------------------------------------------------------
    # CHECK OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 3: CHECK OUTPUT")
    print("=" * 70)

    print()

    if REPORT_FILE.exists():

        print(
            "Report file created:"
        )

        print(
            REPORT_FILE
        )

    else:

        print(
            "WARNING: Report file was not found."
        )

    if OUTPUT_FILE.exists():

        print(
            "Output file created:"
        )

        print(
            OUTPUT_FILE
        )

    else:

        print(
            "WARNING: Output file was not found."
        )

    # --------------------------------------------------------
    # FINISHED
    # --------------------------------------------------------

    print()
    print("#" * 70)
    print("# SWMM SIMULATION COMPLETE")
    print("#" * 70)

    print()
    print(
        f"Simulation steps: {step_count}"
    )

    print()
    print(
        "Next stage:"
    )

    print(
        "Read SWMM results and build the RL environment."
    )

    return True


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_swmm()