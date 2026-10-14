from stable_baselines3 import PPO

from rl import DrainageEnvironment, RL_INP

from config import PPO_MODEL_NAME

import pandas as pd
from pathlib import Path
# ============================================================
# LOAD REGION-SPECIFIC PPO MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING PPO MODEL")
print("=" * 70)

print()
print("PPO model:")
print(PPO_MODEL_NAME)

model = PPO.load(
    PPO_MODEL_NAME
)


# ============================================================
# STORM SCENARIOS
# ============================================================

STORMS = {

    "MODERATE": [
        0.00, 0.05, 0.10, 0.25,
        0.50, 0.75, 1.00, 1.00,
        0.75, 0.50, 0.25, 0.10,
        0.05, 0.00
    ],

    "HEAVY": [
        0.00, 0.10, 0.25, 0.50,
        1.00, 1.50, 2.00, 2.00,
        1.50, 1.00, 0.50, 0.25,
        0.10, 0.00
    ],

    "EXTREME": [
        0.00, 0.20, 0.50, 1.00,
        2.00, 3.00, 4.00, 4.00,
        3.00, 2.00, 1.00, 0.50,
        0.20, 0.00
    ]
}


# ============================================================
# RUN SCENARIO
# ============================================================

def run_scenario(
    storm_values,
    use_ppo
):

    # --------------------------------------------------------
    # CREATE ENVIRONMENT
    # --------------------------------------------------------

    env = DrainageEnvironment(
        storm_values=storm_values
    )

    state, info = env.reset()

    maximum_depth = 0.0

    total_flooding = 0.0

    pump_on = 0

    steps = 0


    try:

        while True:

            # ------------------------------------------------
            # SELECT ACTION
            # ------------------------------------------------

            if use_ppo:

                action, _ = model.predict(
                    state,
                    deterministic=True
                )

            else:

                # Baseline:
                # Pump remains OFF

                action = 0


            # ------------------------------------------------
            # STEP ENVIRONMENT
            # ------------------------------------------------

            (
                state,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(
                action
            )


            # ------------------------------------------------
            # COLLECT RESULTS
            # ------------------------------------------------

            maximum_depth = max(
                maximum_depth,
                float(
                    info["depth"]
                )
            )


            total_flooding += float(
                info["flooding"]
            )


            # ------------------------------------------------
            # COUNT PUMP OPERATION
            # ------------------------------------------------

            if int(action) == 1:

                pump_on += 1


            steps += 1


            # ------------------------------------------------
            # CHECK END
            # ------------------------------------------------

            if (
                terminated
                or truncated
            ):

                break


    finally:

        env.close()


    # --------------------------------------------------------
    # PUMP USAGE
    # --------------------------------------------------------

    pump_usage = (
        pump_on / steps * 100.0
        if steps > 0
        else 0.0
    )


    return (
        maximum_depth,
        total_flooding,
        pump_usage
    )


# ============================================================
# MAIN
# ============================================================

print()

print("=" * 70)

print("MULTI-STORM PPO FLOOD CONTROL TEST")

print("=" * 70)


results = []


# ============================================================
# RUN ALL STORMS
# ============================================================

for storm_name, storm_values in STORMS.items():

    print()

    print("=" * 70)

    print(
        f"STORM: {storm_name}"
    )

    print("=" * 70)


    # ========================================================
    # BASELINE
    # ========================================================

    print()

    print(
        "Running NO-PUMP baseline..."
    )


    (
        baseline_depth,
        baseline_flood,
        _
    ) = run_scenario(
        storm_values,
        False
    )


    # ========================================================
    # PPO
    # ========================================================

    print()

    print(
        "Running PPO controller..."
    )


    (
        ppo_depth,
        ppo_flood,
        pump_usage
    ) = run_scenario(
        storm_values,
        True
    )


    # ========================================================
    # FLOOD REDUCTION
    # ========================================================

    if baseline_flood > 0:

        reduction = (

            (
                baseline_flood
                - ppo_flood
            )

            / baseline_flood

        ) * 100.0

    else:

        reduction = 0.0


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()

    print(
        f"Baseline max depth : "
        f"{baseline_depth:.3f}"
    )

    print(
        f"Baseline flooding  : "
        f"{baseline_flood:.3f}"
    )

    print(
        f"PPO max depth      : "
        f"{ppo_depth:.3f}"
    )

    print(
        f"PPO flooding       : "
        f"{ppo_flood:.3f}"
    )

    print(
        f"Flood reduction    : "
        f"{reduction:.2f}%"
    )

    print(
        f"Pump usage         : "
        f"{pump_usage:.2f}%"
    )


    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    results.append(
        (
            storm_name,
            baseline_flood,
            ppo_flood,
            reduction,
            ppo_depth,
            pump_usage
        )
    )


# ============================================================
# FINAL TABLE
# ============================================================

print()

print("=" * 100)

print("FINAL MULTI-STORM RESULTS")

print("=" * 100)

print()


print(
    f"{'Storm':<12}"
    f"{'Base Flood':>15}"
    f"{'PPO Flood':>15}"
    f"{'Reduction':>15}"
    f"{'PPO Depth':>15}"
    f"{'Pump Use':>15}"
)


print("-" * 100)


for result in results:

    (
        name,
        base_flood,
        ppo_flood,
        reduction,
        ppo_depth,
        pump_usage
    ) = result


    print(
        f"{name:<12}"
        f"{base_flood:>15.3f}"
        f"{ppo_flood:>15.3f}"
        f"{reduction:>14.2f}%"
        f"{ppo_depth:>15.3f}"
        f"{pump_usage:>14.2f}%"
    )


# ============================================================
# COMPLETE
# ============================================================

print()

print("=" * 100)

print("MULTI-STORM TEST COMPLETE")

print("=" * 100)

# ============================================================
# SAVE RESULTS FOR STREAMLIT DASHBOARD
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

results_df = pd.DataFrame(
    results,
    columns=[
        "Storm",
        "Baseline Flood",
        "PPO Flood",
        "Reduction",
        "PPO Depth",
        "Pump Use"
    ]
)

output_file = (
    RESULTS_DIR / "multi_storm_results.csv"
)

results_df.to_csv(
    output_file,
    index=False
)

print()
print("Results saved for Streamlit:")
print(output_file)