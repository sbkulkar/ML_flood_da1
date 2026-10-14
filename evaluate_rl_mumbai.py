import csv
from pathlib import Path

from stable_baselines3 import PPO

from mumbai_rl import (
    MumbaiDrainageEnvironment,
    STORM_CONDITIONS,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILE = Path(
    "models/ppo_mumbai_hindmata_flood_control.zip"
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = RESULTS_DIR / "mumbai_hindmata_ppo_evaluation.csv"

STORMS = [
    "MODERATE",
    "HEAVY",
    "EXTREME",
]


# ============================================================
# LOAD PPO MODEL
# ============================================================

def load_model():

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"\nPPO model not found:\n{MODEL_FILE}\n"
            "Train it first with train_rl_mumbai.py"
        )

    print()
    print("Loading PPO model:", MODEL_FILE)

    model = PPO.load(str(MODEL_FILE))

    print("PPO model loaded.")

    return model


# ============================================================
# BASELINE
# ============================================================

def run_baseline(storm_condition):

    env = MumbaiDrainageEnvironment(
        storm_condition=storm_condition
    )

    state, info = env.reset(
        seed=42,
        options={
            "storm_condition": storm_condition
        }
    )

    total_flooding_volume = 0.0
    maximum_depth = 0.0
    maximum_node_flooding = 0.0
    steps = 0

    try:

        while True:

            state, reward, terminated, truncated, info = env.step(0)

            # Hindmata hotspot flooding volume (m3) = flooding rate (m3/s) * seconds per step
            total_flooding_volume += (
                float(info["flooding"])
                * float(info["seconds_per_step"])
            )

            # Hindmata hotspot junction depth
            maximum_depth = max(
                maximum_depth,
                float(info["depth"])
            )

            maximum_node_flooding = max(
                maximum_node_flooding,
                float(info["flooding"])
            )

            steps += 1

            if terminated or truncated:
                break

    finally:
        env.close()

    return {
        "controller": "OFF",
        "total_flooding_volume": total_flooding_volume,
        "max_network_depth": maximum_depth,
        "maximum_node_flooding": maximum_node_flooding,
        "pump_on_percentage": 0.0,
        "steps": steps,
        "total_reward": 0.0,
    }


# ============================================================
# PPO
# ============================================================

def run_ppo(storm_condition, model):

    env = MumbaiDrainageEnvironment(
        storm_condition=storm_condition
    )

    state, info = env.reset(
        seed=42,
        options={
            "storm_condition": storm_condition
        }
    )

    total_reward = 0.0
    total_flooding_volume = 0.0
    maximum_depth = 0.0
    maximum_node_flooding = 0.0

    pump_on_steps = 0
    steps = 0

    try:

        while True:

            action, _ = model.predict(
                state,
                deterministic=True
            )

            action = int(action)

            (
                state,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            total_reward += float(reward)

            # Hindmata hotspot flooding volume (m3) = flooding rate (m3/s) * seconds per step
            total_flooding_volume += (
                float(info["flooding"])
                * float(info["seconds_per_step"])
            )

            # Hindmata hotspot junction depth
            maximum_depth = max(
                maximum_depth,
                float(info["depth"])
            )

            maximum_node_flooding = max(
                maximum_node_flooding,
                float(info["flooding"])
            )

            if action == 1:
                pump_on_steps += 1

            steps += 1

            if terminated or truncated:
                break

    finally:
        env.close()

    pump_percentage = (
        (pump_on_steps / steps) * 100.0
        if steps > 0
        else 0.0
    )

    return {
        "controller": "PPO",
        "total_flooding_volume": total_flooding_volume,
        "max_network_depth": maximum_depth,
        "maximum_node_flooding": maximum_node_flooding,
        "pump_on_percentage": pump_percentage,
        "steps": steps,
        "total_reward": total_reward,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("MUMBAI HINDMATA — MULTI-STORM PPO EVALUATION")
    print("=" * 70)

    print()
    print("Region :", "mumbai_hindmata")
    print("Model  :", MODEL_FILE)

    print()
    print("Storm conditions:")
    for storm in STORMS:
        print(
            f"  {storm:<8} "
            f"rainfall scale = "
            f"{STORM_CONDITIONS[storm]:.2f}x"
        )

    model = load_model()

    rows = []

    for storm_condition in STORMS:

        print()
        print("=" * 70)
        print(f"STORM: {storm_condition}")
        print(
            f"Rainfall scale: "
            f"{STORM_CONDITIONS[storm_condition]:.2f}x"
        )
        print("=" * 70)

        # ----------------------------------------------------
        # BASELINE
        # ----------------------------------------------------

        print()
        print("Running baseline (pump OFF)...")

        baseline_result = run_baseline(
            storm_condition
        )

        print(
            "  Baseline flooding volume: "
            f"{baseline_result['total_flooding_volume']:.3f}"
        )

        print(
            "  Baseline max depth: "
            f"{baseline_result['max_network_depth']:.3f}"
        )

        print(
            "  Baseline max node flooding: "
            f"{baseline_result['maximum_node_flooding']:.3f}"
        )

        # ----------------------------------------------------
        # PPO
        # ----------------------------------------------------

        print()
        print("Running PPO...")

        ppo_result = run_ppo(
            storm_condition,
            model
        )

        print(
            "  PPO flooding volume: "
            f"{ppo_result['total_flooding_volume']:.3f}"
        )

        print(
            "  PPO max depth: "
            f"{ppo_result['max_network_depth']:.3f}"
        )

        print(
            "  PPO max node flooding: "
            f"{ppo_result['maximum_node_flooding']:.3f}"
        )

        print(
            "  PPO pump usage: "
            f"{ppo_result['pump_on_percentage']:.2f}%"
        )

        # ----------------------------------------------------
        # REDUCTION
        # ----------------------------------------------------

        baseline_flood = (
            baseline_result["total_flooding_volume"]
        )

        ppo_flood = (
            ppo_result["total_flooding_volume"]
        )

        if baseline_flood > 0:

            reduction = (
                (baseline_flood - ppo_flood)
                / baseline_flood
            ) * 100.0

            print(
                f"  Flood reduction: {reduction:.2f}%"
            )

        else:

            reduction = 0.0

            print(
                "  Flood reduction: N/A "
                "(baseline produced zero flooding)"
            )

        # ----------------------------------------------------
        # CSV ROWS
        # ----------------------------------------------------

        rows.append({
            "storm": storm_condition,
            "controller": "OFF",
            "total_flooding_volume":
                baseline_result[
                    "total_flooding_volume"
                ],
            "max_network_depth":
                baseline_result[
                    "max_network_depth"
                ],
            "maximum_node_flooding":
                baseline_result[
                    "maximum_node_flooding"
                ],
            "pump_on_percentage": 0.0,
            "steps":
                baseline_result["steps"],
            "total_reward": 0.0,
        })

        rows.append({
            "storm": storm_condition,
            "controller": "PPO",
            "total_flooding_volume":
                ppo_result[
                    "total_flooding_volume"
                ],
            "max_network_depth":
                ppo_result[
                    "max_network_depth"
                ],
            "maximum_node_flooding":
                ppo_result[
                    "maximum_node_flooding"
                ],
            "pump_on_percentage":
                ppo_result[
                    "pump_on_percentage"
                ],
            "steps":
                ppo_result["steps"],
            "total_reward":
                ppo_result["total_reward"],
        })

    # ========================================================
    # SAVE
    # ========================================================

    fieldnames = [
        "storm",
        "controller",
        "total_flooding_volume",
        "max_network_depth",
        "maximum_node_flooding",
        "pump_on_percentage",
        "steps",
        "total_reward",
    ]

    with open(
        RESULTS_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)

    print()
    print(RESULTS_FILE)

    print()
    print("Dashboard command:")
    print(
        "  python3 -m streamlit run "
        "dashboard/app.py"
    )


if __name__ == "__main__":
    main()