import sys
sys.argv.append("vellore")

from stable_baselines3 import PPO
from rl import DrainageEnvironment


MODEL_FILE = "ppo_vellore_flood_control"


def run_controller(env, model=None):
    state, info = env.reset()

    total_reward = 0.0
    maximum_depth = 0.0
    maximum_flooding = 0.0
    total_flooding_exposure = 0.0

    pump_on_steps = 0
    steps = 0

    try:
        while True:

            # Pump OFF baseline
            if model is None:
                action = 0

            # PPO controller
            else:
                action, _ = model.predict(
                    state,
                    deterministic=True
                )
                action = int(action)

            state, reward, terminated, truncated, info = env.step(action)

            depth = float(info["depth"])
            flooding = float(info["flooding"])

            total_reward += float(reward)

            maximum_depth = max(
                maximum_depth,
                depth
            )

            maximum_flooding = max(
                maximum_flooding,
                flooding
            )

            # Flooding is a flow rate.
            # Multiply by the actual RL-action duration.
            total_flooding_exposure += (
                flooding * 281.0
            )

            if action == 1:
                pump_on_steps += 1

            steps += 1

            if terminated or truncated:
                break

    finally:
        env.close()

    return {
        "steps": steps,
        "maximum_depth": maximum_depth,
        "maximum_flooding": maximum_flooding,
        "total_flooding_exposure": total_flooding_exposure,
        "total_reward": total_reward,
        "pump_on_steps": pump_on_steps
    }


print()
print("=" * 70)
print("VELLORE FLOOD CONTROL COMPARISON")
print("=" * 70)

# ---------------------------------------------------------
# PUMP OFF BASELINE
# ---------------------------------------------------------

print()
print("=" * 70)
print("1. PUMP OFF BASELINE")
print("=" * 70)

baseline_env = DrainageEnvironment()

baseline = run_controller(
    baseline_env,
    model=None
)

print()
print("Pump OFF results:")
print("Steps:", baseline["steps"])
print(
    f"Maximum depth: "
    f"{baseline['maximum_depth']:.6f} m"
)
print(
    f"Maximum flooding flow: "
    f"{baseline['maximum_flooding']:.6f} CMS"
)
print(
    f"Flooding exposure: "
    f"{baseline['total_flooding_exposure']:.6f} m3"
)
print(
    f"Total reward: "
    f"{baseline['total_reward']:.3f}"
)


# ---------------------------------------------------------
# PPO
# ---------------------------------------------------------

print()
print("=" * 70)
print("2. VELLORE PPO")
print("=" * 70)

print()
print("Loading:", MODEL_FILE)

model = PPO.load(MODEL_FILE)

ppo_env = DrainageEnvironment()

ppo = run_controller(
    ppo_env,
    model=model
)

print()
print("PPO results:")
print("Steps:", ppo["steps"])
print(
    f"Maximum depth: "
    f"{ppo['maximum_depth']:.6f} m"
)
print(
    f"Maximum flooding flow: "
    f"{ppo['maximum_flooding']:.6f} CMS"
)
print(
    f"Flooding exposure: "
    f"{ppo['total_flooding_exposure']:.6f} m3"
)
print(
    f"Total reward: "
    f"{ppo['total_reward']:.3f}"
)
print(
    "Pump ON steps:",
    ppo["pump_on_steps"]
)

if ppo["steps"] > 0:
    pump_percentage = (
        ppo["pump_on_steps"]
        / ppo["steps"]
    ) * 100

    print(
        f"Pump operated for "
        f"{pump_percentage:.2f}% of simulation"
    )


# ---------------------------------------------------------
# COMPARISON
# ---------------------------------------------------------

print()
print("=" * 70)
print("3. COMPARISON")
print("=" * 70)

depth_reduction = 0.0

if baseline["maximum_depth"] > 0:
    depth_reduction = (
        (baseline["maximum_depth"]
         - ppo["maximum_depth"])
        / baseline["maximum_depth"]
    ) * 100

flood_reduction = 0.0

if baseline["total_flooding_exposure"] > 0:
    flood_reduction = (
        (baseline["total_flooding_exposure"]
         - ppo["total_flooding_exposure"])
        / baseline["total_flooding_exposure"]
    ) * 100

print()
print(
    f"Baseline maximum depth : "
    f"{baseline['maximum_depth']:.6f} m"
)

print(
    f"PPO maximum depth      : "
    f"{ppo['maximum_depth']:.6f} m"
)

print(
    f"Maximum depth reduction: "
    f"{depth_reduction:.2f}%"
)

print()

print(
    f"Baseline flooding exposure : "
    f"{baseline['total_flooding_exposure']:.6f} m3"
)

print(
    f"PPO flooding exposure      : "
    f"{ppo['total_flooding_exposure']:.6f} m3"
)

print(
    f"Flooding exposure reduction: "
    f"{flood_reduction:.2f}%"
)

print()
print("=" * 70)
print("COMPARISON COMPLETE")
print("=" * 70)