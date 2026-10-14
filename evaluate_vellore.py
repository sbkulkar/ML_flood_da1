import sys

# Force Vellore before importing the environment
sys.argv = ["evaluate_vellore.py", "vellore"]

from stable_baselines3 import PPO
from vellore_rl import DrainageEnvironment


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "ppo_vellore_flood_control"

TOTAL_STEPS = 288

# Measured from the Vellore SWMM/RL setup:
# one RL action advances approximately 281 seconds
SECONDS_PER_STEP = 281.0


# ============================================================
# RUN EVALUATION
# ============================================================

def run_episode(env, model=None, pump_off=False):

    state, _ = env.reset()

    total_reward = 0.0
    total_flood_exposure = 0.0

    max_depth = 0.0
    max_flooding = 0.0

    pump_on_steps = 0
    steps = 0

    for _ in range(TOTAL_STEPS):

        if pump_off:
            action = 0
        else:
            action, _ = model.predict(
                state,
                deterministic=True
            )

            action = int(action)

        state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        flooding = float(info["flooding"])
        depth = float(info["depth"])

        total_flood_exposure += (
            flooding * SECONDS_PER_STEP
        )

        max_depth = max(max_depth, depth)
        max_flooding = max(max_flooding, flooding)

        if action == 1:
            pump_on_steps += 1

        steps += 1

        if terminated or truncated:
            break

    return {
        "steps": steps,
        "max_depth": max_depth,
        "max_flooding": max_flooding,
        "flood_exposure": total_flood_exposure,
        "reward": total_reward,
        "pump_on_steps": pump_on_steps
    }


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 60)
print("VELLORE PPO EVALUATION")
print("=" * 60)

print()
print("Loading PPO model...")

model = PPO.load(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

env = DrainageEnvironment()


# ============================================================
# PUMP OFF BASELINE
# ============================================================

print()
print("=" * 60)
print("RUNNING PUMP-OFF BASELINE")
print("=" * 60)

baseline = run_episode(
    env,
    pump_off=True
)


# ============================================================
# PPO
# ============================================================

print()
print("=" * 60)
print("RUNNING VELLORE PPO")
print("=" * 60)

ppo_result = run_episode(
    env,
    model=model,
    pump_off=False
)


# ============================================================
# CALCULATE IMPROVEMENT
# ============================================================

if baseline["flood_exposure"] > 0:

    flood_reduction = (
        (baseline["flood_exposure"]
         - ppo_result["flood_exposure"])
        / baseline["flood_exposure"]
    ) * 100

else:

    flood_reduction = 0.0


if baseline["max_depth"] > 0:

    depth_reduction = (
        (baseline["max_depth"]
         - ppo_result["max_depth"])
        / baseline["max_depth"]
    ) * 100

else:

    depth_reduction = 0.0


pump_usage = (
    ppo_result["pump_on_steps"]
    / ppo_result["steps"]
) * 100


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("PUMP-OFF BASELINE")
print("=" * 60)

print(f"Steps:              {baseline['steps']}")
print(f"Maximum depth:      {baseline['max_depth']:.6f} m")
print(f"Maximum flooding:   {baseline['max_flooding']:.6f} CMS")
print(f"Flood exposure:     {baseline['flood_exposure']:.6f} m3")
print(f"Total reward:       {baseline['reward']:.3f}")


print()
print("=" * 60)
print("VELLORE PPO")
print("=" * 60)

print(f"Steps:              {ppo_result['steps']}")
print(f"Maximum depth:      {ppo_result['max_depth']:.6f} m")
print(f"Maximum flooding:   {ppo_result['max_flooding']:.6f} CMS")
print(f"Flood exposure:     {ppo_result['flood_exposure']:.6f} m3")
print(f"Total reward:       {ppo_result['reward']:.3f}")
print(f"Pump ON steps:      {ppo_result['pump_on_steps']}")
print(f"Pump usage:         {pump_usage:.2f}%")


print()
print("=" * 60)
print("PPO IMPROVEMENT")
print("=" * 60)

print(f"Flood exposure reduction: {flood_reduction:.2f}%")
print(f"Maximum depth reduction:  {depth_reduction:.2f}%")


# ============================================================
# CLOSE
# ============================================================

env.close()

print()
print("=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)