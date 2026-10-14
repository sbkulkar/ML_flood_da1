from stable_baselines3 import PPO
from rl import DrainageEnvironment


# ============================================================
# LOAD TRAINED PPO MODEL
# ============================================================

MODEL_FILE = "ppo_velachery_flood_control"

model = PPO.load(MODEL_FILE)


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

env = DrainageEnvironment()

state, info = env.reset()


# ============================================================
# RUN PPO
# ============================================================

print()
print("=" * 60)
print("PPO FLOOD CONTROL EVALUATION")
print("=" * 60)

total_reward = 0.0
maximum_depth = 0.0
total_flooding = 0.0

pump_on_steps = 0
pump_off_steps = 0

steps = 0


try:

    while True:

        # PPO chooses action
        action, _ = model.predict(
            state,
            deterministic=True
        )

        state, reward, terminated, truncated, info = (
            env.step(action)
        )

        total_reward += reward

        depth = float(info["depth"])
        flooding = float(info["flooding"])

        maximum_depth = max(
            maximum_depth,
            depth
        )

        total_flooding += flooding

        # Count pump usage
        if int(action) == 1:
            pump_on_steps += 1
        else:
            pump_off_steps += 1

        steps += 1

        # Print every 500 steps
        if steps % 500 == 0:

            print(
                f"Step: {steps:4d} | "
                f"Action: {int(action)} | "
                f"Depth: {depth:.3f} | "
                f"Flooding: {flooding:.3f}"
            )

        if terminated or truncated:
            break


finally:

    env.close()


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("PPO RESULTS")
print("=" * 60)

print()
print("Steps:", steps)

print(
    f"Maximum depth: "
    f"{maximum_depth:.3f}"
)

print(
    f"Total flooding: "
    f"{total_flooding:.3f}"
)

print(
    f"Total reward: "
    f"{total_reward:.3f}"
)

print()
print(
    "Pump ON steps:",
    pump_on_steps
)

print(
    "Pump OFF steps:",
    pump_off_steps
)


# ============================================================
# PUMP USAGE
# ============================================================

if steps > 0:

    pump_percentage = (
        pump_on_steps / steps
    ) * 100

    print()
    print(
        f"Pump operated for "
        f"{pump_percentage:.2f}% "
        f"of simulation"
    )


# ============================================================
# FLOOD REDUCTION
# ============================================================

baseline_flooding = 1914.280

if baseline_flooding > 0:

    flood_reduction = (
        (baseline_flooding - total_flooding)
        / baseline_flooding
    ) * 100

    print()
    print(
        f"Flood reduction compared "
        f"with baseline: "
        f"{flood_reduction:.2f}%"
    )


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 60)
print("PPO EVALUATION COMPLETE")
print("=" * 60)