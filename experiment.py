from rl import DrainageEnvironment


# ============================================================
# RUN ONE COMPLETE SIMULATION
# ============================================================

def run_simulation(pump_action, name):

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    env = DrainageEnvironment()

    state, info = env.reset()

    total_reward = 0.0
    maximum_depth = 0.0
    total_flooding = 0.0

    steps = 0

    try:

        while True:

            state, reward, terminated, truncated, info = (
                env.step(pump_action)
            )

            total_reward += reward

            maximum_depth = max(
                maximum_depth,
                float(info["depth"])
            )

            total_flooding += float(
                info["flooding"]
            )

            steps += 1

            if steps % 500 == 0:

                print(
                    f"Step: {steps:4d} | "
                    f"Depth: {info['depth']:.3f} | "
                    f"Flooding: {info['flooding']:.3f}"
                )

            if terminated or truncated:

                break

    finally:

        env.close()

    print()
    print("Steps:", steps)
    print(
        f"Maximum depth: {maximum_depth:.3f}"
    )
    print(
        f"Total flooding: {total_flooding:.3f}"
    )
    print(
        f"Total reward: {total_reward:.3f}"
    )

    return (
        maximum_depth,
        total_flooding
    )


# ============================================================
# BASELINE
# ============================================================

print()
print("STARTING BASELINE")
print("Pump permanently OFF")

baseline_depth, baseline_flood = (
    run_simulation(
        0,
        "BASELINE - PUMP OFF"
    )
)


# ============================================================
# PUMP CONTROL
# ============================================================

print()
print("STARTING PUMP TEST")
print("Pump permanently ON")

pump_depth, pump_flood = (
    run_simulation(
        1,
        "CONTROL - PUMP ON"
    )
)


# ============================================================
# COMPARISON
# ============================================================

print()
print("=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

print()
print(
    f"OFF maximum depth : "
    f"{baseline_depth:.3f}"
)

print(
    f"ON maximum depth  : "
    f"{pump_depth:.3f}"
)

print()
print(
    f"OFF flooding      : "
    f"{baseline_flood:.3f}"
)

print(
    f"ON flooding       : "
    f"{pump_flood:.3f}"
)


# ============================================================
# DEPTH REDUCTION
# ============================================================

if baseline_depth > 0:

    depth_reduction = (
        (baseline_depth - pump_depth)
        / baseline_depth
    ) * 100

    print()
    print(
        f"Maximum depth reduction: "
        f"{depth_reduction:.2f}%"
    )


# ============================================================
# FLOOD REDUCTION
# ============================================================

if baseline_flood > 0:

    flood_reduction = (
        (baseline_flood - pump_flood)
        / baseline_flood
    ) * 100

    print(
        f"Flooding reduction: "
        f"{flood_reduction:.2f}%"
    )

else:

    print()
    print(
        "Flooding reduction: "
        "NOT CALCULATED"
    )

    print(
        "No SWMM overflow was recorded."
    )


print()
print("=" * 60)
print("EXPERIMENT COMPLETE")
print("=" * 60)