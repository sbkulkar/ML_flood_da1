from rl import DrainageEnvironment


def run_test(action, name):

    env = DrainageEnvironment()

    state, info = env.reset()

    total_flooding = 0.0
    max_depth = 0.0
    steps = 0

    while True:

        state, reward, terminated, truncated, info = env.step(action)

        total_flooding += info["total_flooding"]

        max_depth = max(
            max_depth,
            info["depth"]
        )

        steps += 1

        if terminated or truncated:
            break

    env.close()

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    print("Steps:", steps)
    print("Maximum depth:", max_depth)
    print("Total flooding:", total_flooding)

    return total_flooding, max_depth


# ============================================================
# BASELINE
# ============================================================

print()
print("RUNNING BASELINE")
print("Pump permanently OFF")

baseline_flood, baseline_depth = run_test(
    0,
    "BASELINE - PUMP OFF"
)


# ============================================================
# PUMP CONTROL
# ============================================================

print()
print("RUNNING PUMP CONTROL")
print("Pump permanently ON")

pump_flood, pump_depth = run_test(
    1,
    "CONTROL - PUMP ON"
)


# ============================================================
# COMPARISON
# ============================================================

print()
print("=" * 60)
print("FLOOD CONTROL COMPARISON")
print("=" * 60)

print()
print(
    "Baseline flooding:",
    baseline_flood
)

print(
    "Pump flooding:",
    pump_flood
)

print()
print(
    "Baseline maximum depth:",
    baseline_depth
)

print(
    "Pump maximum depth:",
    pump_depth
)


# ============================================================
# FLOOD REDUCTION
# ============================================================

if baseline_flood > 0:

    reduction = (
        (baseline_flood - pump_flood)
        / baseline_flood
    ) * 100

    print()
    print(
        f"Flood reduction: {reduction:.2f}%"
    )

else:

    print()
    print(
        "Flood reduction cannot be calculated."
    )

    print(
        "The test scenario produced no flooding."
    )