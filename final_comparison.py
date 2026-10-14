print("=" * 80)
print("FINAL VELLORE FLOOD CONTROL COMPARISON")
print("=" * 80)

controllers = [
    {
        "name": "Pump OFF",
        "depth": 4.000,
        "flooding": 0.034436,
        "reward": -2381.629,
        "pump_usage": 0.00,
        "exposure": 455.020845,
    },
    {
        "name": "Zero-Shot Chennai+Mumbai PPO",
        "depth": 4.000,
        "flooding": 0.034436,
        "reward": -2381.629,
        "pump_usage": 0.00,
        "exposure": 455.020845,
    },
    {
        "name": "Rule-Based",
        "depth": 4.000,
        "flooding": 0.034436,
        "reward": 1919.742,
        "pump_usage": 1.74,
        "exposure": 33.176554,
    },
    {
        "name": "Vellore PPO",
        "depth": 4.000,
        "flooding": 0.034436,
        "reward": 1919.742,
        "pump_usage": 1.74,
        "exposure": 33.176554,
    },
]

baseline = controllers[0]["exposure"]

print()
print(
    f"{'Controller':<30}"
    f"{'Max Depth':>12}"
    f"{'Flooding':>14}"
    f"{'Reward':>14}"
    f"{'Pump %':>10}"
    f"{'Exposure':>16}"
)

print("-" * 96)

for c in controllers:

    reduction = ((baseline - c["exposure"]) / baseline) * 100

    print(
        f"{c['name']:<30}"
        f"{c['depth']:>12.3f}"
        f"{c['flooding']:>14.6f}"
        f"{c['reward']:>14.3f}"
        f"{c['pump_usage']:>9.2f}%"
        f"{c['exposure']:>16.3f}"
    )

    print(
        f"    Flood exposure reduction: {reduction:.2f}%"
    )

print()
print("=" * 80)
print("KEY FINDINGS")
print("=" * 80)

print("1. Pump-OFF is the uncontrolled baseline.")
print("2. Zero-shot Chennai+Mumbai PPO did not activate the Vellore pump.")
print("3. Zero-shot transfer therefore achieved 0% flood-exposure reduction.")
print("4. The Rule-Based controller reduced flood exposure by 92.71%.")
print("5. The Vellore-specific PPO reduced flood exposure by 92.71%.")
print("6. Both Rule-Based and Vellore PPO used the pump for only 1.74% of steps.")
print("7. Maximum simulated depth remained 4.0 m in all four evaluations.")
print()
print("NOTE:")
print("Flood exposure here is the integrated control-node flooding flow")
print("over the RL decision intervals. It is not SWMM total flooded-node volume.")
print("=" * 80)