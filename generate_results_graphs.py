import matplotlib.pyplot as plt


controllers = [
    "Pump OFF",
    "Zero-Shot\nChennai+Mumbai PPO",
    "Rule-Based",
    "Vellore PPO"
]

flood_exposure = [
    455.020845,
    455.020845,
    33.176554,
    33.176554
]

pump_usage = [
    0.00,
    0.00,
    1.74,
    1.74
]

rewards = [
    -2381.629,
    -2381.629,
    1919.742,
    1919.742
]

max_depth = [
    4.0,
    4.0,
    4.0,
    4.0
]


# ---------------------------------------------------------
# 1. Flood Exposure
# ---------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.bar(controllers, flood_exposure)

plt.ylabel("Integrated Flooding Exposure (m³)")
plt.title("Flood Exposure Comparison")

plt.tight_layout()
plt.savefig("flood_exposure_comparison.png", dpi=300)
plt.show()


# ---------------------------------------------------------
# 2. Pump Usage
# ---------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.bar(controllers, pump_usage)

plt.ylabel("Pump Usage (%)")
plt.title("Pump Usage Comparison")

plt.tight_layout()
plt.savefig("pump_usage_comparison.png", dpi=300)
plt.show()


# ---------------------------------------------------------
# 3. Reward
# ---------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.bar(controllers, rewards)

plt.ylabel("Total Reward")
plt.title("Controller Reward Comparison")

plt.tight_layout()
plt.savefig("reward_comparison.png", dpi=300)
plt.show()


# ---------------------------------------------------------
# 4. Maximum Depth
# ---------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.bar(controllers, max_depth)

plt.ylabel("Maximum Depth (m)")
plt.title("Maximum Simulated Depth")

plt.tight_layout()
plt.savefig("maximum_depth_comparison.png", dpi=300)
plt.show()


print("=" * 60)
print("RESULT GRAPHS GENERATED")
print("=" * 60)

print("1. flood_exposure_comparison.png")
print("2. pump_usage_comparison.png")
print("3. reward_comparison.png")
print("4. maximum_depth_comparison.png")

print("=" * 60)