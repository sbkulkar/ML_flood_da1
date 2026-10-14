import sys

sys.argv = ["rule_based_vellore.py", "vellore"]

from vellore_rl import DrainageEnvironment


print("=" * 60)
print("VELLORE RULE-BASED FLOOD CONTROLLER")
print("=" * 60)

env = DrainageEnvironment()

obs, info = env.reset()

total_reward = 0.0
steps = 0
pump_on_steps = 0
max_depth = 0.0
max_flooding = 0.0

while True:

    # Observation:
    # [control_depth/10,
    #  max_network_depth/20,
    #  control_flooding/10,
    #  total_flooding/1000,
    #  pump_status]

    control_depth = float(obs[0]) * 10.0
    network_depth = float(obs[1]) * 20.0
    control_flooding = float(obs[2]) * 10.0

    # RULE:
    # Turn pump ON when control-node depth reaches 2 m.
    if control_depth >= 2.0:
        action = 1
    else:
        action = 0

    if action == 1:
        pump_on_steps += 1

    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += float(reward)
    steps += 1

    control_depth = float(obs[0]) * 10.0
    network_depth = float(obs[1]) * 20.0
    control_flooding = float(obs[2]) * 10.0

    max_depth = max(max_depth, control_depth, network_depth)
    max_flooding = max(max_flooding, control_flooding)

    if terminated or truncated:
        break

env.close()

print("\n" + "=" * 60)
print("RULE-BASED EVALUATION COMPLETE")
print("=" * 60)

print(f"Steps:             {steps}")
print(f"Maximum depth:     {max_depth:.6f} m")
print(f"Maximum flooding:  {max_flooding:.6f} CMS")
print(f"Total reward:      {total_reward:.3f}")
print(f"Pump ON steps:     {pump_on_steps}")
print(f"Pump usage:        {(pump_on_steps / steps) * 100:.2f}%")

print("=" * 60)