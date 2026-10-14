import sys
sys.argv.append("vellore")

import rl

env = rl.DrainageEnvironment()

state, info = env.reset()

total_reward = 0
steps = 0

print("Starting full episode...")

for i in range(20):

    action = 0   # Pump OFF

    state, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    steps += 1

    print(
        "Step:", steps,
        "Reward:", reward,
        "State:", state,
        "Flooding:", info["flooding"]
    )

    if terminated or truncated:
        break

print("\n================================")
print("EPISODE TEST COMPLETE")
print("Steps:", steps)
print("Total reward:", total_reward)
print("================================")

env.close()