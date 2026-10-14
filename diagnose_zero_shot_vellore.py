import sys

sys.argv = ["diagnose_zero_shot_vellore.py", "vellore"]

from stable_baselines3 import PPO
from vellore_rl import DrainageEnvironment

MODEL_PATH = "ppo_chennai_mumbai_source.zip"

print("=" * 60)
print("ZERO-SHOT VELLORE DIAGNOSTIC")
print("=" * 60)

model = PPO.load(MODEL_PATH)
env = DrainageEnvironment()

obs, info = env.reset()

print("\nInitial observation:", obs)

for step in range(288):

    action, _ = model.predict(obs, deterministic=True)
    action = int(action)

    obs, reward, terminated, truncated, info = env.step(action)

    if step % 20 == 0 or action == 1:
        print(
            f"Step {step + 1:3d} | "
            f"Action {action} | "
            f"Obs {obs} | "
            f"Reward {reward:.3f}"
        )

    if terminated or truncated:
        break

env.close()

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)