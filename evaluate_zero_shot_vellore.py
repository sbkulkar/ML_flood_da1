import sys

# Force Vellore configuration
sys.argv = ["evaluate_zero_shot_vellore.py", "vellore"]

from stable_baselines3 import PPO
from vellore_rl import DrainageEnvironment


MODEL_PATH = "ppo_chennai_mumbai_source.zip"

print("=" * 60)
print("ZERO-SHOT CHENNAI + MUMBAI PPO -> VELLORE")
print("=" * 60)

print("\nLoading source-city PPO model...")
model = PPO.load(MODEL_PATH)

print("Model loaded successfully.")
print("Model:", MODEL_PATH)

print("\nCreating Vellore environment...")
env = DrainageEnvironment()

obs, info = env.reset()

print("Vellore environment created successfully.")
print("Initial observation:", obs)

total_reward = 0.0
steps = 0
pump_on_steps = 0
max_depth = 0.0
max_flooding = 0.0

while True:

    action, _ = model.predict(obs, deterministic=True)

    action = int(action)

    if action == 1:
        pump_on_steps += 1

    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += float(reward)
    steps += 1

    # Observation format:
    # [control_depth, max_network_depth,
    #  control_flooding, total_flooding, pump_status]

    control_depth = float(obs[0]) * 10.0
    network_depth = float(obs[1]) * 20.0
    control_flooding = float(obs[2]) * 10.0

    max_depth = max(max_depth, control_depth, network_depth)
    max_flooding = max(max_flooding, control_flooding)

    if terminated or truncated:
        break


env.close()

print("\n" + "=" * 60)
print("ZERO-SHOT EVALUATION COMPLETE")
print("=" * 60)

print(f"Steps:             {steps}")
print(f"Maximum depth:     {max_depth:.6f} m")
print(f"Maximum flooding:   {max_flooding:.6f} CMS")
print(f"Total reward:       {total_reward:.3f}")
print(f"Pump ON steps:      {pump_on_steps}")
print(f"Pump usage:         {(pump_on_steps / steps) * 100:.2f}%")

print("=" * 60)