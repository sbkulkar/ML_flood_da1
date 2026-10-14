import sys

# ============================================================
# FORCE VELLORE REGION BEFORE IMPORTING VELLORE ENVIRONMENT
# ============================================================

sys.argv = ["train_vellore_rl.py", "vellore"]


from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from vellore_rl import DrainageEnvironment


# ============================================================
# VELLORE PPO TRAINING
# ============================================================

print()
print("=" * 60)
print("VELLORE PPO FLOOD CONTROL")
print("=" * 60)

print()
print("Creating Vellore environment...")

env = DrainageEnvironment()


# ============================================================
# CHECK ENVIRONMENT
# ============================================================

print()
print("=" * 60)
print("CHECKING VELLORE ENVIRONMENT")
print("=" * 60)

check_env(env, warn=True)

print()
print("Environment check passed.")


# ============================================================
# CREATE PPO MODEL
# ============================================================

print()
print("=" * 60)
print("CREATING PPO MODEL")
print("=" * 60)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.0003,
    n_steps=288,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.02,
    verbose=1
)


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 60)
print("STARTING VELLORE PPO TRAINING")
print("=" * 60)

model.learn(
    total_timesteps=100000
)


# ============================================================
# SAVE MODEL
# ============================================================

model.save("ppo_vellore_flood_control")


print()
print("=" * 60)
print("VELLORE PPO TRAINING COMPLETE")
print("=" * 60)

print()
print("Model saved as:")
print("ppo_vellore_flood_control.zip")


# ============================================================
# CLOSE ENVIRONMENT
# ============================================================

env.close()

print()
print("Environment closed.")