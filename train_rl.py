import sys

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from rl import DrainageEnvironment
from config import RL_MODEL_NAME


# ============================================================
# REGION
# ============================================================

if len(sys.argv) > 1:
    region = sys.argv[1]
else:
    region = "velachery"

print()
print("=" * 60)
print(f"REGION: {region}")
print("=" * 60)


# ============================================================
# ENVIRONMENT
# ============================================================

env = DrainageEnvironment()


# ============================================================
# CHECK ENVIRONMENT
# ============================================================

print("=" * 60)
print("CHECKING ENVIRONMENT")
print("=" * 60)

check_env(
    env,
    warn=True
)

print("Environment check passed.")


# ============================================================
# PPO MODEL
# ============================================================

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
# TRAIN
# ============================================================

print()
print("=" * 60)
print("STARTING PPO TRAINING")
print("=" * 60)

model.learn(
    total_timesteps=100000
)


# ============================================================
# SAVE
# ============================================================

if region == "velachery":
    save_name = "ppo_velachery_flood_control"

elif region == "adyar_coastal":
    save_name = "ppo_adyar_coastal_flood_control"

else:
    save_name = f"ppo_{region}_flood_control"


model.save(save_name)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print()
print("Region:")
print(region)

print()
print("Saved:")
print(f"{save_name}.zip")

env.close()