
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

region = sys.argv[1]
timesteps = int(sys.argv[2])
mode = sys.argv[3]
model_path = sys.argv[4]

# Select region BEFORE importing rl
sys.argv = ["train_part.py", region]

from rl import DrainageEnvironment

print()
print("=" * 70)
print("TRAINING REGION:", region)
print("=" * 70)

env = DrainageEnvironment()

print()
print("Checking environment...")

check_env(env, warn=True)

print("Environment check passed.")

# ------------------------------------------------------------
# FIRST TRAINING STAGE
# ------------------------------------------------------------

if mode == "new":

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

    print()
    print("=" * 70)
    print("STARTING NEW PPO TRAINING")
    print("=" * 70)

    model.learn(
        total_timesteps=timesteps
    )

# ------------------------------------------------------------
# CONTINUE EXISTING PPO
# ------------------------------------------------------------

else:

    print()
    print("Loading previous PPO:")
    print(model_path)

    model = PPO.load(
        model_path,
        env=env
    )

    print()
    print("=" * 70)
    print("CONTINUING PPO TRAINING")
    print("=" * 70)

    model.learn(
        total_timesteps=timesteps,
        reset_num_timesteps=False
    )

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

model.save(model_path)

print()
print("=" * 70)
print("TRAINING PART COMPLETE")
print("=" * 70)

print()
print("Region:", region)
print("Timesteps:", timesteps)
print("Model:", model_path)

env.close()
