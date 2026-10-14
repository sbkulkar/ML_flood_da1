import subprocess
import sys
from pathlib import Path

from stable_baselines3 import PPO


# ============================================================
# SETTINGS
# ============================================================

CHENNAI_REGION = "velachery"
MUMBAI_REGION = "mumbai_hindmata"

CHENNAI_STEPS = 50000
MUMBAI_STEPS = 50000

MODEL_NAME = "ppo_chennai_mumbai_combined"

TEMP_SCRIPT = Path("_combined_training_part.py")


# ============================================================
# CREATE TEMPORARY TRAINING SCRIPT
# ============================================================

TEMP_SCRIPT.write_text(
    r'''
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
''',
    encoding="utf-8"
)


# ============================================================
# STAGE 1 — CHENNAI
# ============================================================

print()
print("=" * 70)
print("COMBINED CHENNAI + MUMBAI PPO TRAINING")
print("=" * 70)

print()
print("Stage 1:")
print("Velachery / Chennai")
print(f"Timesteps: {CHENNAI_STEPS}")

result = subprocess.run(
    [
        sys.executable,
        str(TEMP_SCRIPT),
        CHENNAI_REGION,
        str(CHENNAI_STEPS),
        "new",
        MODEL_NAME
    ]
)

if result.returncode != 0:
    print()
    print("ERROR: Chennai training failed.")
    sys.exit(1)


# ============================================================
# STAGE 2 — MUMBAI
# ============================================================

print()
print("=" * 70)
print("Stage 2:")
print("Hindmata / Mumbai")
print(f"Timesteps: {MUMBAI_STEPS}")
print("=" * 70)

result = subprocess.run(
    [
        sys.executable,
        str(TEMP_SCRIPT),
        MUMBAI_REGION,
        str(MUMBAI_STEPS),
        "continue",
        MODEL_NAME
    ]
)

if result.returncode != 0:
    print()
    print("ERROR: Mumbai training failed.")
    sys.exit(1)


# ============================================================
# CLEANUP
# ============================================================

if TEMP_SCRIPT.exists():
    TEMP_SCRIPT.unlink()


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("COMBINED PPO TRAINING COMPLETE")
print("=" * 70)

print()
print("Training:")
print("  Chennai  : Velachery")
print("  Mumbai   : Hindmata")

print()
print("Total timesteps:")
print(
    CHENNAI_STEPS + MUMBAI_STEPS
)

print()
print("Saved model:")
print(
    MODEL_NAME + ".zip"
)

print()
print("Next:")
print(
    "Evaluate this model on unseen Vellore."
)

print()
print("=" * 70)