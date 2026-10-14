import os

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from mumbai_rl import (
    MumbaiDrainageEnvironment,
    STORM_CONDITIONS,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "ppo_mumbai_hindmata_flood_control"

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    MODEL_NAME
)

TOTAL_TIMESTEPS = 20000

STORM_SEQUENCE = [
    "MODERATE",
    "HEAVY",
    "EXTREME",
]


# ============================================================
# MULTI-STORM TRAINING ENVIRONMENT
# ============================================================

class MumbaiMultiStormTrainingEnv(gym.Wrapper):
    """
    Training wrapper that cycles PPO through all three
    Mumbai Hindmata rainfall conditions.

    The underlying Mumbai SWMM rainfall pattern remains the
    same. mumbai_rl.py scales the actual MUMBAI_RAIN series
    for each condition.
    """

    def __init__(self):

        env = MumbaiDrainageEnvironment(
            storm_condition="MODERATE"
        )

        super().__init__(env)

        self.storm_sequence = STORM_SEQUENCE
        self.episode_number = 0

    def reset(self, *, seed=None, options=None):

        storm_condition = self.storm_sequence[
            self.episode_number % len(self.storm_sequence)
        ]

        self.episode_number += 1

        if options is None:
            options = {}

        options = dict(options)

        # Training condition is controlled here.
        options["storm_condition"] = storm_condition

        print(
            f"\nTraining episode "
            f"{self.episode_number}: "
            f"{storm_condition}"
        )

        return self.env.reset(
            seed=seed,
            options=options
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("PPO TRAINING - MUMBAI HINDMATA")
    print("=" * 70)

    print("\nTraining storm conditions:")

    for condition in STORM_SEQUENCE:

        print(
            f"  {condition}: "
            f"{STORM_CONDITIONS[condition]:.2f}x "
            f"actual Mumbai rainfall"
        )

    # --------------------------------------------------------
    # Create multi-storm environment
    # --------------------------------------------------------

    print("\nCreating Mumbai multi-storm environment...")

    env = MumbaiMultiStormTrainingEnv()

    # --------------------------------------------------------
    # Check Gymnasium environment
    # --------------------------------------------------------

    print("\nChecking environment...")

    check_env(
        env,
        warn=True
    )

    print(
        "\nEnvironment check completed."
    )

    # --------------------------------------------------------
    # Create model directory
    # --------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    # ========================================================
    # CREATE PPO MODEL
    # ========================================================

    print("\nCreating PPO model...")

    model = PPO(
        policy="MlpPolicy",
        env=env,

        learning_rate=5e-4,

        n_steps=192,

        batch_size=64,

        n_epochs=10,

        gamma=0.99,

        gae_lambda=0.95,

        clip_range=0.2,

        ent_coef=0.03,

        verbose=1,
    )

    # ========================================================
    # TRAIN
    # ========================================================

    print("\n" + "=" * 70)
    print("STARTING MULTI-STORM PPO TRAINING")
    print("=" * 70)

    print(
        f"\nTotal timesteps: "
        f"{TOTAL_TIMESTEPS:,}"
    )

    print(
        "\nStorm sequence:"
        "\n  MODERATE"
        "\n  HEAVY"
        "\n  EXTREME"
        "\n  ...repeating"
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model.save(
        MODEL_PATH
    )

    print("\n" + "=" * 70)
    print("PPO TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"\nModel saved to:"
        f"\n{MODEL_PATH}.zip"
    )

    # --------------------------------------------------------
    # Close environment
    # --------------------------------------------------------

    env.close()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()