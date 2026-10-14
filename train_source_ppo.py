import sys
import gymnasium as gym

# Select Chennai/Velachery before importing rl.py
sys.argv = ["train_source_ppo.py", "velachery"]

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from rl import DrainageEnvironment
from mumbai_rl import MumbaiDrainageEnvironment


class CombinedSourceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        self.chennai = DrainageEnvironment()
        self.mumbai = MumbaiDrainageEnvironment(
            storm_condition="HEAVY"
        )

        self.use_chennai = True
        self.env = self.chennai

        self.observation_space = self.chennai.observation_space
        self.action_space = self.chennai.action_space

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)

        self.use_chennai = not self.use_chennai

        if self.use_chennai:
            self.env = self.chennai
        else:
            self.env = self.mumbai

        result = self.env.reset()

        if isinstance(result, tuple):
            return result

        return result, {}

    def step(self, action):
        result = self.env.step(action)

        if len(result) == 4:
            obs, reward, done, info = result
            return obs, reward, done, False, info

        return result

    def close(self):
        self.chennai.close()
        self.mumbai.close()


def make_env():
    return CombinedSourceEnv()


print("=" * 60)
print("CHENNAI + MUMBAI SOURCE PPO")
print("=" * 60)

env = DummyVecEnv([make_env])

print()
print("Environment created successfully.")
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
print()

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
print("Starting source-city training...")

model.learn(total_timesteps=576)

model.save("ppo_chennai_mumbai_source")

env.close()

print()
print("=" * 60)
print("SOURCE PPO TRAINING COMPLETE")
print("=" * 60)
print("Saved as: ppo_chennai_mumbai_source.zip")