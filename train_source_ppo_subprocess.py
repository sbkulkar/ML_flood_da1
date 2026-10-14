import subprocess
import json
import sys

import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


class SourceSubprocessEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self):

        self.city = "chennai"

        self.process = None

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32
        )

        self.action_space = gym.spaces.Discrete(2)

        self.start_worker(self.city)

    def start_worker(self, city):

        if self.process is not None:
            self.close_worker()

        self.city = city

        self.process = subprocess.Popen(
            [
                sys.executable,
                "source_city_worker.py",
                city
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1
        )

    def read_message(self):

        while True:

            line = self.process.stdout.readline()

            if not line:
                raise RuntimeError(
                    "Source worker stopped unexpectedly."
                )

            line = line.strip()

            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue

            if isinstance(message, dict):
                return message

    def reset(self, seed=None, options=None):

        if seed is not None:
            super().reset(seed=seed)

        # Alternate cities at every episode
        if self.city == "chennai":
            next_city = "mumbai"
        else:
            next_city = "chennai"

        self.start_worker(next_city)

        message = self.read_message()

        while message.get("type") != "reset":
            message = self.read_message()

        observation = np.array(
            message["observation"],
            dtype=np.float32
        )

        return observation, {}

    def step(self, action):

        command = {
            "command": "step",
            "action": int(action)
        }

        self.process.stdin.write(
            json.dumps(command) + "\n"
        )

        self.process.stdin.flush()

        message = self.read_message()

        while message.get("type") != "step":
            message = self.read_message()

        observation = np.array(
            message["observation"],
            dtype=np.float32
        )

        reward = float(message["reward"])

        terminated = bool(
            message["terminated"]
        )

        truncated = bool(
            message["truncated"]
        )

        return (
            observation,
            reward,
            terminated,
            truncated,
            {}
        )

    def close_worker(self):

        if self.process is None:
            return

        try:

            self.process.stdin.write(
                json.dumps(
                    {"command": "close"}
                ) + "\n"
            )

            self.process.stdin.flush()

            self.process.wait(timeout=30)

        except Exception:

            self.process.kill()

        self.process = None

    def close(self):

        self.close_worker()


def make_env():

    return SourceSubprocessEnv()


print("=" * 60)
print("CHENNAI + MUMBAI SOURCE PPO")
print("SUBPROCESS TRAINING")
print("=" * 60)


env = DummyVecEnv(
    [make_env]
)


print()
print("Environment created successfully.")
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)


model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.0003,
    n_steps=288,
    batch_size=32,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.02,
    verbose=1
)


print()
print("Starting FULL SOURCE PPO TRAINING...")
print("Total timesteps: 100000")


model.learn(
    total_timesteps=100000
)


model.save(
    "ppo_chennai_mumbai_source"
)

env.close()


print()
print("=" * 60)
print("SMALL SOURCE PPO TEST COMPLETE")
print("=" * 60)

print(
    "Saved as: "
    "ppo_chennai_mumbai_source_test.zip"
)