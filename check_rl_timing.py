import sys
sys.argv.append("vellore")

from rl import DrainageEnvironment

env = DrainageEnvironment()

try:
    state, info = env.reset()

    print("Initial state:", state)

    start_time = env.sim.current_time
    print("Start time:", start_time)

    state, reward, terminated, truncated, info = env.step(0)

    end_time = env.sim.current_time
    print("End time:", end_time)

    print("Time advanced by:", end_time - start_time)
    print("RL step:", info["step"])

finally:
    env.close()