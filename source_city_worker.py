import sys
import json


city = sys.argv[1]

if city == "chennai":
    # Chennai region must be selected before importing rl.py
    sys.argv = ["source_city_worker.py", "velachery"]

    from rl import DrainageEnvironment

    env = DrainageEnvironment()

elif city == "mumbai":
    from mumbai_rl import MumbaiDrainageEnvironment

    env = MumbaiDrainageEnvironment(
        storm_condition="HEAVY"
    )

else:
    raise ValueError("Use chennai or mumbai")


def send(data):
    print(json.dumps(data), flush=True)


obs, info = env.reset()

send({
    "type": "reset",
    "observation": obs.tolist()
})


for line in sys.stdin:

    line = line.strip()

    if not line:
        continue

    command = json.loads(line)

    if command["command"] == "step":

        action = int(command["action"])

        result = env.step(action)

        if len(result) == 5:
            obs, reward, terminated, truncated, info = result

        else:
            obs, reward, done, info = result
            terminated = done
            truncated = False

        send({
            "type": "step",
            "observation": obs.tolist(),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated)
        })

        if terminated or truncated:
            break

    elif command["command"] == "close":
        break


env.close()

send({
    "type": "closed"
})