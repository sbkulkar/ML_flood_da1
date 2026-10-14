import sys
import json

# -------------------------------------------------
# This script runs ONE city in a separate process.
# Existing Chennai/Mumbai code is not modified.
# -------------------------------------------------

city = sys.argv[1]

if city == "chennai":
    sys.argv = ["source_city_runner.py", "velachery"]

    from rl import DrainageEnvironment

    env = DrainageEnvironment()

elif city == "mumbai":
    from mumbai_rl import MumbaiDrainageEnvironment

    env = MumbaiDrainageEnvironment(
        storm_condition="HEAVY"
    )

else:
    raise ValueError("Use: chennai or mumbai")


print("CITY:", city)
print("OBSERVATION:", env.observation_space)
print("ACTION:", env.action_space)

obs, info = env.reset()

print("RESET SUCCESS")
print("INITIAL OBSERVATION:", obs)

env.close()

print("ENVIRONMENT CLOSED")