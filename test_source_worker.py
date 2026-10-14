import subprocess
import json
import sys


city = sys.argv[1] if len(sys.argv) > 1 else "chennai"

print("=" * 60)
print("TESTING SOURCE WORKER:", city)
print("=" * 60)


process = subprocess.Popen(
    [sys.executable, "source_city_worker.py", city],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=None,
    text=True,
    bufsize=1
)


def read_message():
    while True:
        line = process.stdout.readline()

        if not line:
            raise RuntimeError("Worker stopped unexpectedly.")

        line = line.strip()

        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue

        # We only accept JSON objects/dictionaries.
        if isinstance(message, dict):
            return message


# -------------------------------------------------
# Receive RESET
# -------------------------------------------------

message = read_message()

while message.get("type") != "reset":
    message = read_message()

print()
print("RESET RECEIVED:")
print(message)


# -------------------------------------------------
# Send one pump-OFF action
# -------------------------------------------------

command = {
    "command": "step",
    "action": 0
}

process.stdin.write(json.dumps(command) + "\n")
process.stdin.flush()


# -------------------------------------------------
# Receive STEP response
# -------------------------------------------------

message = read_message()

while message.get("type") != "step":
    message = read_message()

print()
print("STEP RESPONSE:")
print(message)


# -------------------------------------------------
# Close worker
# -------------------------------------------------

process.stdin.write(
    json.dumps({"command": "close"}) + "\n"
)
process.stdin.flush()

process.wait(timeout=30)

print()
print("WORKER CLOSED SUCCESSFULLY")

print("=" * 60)
print("SOURCE WORKER TEST PASSED")
print("=" * 60)