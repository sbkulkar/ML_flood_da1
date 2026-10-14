from pyswmm import Simulation, Nodes
from config import MODEL_NAME

MODEL_PATH = f"data/swmm/{MODEL_NAME}.inp"

print("=" * 70)
print("ADYAR FLOOD-PRONE NODE ANALYSIS")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)

sim = Simulation(MODEL_PATH)
nodes = Nodes(sim)

max_flooding = {}

try:
    sim.start()

    for _ in sim:

        for node in nodes:

            try:
                flooding = float(node.flooding)

                if flooding > 0:

                    node_id = node.nodeid

                    if (
                        node_id not in max_flooding
                        or flooding > max_flooding[node_id]
                    ):
                        max_flooding[node_id] = flooding

            except Exception:
                pass

finally:
    sim.close()


sorted_nodes = sorted(
    max_flooding.items(),
    key=lambda x: x[1],
    reverse=True
)


print("\n" + "=" * 70)
print("TOP FLOOD-PRONE NODES")
print("=" * 70)

if not sorted_nodes:

    print("\nNo flooding detected.")

else:

    print(
        f"\n{'Rank':<8}"
        f"{'Node':<20}"
        f"{'Maximum Flooding':>20}"
    )

    print("-" * 50)

    for rank, (node_id, flooding) in enumerate(
        sorted_nodes[:20],
        start=1
    ):

        print(
            f"{rank:<8}"
            f"{node_id:<20}"
            f"{flooding:>20.4f}"
        )

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
