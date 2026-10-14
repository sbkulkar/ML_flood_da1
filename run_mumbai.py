from pyswmm import Simulation, Nodes

INPUT_FILE = "data/swmm/mumbai_hindmata/mumbai_hindmata_flood_model.inp"
max_depth = 0.0
max_depth_node = None

max_flooding = 0.0
max_flooding_node = None

max_inflow = 0.0
max_inflow_node = None

steps = 0

with Simulation(INPUT_FILE) as sim:

    sim.step_advance(300)

    nodes = Nodes(sim)

    # Store the actual Node objects
    node_objects = list(nodes)

    print("Number of nodes:", len(node_objects))

    for _ in sim:

        steps += 1

        for node in node_objects:

            depth = float(node.depth)
            flooding = float(node.flooding)
            inflow = float(node.total_inflow)

            if depth > max_depth:
                max_depth = depth
                max_depth_node = node.nodeid

            if flooding > max_flooding:
                max_flooding = flooding
                max_flooding_node = node.nodeid

            if inflow > max_inflow:
                max_inflow = inflow
                max_inflow_node = node.nodeid


print()
print("=" * 60)
print("MUMBAI HINDMATA SWMM RESULTS")
print("=" * 60)

print(f"Simulation steps       : {steps}")

print(f"Maximum node depth     : {max_depth:.6f} m")
print(f"Maximum depth node     : {max_depth_node}")

print(f"Maximum node flooding  : {max_flooding:.6f}")
print(f"Maximum flood node     : {max_flooding_node}")

print(f"Maximum node inflow    : {max_inflow:.6f}")
print(f"Maximum inflow node    : {max_inflow_node}")

print("=" * 60)