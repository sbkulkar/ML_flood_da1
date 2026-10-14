from pyswmm import Simulation, Subcatchments, Nodes


INPUT_FILE = (
    r"data\swmm\mumbai_hindmata"
    r"\mumbai_hindmata_flood_model.inp"
)


print("=" * 70)
print("MUMBAI SWMM FLOW DIAGNOSTIC")
print("=" * 70)


with Simulation(INPUT_FILE) as sim:

    sim.step_advance(300)

    subcatchments = Subcatchments(sim)
    nodes = Nodes(sim)

    first_subcatchment = next(iter(subcatchments))

    print()
    print("Subcatchment:", first_subcatchment.subcatchmentid)
    print("Outlet:", first_subcatchment.outlet)

    max_runoff = 0.0
    max_inflow = 0.0
    max_depth = 0.0

    for _ in sim:

        runoff = first_subcatchment.runoff

        if runoff > max_runoff:
            max_runoff = runoff

        # Check the node receiving this subcatchment
        outlet_id = first_subcatchment.outlet

        try:
            node = nodes[outlet_id]

            inflow = node.total_inflow
            depth = node.depth

            if inflow > max_inflow:
                max_inflow = inflow

            if depth > max_depth:
                max_depth = depth

        except Exception:
            pass


print()
print("=" * 70)
print("RESULT")
print("=" * 70)

print("Maximum subcatchment runoff :", max_runoff)
print("Maximum outlet node inflow  :", max_inflow)
print("Maximum outlet node depth   :", max_depth)

print("=" * 70)