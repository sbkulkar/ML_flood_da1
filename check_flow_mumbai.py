from pyswmm import Simulation, Links

INPUT_FILE = r"data\swmm\mumbai_hindmata\mumbai_hindmata_flood_model.inp"

with Simulation(INPUT_FILE) as sim:

    sim.step_advance(300)

    links = Links(sim)
    c0000 = links["C_0000"]

    max_flow = 0.0
    step = 0

    for _ in sim:

        step += 1

        flow = c0000.flow

        max_flow = max(max_flow, flow)

        if step % 5 == 0:
            print(
                f"Step {step:03d} | "
                f"C_0000 Flow: {flow:.6f}"
            )

    print("\n" + "=" * 50)
    print("C_0000 RESULTS")
    print("=" * 50)

    print(f"Maximum flow: {max_flow:.6f}")