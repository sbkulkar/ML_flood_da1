import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path

from pyswmm import Simulation, Nodes, Links


# ============================================================
# MUMBAI HINDMATA CONFIGURATION
# ============================================================

REGION = "mumbai_hindmata"

BASE_INP = Path(
    "data/swmm/mumbai_hindmata/mumbai_hindmata_flood_model.inp"
)

RL_INP = Path(
    "data/swmm/mumbai_hindmata/mumbai_hindmata_rl_model.inp"
)

CONTROL_NODE = "J_0432"
PUMP_ID = "P1"
OUTFALL_ID = "OUTFALL_01"
ORIGINAL_LINK_ID = "C_0273"

SWMM_STEPS_PER_ACTION = 15

# Mumbai model: 6 hours × 3600 / (15 × 30) = 48 RL actions.
MAX_RL_STEPS = 48

# Mumbai Hindmata SWMM model uses ROUTING_STEP 00:00:30.
ROUTING_STEP_SECONDS = 30.0
SECONDS_PER_ACTION = SWMM_STEPS_PER_ACTION * ROUTING_STEP_SECONDS


# ============================================================
# STORM CONDITIONS
# ============================================================
#
# IMPORTANT:
# Mumbai rainfall is NOT replaced by a synthetic inflow.
#
# The real Mumbai rainfall time series remains:
#     RG_MUMBAI -> MUMBAI_RAIN
#
# The three dashboard conditions are scenarios of that same
# Mumbai rainfall event. SWMM's rain-gage SCF is used to scale
# the original rainfall without inventing a new hyetograph.
#
# MODERATE = 0.50 × actual Mumbai event
# HEAVY    = 1.00 × actual Mumbai event
# EXTREME  = 1.50 × actual Mumbai event
#
# The underlying rainfall timing and all subcatchment
# connections remain unchanged.
# ============================================================

STORM_CONDITIONS = {
    "MODERATE": 0.50,
    "HEAVY": 1.00,
    "EXTREME": 1.50,
}

DEFAULT_STORM_CONDITION = "HEAVY"


# ============================================================
# PUMP CURVE
# ============================================================

PUMP_CURVE_NAME = "PUMP_CURVE"

PUMP_CURVE = [
    ("0.00", "0.00"),
    ("1.00", "0.013"),
    ("2.00", "0.022"),
    ("3.00", "0.030"),
    ("4.00", "0.034"),
]


# ============================================================
# CREATE RL SWMM MODEL
# ============================================================

def create_rl_model(storm_condition=DEFAULT_STORM_CONDITION):
    """
    Create the Mumbai Hindmata RL copy.

    The ONLY hydraulic changes are:
      1. Replace C_0273 (J_0432 -> OUTFALL_01) with controllable P1.
      2. Apply a scenario scale to the existing MUMBAI_RAIN values.

    The original Mumbai rainfall timing and shape are preserved.
    No [INFLOWS] or synthetic [TIMESERIES] section is created.
    """

    if not BASE_INP.exists():
        raise FileNotFoundError(
            f"Mumbai SWMM model not found:\n{BASE_INP}"
        )

    storm_condition = str(storm_condition).upper()

    if storm_condition not in STORM_CONDITIONS:
        raise ValueError(
            f"Unknown storm condition '{storm_condition}'. "
            f"Choose one of: {', '.join(STORM_CONDITIONS)}"
        )

    rainfall_scale = float(STORM_CONDITIONS[storm_condition])

    text = BASE_INP.read_text(encoding="utf-8")

    print()
    print("=" * 70)
    print("CREATING MUMBAI HINDMATA RL SWMM MODEL")
    print("=" * 70)
    print(f"Storm condition: {storm_condition}")
    print(f"Mumbai rainfall scale: {rainfall_scale:.2f}x")
    print("Rainfall source: RG_MUMBAI -> MUMBAI_RAIN (PRESERVED)")

    # --------------------------------------------------------
    # Remove/rebuild pump-related sections only.
    # DO NOT remove [RAINGAGES], [TIMESERIES], or [INFLOWS].
    # --------------------------------------------------------

    def remove_section(content, section_name):
        lines = content.splitlines()
        output = []
        inside = False

        for line in lines:
            stripped = line.strip()

            if stripped.upper() == section_name.upper():
                inside = True
                continue

            if (
                inside
                and stripped.startswith("[")
                and stripped.endswith("]")
            ):
                inside = False

            if not inside:
                output.append(line)

        return "\n".join(output)

    text = remove_section(text, "[PUMPS]")
    text = remove_section(text, "[CURVES]")

    # --------------------------------------------------------
    # Apply the storm scenario to the EXISTING Mumbai rainfall
    # time series. Keep the real Mumbai timing and shape; only
    # scale its recorded rainfall values.
    # --------------------------------------------------------

    lines = text.splitlines()
    in_raingages = False
    in_timeseries = False
    rainfall_gage_found = False
    rainfall_updated = False
    updated_lines = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped.upper()
            in_raingages = section == "[RAINGAGES]"
            in_timeseries = section == "[TIMESERIES]"
            updated_lines.append(line)
            continue

        if in_raingages and stripped and not stripped.startswith(";"):
            parts = stripped.split()
            if parts and parts[0] == "RG_MUMBAI":
                if len(parts) < 5:
                    raise RuntimeError("RG_MUMBAI rain-gage definition is malformed.")
                # Scenario scaling is applied directly to MUMBAI_RAIN below.
                parts[3] = "1.000"
                line = " ".join(parts)
                rainfall_gage_found = True

        elif in_timeseries and stripped and not stripped.startswith(";"):
            parts = stripped.split()
            if len(parts) >= 4 and parts[0] == "MUMBAI_RAIN":
                try:
                    original_value = float(parts[3])
                except ValueError:
                    updated_lines.append(line)
                    continue
                parts[3] = f"{original_value * rainfall_scale:.3f}"
                line = " ".join(parts)
                rainfall_updated = True

        updated_lines.append(line)

    text = "\n".join(updated_lines)

    if not rainfall_gage_found:
        raise RuntimeError("Could not find RG_MUMBAI in [RAINGAGES].")

    if not rainfall_updated:
        raise RuntimeError("Could not find MUMBAI_RAIN values in [TIMESERIES].")

    # --------------------------------------------------------
    # Remove original C_0273 only.
    # --------------------------------------------------------

    sections_to_check = {
        "[CONDUITS]",
        "[XSECTIONS]",
        "[LOSSES]",
    }

    lines = text.splitlines()
    filtered_lines = []
    current_section = None
    removed_conduit = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("[") and stripped.endswith("]"):
            current_section = stripped.upper()
            filtered_lines.append(line)
            continue

        if not stripped or stripped.startswith(";"):
            filtered_lines.append(line)
            continue

        if current_section in sections_to_check:
            parts = stripped.split()

            if parts and parts[0] == ORIGINAL_LINK_ID:
                if current_section == "[CONDUITS]":
                    if (
                        len(parts) >= 3
                        and parts[1] == CONTROL_NODE
                        and parts[2] == OUTFALL_ID
                    ):
                        print(
                            f"Replacing {ORIGINAL_LINK_ID}: "
                            f"{CONTROL_NODE} -> {OUTFALL_ID}"
                        )
                        removed_conduit = True
                        continue
                else:
                    continue

        filtered_lines.append(line)

    text = "\n".join(filtered_lines)

    if not removed_conduit:
        raise RuntimeError(
            f"Could not find expected conduit {ORIGINAL_LINK_ID}: "
            f"{CONTROL_NODE} -> {OUTFALL_ID}"
        )

    # --------------------------------------------------------
    # Build controllable pump + curve.
    # --------------------------------------------------------

    pump_section = "\n[PUMPS]\n"
    pump_section += (
        ";;Name   From Node   To Node       Curve         "
        "Status   Startup   Shutoff\n"
    )
    pump_section += (
        f"{PUMP_ID:<8}"
        f"{CONTROL_NODE:<12}"
        f"{OUTFALL_ID:<14}"
        f"{PUMP_CURVE_NAME:<13}"
        f"OFF      0.000     0.000\n"
    )

    curve_section = "\n[CURVES]\n"
    curve_section += ";;Name          Type       X-Value    Y-Value\n"

    for i, (x, y) in enumerate(PUMP_CURVE):
        if i == 0:
            curve_section += (
                f"{PUMP_CURVE_NAME:<16}"
                f"{'PUMP4':<11}"
                f"{x:<11}"
                f"{y}\n"
            )
        else:
            curve_section += (
                f"{PUMP_CURVE_NAME:<16}"
                f"{'':<11}"
                f"{x:<11}"
                f"{y}\n"
            )

    # --------------------------------------------------------
    # Insert pump sections before [REPORT].
    # --------------------------------------------------------

    if "[REPORT]" in text:
        text = text.replace(
            "[REPORT]",
            pump_section.strip()
            + "\n\n"
            + curve_section.strip()
            + "\n\n"
            + "[REPORT]",
            1,
        )
    else:
        text += "\n\n" + pump_section + curve_section

    # --------------------------------------------------------
    # Safety checks.
    # --------------------------------------------------------

    if "[TIMESERIES]" not in text:
        raise RuntimeError(
            "Mumbai [TIMESERIES] section disappeared. "
            "Refusing to create RL model."
        )

    if "MUMBAI_RAIN" not in text:
        raise RuntimeError(
            "MUMBAI_RAIN disappeared from RL model. "
            "Refusing to create RL model."
        )

    if "RG_MUMBAI" not in text:
        raise RuntimeError(
            "RG_MUMBAI disappeared from RL model. "
            "Refusing to create RL model."
        )

    remaining = []

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.startswith(ORIGINAL_LINK_ID):
            remaining.append(stripped)

    if remaining:
        raise RuntimeError(
            f"{ORIGINAL_LINK_ID} still exists in RL model:\n"
            + "\n".join(remaining)
        )

    # --------------------------------------------------------
    # Write RL model.
    # --------------------------------------------------------

    RL_INP.parent.mkdir(parents=True, exist_ok=True)
    RL_INP.write_text(text, encoding="utf-8")

    print()
    print("Base model:")
    print(BASE_INP)

    print()
    print("RL model:")
    print(RL_INP)

    print()
    print("Control node:")
    print(CONTROL_NODE)

    print()
    print("Pump:")
    print(f"{PUMP_ID}: {CONTROL_NODE} -> {OUTFALL_ID}")

    print()
    print("Rainfall:")
    print("RG_MUMBAI -> MUMBAI_RAIN")
    print(f"Scenario scale: {rainfall_scale:.2f}x")
    print("Mumbai rainfall timing/shape: PRESERVED")

    print()
    print("=" * 70)
    print("MUMBAI RL SWMM MODEL CREATED")
    print("=" * 70)

    return RL_INP


# ============================================================
# MUMBAI DRAINAGE RL ENVIRONMENT
# ============================================================

class MumbaiDrainageEnvironment(gym.Env):

    metadata = {
        "render_modes": []
    }

    def __init__(
        self,
        storm_condition=DEFAULT_STORM_CONDITION,
    ):

        super().__init__()

        self.storm_condition = str(storm_condition).upper()

        if self.storm_condition not in STORM_CONDITIONS:
            raise ValueError(
                f"Unknown storm condition '{self.storm_condition}'. "
                f"Choose one of: {', '.join(STORM_CONDITIONS)}"
            )

        # ----------------------------------------------------
        # SWMM objects
        # ----------------------------------------------------

        self.sim = None
        self.nodes = None
        self.links = None

        self.node_objects = []

        self.control_node = None
        self.pump = None

        self.current_step = 0

        self.previous_flooding = 0.0
        self.previous_depth = 0.0

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(2)

    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
        *,
        seed=None,
        options=None
    ):

        super().reset(seed=seed)

        # ----------------------------------------------------
        # Optional storm-condition override from evaluator/dashboard
        # ----------------------------------------------------

        if options is not None:
            requested_condition = options.get("storm_condition")

            if requested_condition is not None:
                requested_condition = str(requested_condition).upper()

                if requested_condition not in STORM_CONDITIONS:
                    raise ValueError(
                        f"Unknown storm condition '{requested_condition}'. "
                        f"Choose one of: {', '.join(STORM_CONDITIONS)}"
                    )

                self.storm_condition = requested_condition

        self.close()

        create_rl_model(self.storm_condition)

        try:

            self.sim = Simulation(str(RL_INP))
            self.nodes = Nodes(self.sim)
            self.links = Links(self.sim)

            self.sim.start()

        except Exception as e:

            self.close()

            raise RuntimeError(
                "Could not start Mumbai RL SWMM "
                f"environment:\n{e}"
            )

        try:
            self.control_node = self.nodes[CONTROL_NODE]
        except Exception:
            self.close()
            raise RuntimeError(
                f"Node '{CONTROL_NODE}' "
                "was not found in Mumbai SWMM model."
            )

        try:
            self.pump = self.links[PUMP_ID]
        except Exception:
            self.close()
            raise RuntimeError(
                f"Pump '{PUMP_ID}' "
                "was not found in Mumbai RL model."
            )

        # Cache node references ONCE here (proven-safe pattern
        # from the Chennai fix -- direct indexed/cached object
        # access, not repeated fresh iteration every step).
        self.node_objects = list(self.nodes)

        self.current_step = 0
        self.previous_flooding = 0.0
        self.previous_depth = 0.0

        state = self.get_state()

        info = {
            "region": REGION,
            "storm_condition": self.storm_condition,
            "rainfall_scale": STORM_CONDITIONS[self.storm_condition],
            "control_node": CONTROL_NODE,
            "pump": PUMP_ID,
            "outfall": OUTFALL_ID,
        }

        return state, info

    # ========================================================
    # GET NODE FLOODING
    # ========================================================

    def get_node_flooding(self, node):

        try:
            return max(float(node.flooding), 0.0)
        except Exception:
            return 0.0

    # ========================================================
    # GET NETWORK METRICS
    # ========================================================

    def get_network_metrics(self):

        control_depth = max(
            float(self.control_node.depth), 0.0
        )

        control_flooding = self.get_node_flooding(
            self.control_node
        )

        max_depth = 0.0
        total_flooding = 0.0
        maximum_node_flooding = 0.0

        for node in self.node_objects:

            try:

                depth = max(float(node.depth), 0.0)
                flooding = self.get_node_flooding(node)

                max_depth = max(max_depth, depth)
                total_flooding += flooding
                maximum_node_flooding = max(
                    maximum_node_flooding, flooding
                )

            except Exception:
                continue

        pump_flow = self.get_pump_flow()

        try:
            pump_status = float(self.pump.target_setting)
        except Exception:
            pump_status = 0.0

        pump_status = np.clip(pump_status, 0.0, 1.0)

        return {
            "control_depth": control_depth,
            "control_flooding": control_flooding,
            "max_depth": max_depth,
            "total_flooding": max(total_flooding, 0.0),
            "maximum_node_flooding": max(maximum_node_flooding, 0.0),
            "pump_flow": max(pump_flow, 0.0),
            "pump_status": pump_status,
        }

    # ========================================================
    # GET STATE
    # ========================================================

    def get_state(self):

        metrics = self.get_network_metrics()

        state = np.array(
            [
                np.clip(metrics["control_depth"] / 10.0, 0.0, 1.0),
                np.clip(metrics["max_depth"] / 20.0, 0.0, 1.0),
                np.clip(metrics["control_flooding"] / 10.0, 0.0, 1.0),
                np.clip(metrics["total_flooding"] / 1000.0, 0.0, 1.0),
                metrics["pump_status"],
            ],
            dtype=np.float32
        )

        return state

    # ========================================================
    # GET PUMP FLOW
    # ========================================================

    def get_pump_flow(self):

        try:
            return max(float(self.pump.flow), 0.0)
        except Exception:
            return 0.0

    # ========================================================
    # REWARD (unchanged from before -- Chennai-style)
    # ========================================================

    def calculate_reward(
        self, previous_metrics, current_metrics, action
    ):

        # Track the Hindmata control junction depth, NOT whole-network max_depth
        current_depth = float(current_metrics["control_depth"])
        current_flooding = float(current_metrics["control_flooding"])
        current_total_flooding = float(
            current_metrics["total_flooding"]
        )
        pump_status = float(current_metrics["pump_status"])

        reward = 0.0

        # Flooding penalties
        reward -= 80.0 * current_flooding
        reward -= 2.0 * min(current_total_flooding, 10.0)

        # Depth penalties at Hindmata control node
        if current_depth > 2.5:
            reward -= 4.0 * (current_depth - 2.5)

        if current_depth > 3.5:
            reward -= 8.0 * (current_depth - 3.5)

        # Reward flooding reduction at Hindmata
        previous_flooding = float(previous_metrics["control_flooding"])
        flooding_reduction = previous_flooding - current_flooding
        reward += 40.0 * flooding_reduction

        # Reward depth reduction at Hindmata
        previous_depth = float(previous_metrics["control_depth"])
        depth_reduction = previous_depth - current_depth
        reward += 5.0 * depth_reduction

        # Base pump operating cost
        if pump_status > 0.5:
            reward -= 12.0

        # Strong penalty for dry/unnecessary pumping
        if (
            pump_status > 0.5
            and current_depth < 0.75
            and current_flooding <= 0.0
        ):
            reward -= 35.0

        # Moderate penalty for pumping when water level is safely low
        elif (
            pump_status > 0.5
            and current_depth < 1.5
            and current_flooding <= 0.0
        ):
            reward -= 12.0

        # Reward pump operation when Hindmata water level is high
        if pump_status > 0.5 and current_depth >= 2.0:
            reward += 8.0

        # Reward pump operation when Hindmata is flooding
        if pump_status > 0.5 and current_flooding > 0.0:
            reward += 15.0

        # Penalize pump OFF during flooding
        if pump_status <= 0.5 and current_flooding > 0.0:
            reward -= 30.0

        # Reward safe pump-off state when Hindmata is clear and depth is low
        if (
            pump_status <= 0.5
            and current_depth < 1.5
            and current_flooding <= 0.0
        ):
            reward += 8.0

        return float(reward)

    # ========================================================
    # STEP
    # ========================================================

    def step(self, action):

        action = int(action)

        previous_metrics = self.get_network_metrics()

        if action == 0:
            self.pump.target_setting = 0.0
        else:
            self.pump.target_setting = 1.0

        simulation_finished = False

        for _ in range(SWMM_STEPS_PER_ACTION):

            try:
                next(self.sim)
            except StopIteration:
                simulation_finished = True
                break

        self.current_step += 1

        current_metrics = self.get_network_metrics()

        reward = self.calculate_reward(
            previous_metrics, current_metrics, action
        )

        state = self.get_state()

        terminated = False
        truncated = (
            self.current_step >= MAX_RL_STEPS
            or simulation_finished
        )

        info = {
            "region": REGION,
            "storm_condition": self.storm_condition,
            "rainfall_scale": STORM_CONDITIONS[self.storm_condition],
            "step": self.current_step,
            "pump": action,
            "depth": current_metrics["control_depth"],
            "max_depth": current_metrics["max_depth"],
            "flooding": current_metrics["control_flooding"],
            "total_flooding": current_metrics["total_flooding"],
            "maximum_node_flooding": current_metrics[
                "maximum_node_flooding"
            ],
            "pump_flow": current_metrics["pump_flow"],
            "pump_status": current_metrics["pump_status"],
            "seconds_per_step": SECONDS_PER_ACTION,
        }

        return state, reward, terminated, truncated, info

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        if self.sim is not None:
            try:
                self.sim.close()
            except Exception:
                pass

        self.sim = None
        self.nodes = None
        self.links = None
        self.control_node = None
        self.pump = None
        self.node_objects = []


# ============================================================
# SIMPLE MANUAL TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("MUMBAI HINDMATA RL ENVIRONMENT TEST")
    print("=" * 70)

    env = MumbaiDrainageEnvironment(storm_condition=DEFAULT_STORM_CONDITION)

    try:

        state, info = env.reset(seed=42)

        print()
        print("Initial state:")
        print(state)

        print()
        print("Taking one pump-OFF action...")

        state, reward, terminated, truncated, info = env.step(0)

        print()
        print("Next state:", state)
        print("Reward:", reward)
        print("Info:", info)

        print()
        print("=" * 70)
        print("MUMBAI RL TEST PASSED")
        print("=" * 70)

    finally:

        env.close()