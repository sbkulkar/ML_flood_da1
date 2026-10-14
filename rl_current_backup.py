import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pathlib import Path
import shutil

from pyswmm import Simulation, Nodes, Links
from config import (
    MODEL_NAME,
    RL_MODEL_NAME,
    CURRENT_REGION,
    REGION,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
SWMM_DIR = PROJECT_DIR / "data" / "swmm" / REGION 

ORIGINAL_INP = SWMM_DIR / f"{MODEL_NAME}.inp"
RL_INP = SWMM_DIR / f"{RL_MODEL_NAME}.inp"


# ============================================================
# CONTROL SETTINGS
# ============================================================

CONTROL_NODE = CURRENT_REGION.get("control_node", "J_0431")
PUMP_ID = CURRENT_REGION.get("pump_id", "P1")
OUTFALL_ID = CURRENT_REGION.get("outfall_id", "OUTFALL_01")

# Number of SWMM routing steps for one RL action
SWMM_STEPS_PER_ACTION = 15

# Maximum RL decisions per episode
MAX_RL_STEPS = 288


# ============================================================
# STORM DATA
# ============================================================

DEFAULT_STORM = [
    0.00,
    0.10,
    0.25,
    0.50,
    1.00,
    1.50,
    2.00,
    2.00,
    1.50,
    1.00,
    0.50,
    0.25,
    0.10,
    0.00
]


# ============================================================
# CREATE RL SWMM MODEL
# ============================================================

def create_rl_model(storm_values=None):

    # --------------------------------------------------------
    # CHECK ORIGINAL MODEL
    # --------------------------------------------------------

    if not ORIGINAL_INP.exists():

        raise FileNotFoundError(
            f"Original SWMM model not found:\n{ORIGINAL_INP}"
        )


    # --------------------------------------------------------
    # COPY ORIGINAL MODEL
    # --------------------------------------------------------

    shutil.copyfile(
        ORIGINAL_INP,
        RL_INP
    )


    text = RL_INP.read_text(
        encoding="utf-8"
    )


    # --------------------------------------------------------
    # STORM
    # --------------------------------------------------------

    if storm_values is None:

        storm_values = DEFAULT_STORM


    # --------------------------------------------------------
    # REMOVE OLD RL SECTIONS
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


    text = remove_section(
        text,
        "[INFLOWS]"
    )

    text = remove_section(
        text,
        "[TIMESERIES]"
    )
    
    text = remove_section(
    text,
    "[RAINGAGES]"
    )

    text = remove_section(
        text,
        "[PUMPS]"
    )

    text = remove_section(
        text,
        "[CURVES]"
    )
    
    text = remove_section(
    text,
    "[CURVES]"
    )

    # Detect the rain gage name from the ORIGINAL SWMM model
    rain_gage_name = "RG_VELLORE"

    original_lines = ORIGINAL_INP.read_text(
        encoding="utf-8"
    ).splitlines()

    for line in original_lines:
        stripped = line.strip()

        if stripped.startswith("RG_") and "TIMESERIES" in stripped:
            parts = stripped.split()

            if len(parts) >= 1:
                rain_gage_name = parts[0]

            break

    rain_gage = "\n[RAINGAGES]\n"
    rain_gage += ";;Name           Format       Interval       SCF      Source\n"
    rain_gage += (
        f"{rain_gage_name:<15} INTENSITY    01:00          "
        f"1.0      TIMESERIES RL_STORM\n"
    )
    
    storm = "\n\n[TIMESERIES]\n"
    storm += ";;Name        Time        Value\n"

    for hour, value in enumerate(storm_values):
        storm += f"RL_STORM     {hour:02d}:00       {float(value):.3f}\n"


    # ========================================================
    # CREATE STORM
    # ========================================================
     
     
    # Detect the rain gage name used by the original SWMM model

        


    # ========================================================
    # CREATE PUMP
    # ========================================================

    pump = "\n[PUMPS]\n"

    pump += (
        ";;Name    From Node    To Node      "
        "Pump Curve    Status\n"
    )

    pump += (
        f"{PUMP_ID:<9}"
        f"{CONTROL_NODE:<13}"
        f"{OUTFALL_ID:<13}"
        f"PUMP_CURVE    OFF\n"
    )


    # ========================================================
    # CREATE PUMP CURVE
    # ========================================================

    # SWMM PUMP2:
    # X = inlet depth in metres
    # Y = pump flow in CMS

    curve = "\n[CURVES]\n"

    curve += (
        ";;Name          Type        "
        "X-Value     Y-Value\n"
    )

    curve += (
        "PUMP_CURVE      PUMP2       "
        "0.00        0.00\n"
    )

    curve += (
        "PUMP_CURVE                  "
        "0.50        0.25\n"
    )

    curve += (
        "PUMP_CURVE                  "
        "1.00        0.50\n"
    )

    curve += (
        "PUMP_CURVE                  "
        "2.00        0.75\n"
    )

    curve += (
        "PUMP_CURVE                  "
        "3.00        1.00\n"
    )

    curve += (
        "PUMP_CURVE                  "
        "4.00        1.00\n"
    )


    # ========================================================
    # INSERT SECTIONS
    # ========================================================

    lines = text.splitlines()

    output = []

    inserted = False


    for line in lines:

        if (
            line.strip().upper() == "[REPORT]"
            and not inserted
        ):    
            output.append(
            rain_gage.rstrip()
            )

            output.append(
                storm.rstrip()
            )

            output.append(
                pump.rstrip()
            )

            output.append(
                curve.rstrip()
            )

            output.append("")

            output.append(line)

            inserted = True


        else:

            output.append(line)


    # --------------------------------------------------------
    # REPORT SECTION NOT FOUND
    # --------------------------------------------------------

    if not inserted:

        text = "\n".join(output)

        text += storm

        text += pump

        text += curve


    else:

        text = "\n".join(output)


    # ========================================================
    # SAVE MODEL
    # ========================================================

    RL_INP.write_text(
        text,
        encoding="utf-8"
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    print()

    print("=" * 60)

    print("RL SWMM MODEL CREATED")

    print("=" * 60)

    print()

    print("Model:")

    print(RL_INP)

    print()

    print("Control node:")

    print(CONTROL_NODE)

    print()

    print("Pump:")

    print(
        f"{PUMP_ID}: "
        f"{CONTROL_NODE} -> {OUTFALL_ID}"
    )

    print()

    print("Pump curve:")

    print("PUMP_CURVE (PUMP2)")

    print()

    print("=" * 60)


# ============================================================
# RL ENVIRONMENT
# ============================================================

class DrainageEnvironment(gym.Env):

    metadata = {
        "render_modes": []
    }


    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(self, storm_values=None):

        super().__init__()

        self.storm_values = storm_values


        # ----------------------------------------------------
        # OBSERVATION SPACE
        # ----------------------------------------------------

        # [0] Control node depth
        # [1] Maximum network depth
        # [2] Control node flooding
        # [3] Total network flooding
        # [4] Pump status

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32
        )


        # ----------------------------------------------------
        # ACTION SPACE
        # ----------------------------------------------------

        # 0 = pump OFF
        # 1 = pump ON

        self.action_space = spaces.Discrete(2)


        # ----------------------------------------------------
        # SWMM OBJECTS
        # ----------------------------------------------------

        self.sim = None

        self.nodes = None

        self.links = None

        self.control_node = None

        self.pump = None


        # ----------------------------------------------------
        # RL VARIABLES
        # ----------------------------------------------------

        self.rl_step = 0

        self.previous_flooding = 0.0

        self.previous_depth = 0.0


    # ========================================================
    # RESET
    # ========================================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)


        # ----------------------------------------------------
        # CLOSE PREVIOUS SIMULATION
        # ----------------------------------------------------

        self.close()


        # ----------------------------------------------------
        # CREATE FRESH SWMM MODEL
        # ----------------------------------------------------

        create_rl_model(
            self.storm_values
        )


        # ----------------------------------------------------
        # START SWMM
        # ----------------------------------------------------

        self.sim = Simulation(
            str(RL_INP)
        )

        self.nodes = Nodes(
            self.sim
        )

        self.links = Links(
            self.sim
        )


        # ----------------------------------------------------
        # GET CONTROL NODE
        # ----------------------------------------------------

        try:

            self.control_node = self.nodes[
                CONTROL_NODE
            ]

        except Exception:

            self.close()

            raise RuntimeError(
                f"Node '{CONTROL_NODE}' "
                f"was not found in SWMM model."
            )


        # ----------------------------------------------------
        # GET PUMP
        # ----------------------------------------------------

        try:

            self.pump = self.links[
                PUMP_ID
            ]

        except Exception:

            self.close()

            raise RuntimeError(
                f"Pump '{PUMP_ID}' "
                f"was not found in SWMM model."
            )


        # ----------------------------------------------------
        # START SIMULATION
        # ----------------------------------------------------

        self.sim.start()

        self.rl_step = 0

        self.previous_flooding = 0.0

        self.previous_depth = 0.0


        # ----------------------------------------------------
        # INITIAL STATE
        # ----------------------------------------------------

        state = self.get_state()

        return state, {}


    # ========================================================
    # GET STATE
    # ========================================================

    def get_state(self):

        # ----------------------------------------------------
        # CONTROL NODE DEPTH
        # ----------------------------------------------------

        depth = max(
            float(
                self.control_node.depth
            ),
            0.0
        )


        # ----------------------------------------------------
        # CONTROL NODE FLOODING
        # ----------------------------------------------------

        flooding = max(
            float(
                self.control_node.flooding
            ),
            0.0
        )


        # ----------------------------------------------------
        # NETWORK MAXIMUM DEPTH
        # ----------------------------------------------------

        max_depth = 0.0


        # ----------------------------------------------------
        # TOTAL NETWORK FLOODING
        # ----------------------------------------------------

        total_flooding = 0.0


        for node in self.nodes:

            node_depth = max(
                float(node.depth),
                0.0
            )

            node_flooding = max(
                float(node.flooding),
                0.0
            )

            max_depth = max(
                max_depth,
                node_depth
            )

            total_flooding += node_flooding


        # ----------------------------------------------------
        # PUMP STATUS
        # ----------------------------------------------------

        try:

            pump_status = float(
                self.pump.target_setting
            )

        except Exception:

            pump_status = 0.0


        # ----------------------------------------------------
        # NORMALIZE STATE
        # ----------------------------------------------------

        state = np.array(
            [
                np.clip(
                    depth / 10.0,
                    0.0,
                    1.0
                ),

                np.clip(
                max_depth / 20.0,
                  0.0,
                  1.0
                ),

                np.clip(
                    flooding / 10.0,
                    0.0,
                    1.0
                ),

                np.clip(
                    total_flooding / 1000.0,
                    0.0,
                    1.0
                ),

                np.clip(
                    pump_status,
                    0.0,
                    1.0
                )
            ],
            dtype=np.float32
        )

        return state


    # ========================================================
    # REWARD
    # ========================================================

    def get_reward(self, state):

        # ----------------------------------------------------
        # CONVERT NORMALIZED VALUES
        # ----------------------------------------------------

        depth = (
            float(state[0]) * 10.0
        )

        flooding = (
            float(state[2]) * 10.0
        )

        total_flooding = (
            float(state[3]) * 1000.0
        )

        pump_status = float(
            state[4]
        )


        reward = 0.0


        # ====================================================
        # 1. FLOODING PENALTY
        # ====================================================

        reward -= (
            80.0 * flooding
        )

        reward -= (
            8.0 * total_flooding
        )


        # ====================================================
        # 2. DANGEROUS DEPTH
        # ====================================================

        if depth > 3.0:

            reward -= (
                4.0 * (depth - 3.0)
            )


        if depth > 5.0:

            reward -= (
                8.0 * (depth - 5.0)
            )


        # ====================================================
        # 3. FLOOD REDUCTION
        # ====================================================

        flood_reduction = (
            self.previous_flooding
            - flooding
        )

        reward += (
            40.0 * flood_reduction
        )


        # ====================================================
        # 4. DEPTH REDUCTION
        # ====================================================

        depth_reduction = (
            self.previous_depth
            - depth
        )

        reward += (
            5.0 * depth_reduction
        )


        # ====================================================
        # 5. PUMP OPERATING COST
        # ====================================================

        if pump_status > 0.5:

            reward -= 12.0


        # ====================================================
        # 6. STRONG PENALTY FOR UNNECESSARY PUMPING
        # ====================================================

        if (
            pump_status > 0.5
            and depth < 0.75
            and flooding <= 0.0
            and total_flooding <= 0.0
        ):

            reward -= 35.0


        # ====================================================
        # 7. MODERATE PENALTY FOR PUMPING AT LOW DEPTH
        # ====================================================

        elif (
            pump_status > 0.5
            and depth < 1.5
            and flooding <= 0.0
        ):

            reward -= 12.0


        # ====================================================
        # 8. REWARD PUMPING WHEN WATER LEVEL IS HIGH
        # ====================================================

        if (
            pump_status > 0.5
            and depth >= 2.0
        ):

            reward += 8.0


        # ====================================================
        # 9. REWARD PUMP DURING FLOODING
        # ====================================================

        if (
            pump_status > 0.5
            and flooding > 0.0
        ):

            reward += 15.0


        # ====================================================
        # 10. PENALIZE PUMP OFF DURING FLOODING
        # ====================================================

        if (
            pump_status <= 0.5
            and flooding > 0.0
        ):

            reward -= 30.0


        # ====================================================
        # 11. REWARD SAFE PUMP-OFF STATE
        # ====================================================

        if (
            pump_status <= 0.5
            and depth < 1.5
            and flooding <= 0.0
            and total_flooding <= 0.0
        ):

            reward += 8.0


        # ====================================================
        # STORE CURRENT VALUES
        # ====================================================

        self.previous_flooding = flooding

        self.previous_depth = depth


        return float(reward)


    # ========================================================
    # STEP
    # ========================================================

    def step(self, action):

        action = int(action)


        # ----------------------------------------------------
        # CONTROL PUMP
        # ----------------------------------------------------

        if action == 0:

            self.pump.target_setting = 0.0

        else:

            self.pump.target_setting = 1.0


        # ----------------------------------------------------
        # ADVANCE SWMM
        # ----------------------------------------------------

        simulation_finished = False


        for _ in range(
            SWMM_STEPS_PER_ACTION
        ):

            try:

                next(self.sim)

            except StopIteration:

                simulation_finished = True

                break


        # ----------------------------------------------------
        # RL STEP
        # ----------------------------------------------------

        self.rl_step += 1


        # ----------------------------------------------------
        # GET STATE
        # ----------------------------------------------------

        state = self.get_state()


        # ----------------------------------------------------
        # GET REWARD
        # ----------------------------------------------------

        reward = self.get_reward(
            state
        )


        # ----------------------------------------------------
        # EPISODE STATUS
        # ----------------------------------------------------

        terminated = False

        truncated = (
            self.rl_step >= MAX_RL_STEPS
            or simulation_finished
        )


        # ----------------------------------------------------
        # INFORMATION
        # ----------------------------------------------------

        info = {

            "step": self.rl_step,

            "pump": action,

            "depth": float(
                state[0] * 10.0
            ),

            "max_depth": float(
                state[1] * 10.0
            ),

            "flooding": float(
                state[2] * 10.0
            ),

            "total_flooding": float(
                state[3] * 1000.0
            )
        }


        return (
            state,
            reward,
            terminated,
            truncated,
            info
        )


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


# ============================================================
# BASIC TEST
# ============================================================

if __name__ == "__main__":

    print()

    print("=" * 60)

    print("DRAINAGE RL ENVIRONMENT TEST")

    print("=" * 60)


    env = DrainageEnvironment()


    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    try:

        state, info = env.reset()

    except Exception as e:

        print()

        print(
            "ERROR WHILE STARTING ENVIRONMENT:"
        )

        print(e)

        print()

        print(
            "Check the SWMM .inp file and the IDs:"
        )

        print(
            f"CONTROL_NODE = {CONTROL_NODE}"
        )

        print(
            f"PUMP_ID      = {PUMP_ID}"
        )

        print(
            f"OUTFALL_ID   = {OUTFALL_ID}"
        )

        env.close()

        raise


    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print()

    print("Observation space:")

    print(env.observation_space)

    print()

    print("Action space:")

    print(env.action_space)

    print()

    print("Initial state:")

    print(state)


    # --------------------------------------------------------
    # TEST ACTIONS
    # --------------------------------------------------------

    print()

    print("Testing actions...")

    print()


    for i in range(10):

        # Alternate OFF / ON

        action = i % 2


        (
            state,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(action)


        print(
            f"Step {i + 1:02d} | "
            f"Pump {'ON ' if action == 1 else 'OFF'} | "
            f"Depth {info['depth']:.3f} | "
            f"MaxDepth {info['max_depth']:.3f} | "
            f"Flooding {info['flooding']:.3f} | "
            f"Reward {reward:.3f}"
        )


        if terminated or truncated:

            print()

            print(
                "Simulation finished."
            )

            break


    # --------------------------------------------------------
    # CLOSE
    # --------------------------------------------------------

    env.close()


    print()

    print("=" * 60)

    print("ENVIRONMENT TEST COMPLETE")

    print("=" * 60)