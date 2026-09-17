"""
problem.py
----------
Encapsulates the connection and communication with Tecnomatix Plant Simulation.
Implements the problem interface (environment) for the reinforcement learning agent.
"""

import time


def parse_plantsim_time(time_val):
    """
    Converts Plant Simulation time strings (e.g., '31:07.2186' or '01:30:15.5')
    or int/float values into seconds (float).
    """
    if isinstance(time_val, (int, float)):
        return float(time_val)

    str_val = str(time_val).strip()
    try:
        return float(str_val)
    except ValueError:
        pass

    parts = str_val.split(":")
    if len(parts) == 2:
        minutes = float(parts[0])
        seconds = float(parts[1])
        return minutes * 60.0 + seconds
    elif len(parts) == 3:
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return hours * 3600.0 + minutes * 60.0 + seconds

    return 0.0


def format_mm_ss(sim_seconds):
    """Formats seconds into 'MM:SS' format without milliseconds (e.g., '21:04')."""
    total_sec = int(round(sim_seconds))
    minutes = total_sec // 60
    seconds = total_sec % 60
    return f"{minutes:02d}:{seconds:02d}"


def get_relative_type(ptyp, inc, cnt):
    """
    Categorizes the buffer state relative to the incoming part:
    - 0: Empty (ptyp == 0 or cnt == 0)
    - 1: Match (ptyp == inc)
    - 2: Mismatch (ptyp != inc and ptyp != 0)
    """
    if ptyp == 0 or cnt == 0:
        return 0
    elif ptyp == inc:
        return 1
    else:
        return 2


class State:
    """Represents a concrete simulation state from Plant Simulation."""

    def __init__(self, inc, b1cnt, b1typ, b2cnt, b2typ, drain_total=0, sim_time=0.0):
        self.inc = inc
        self.b1cnt = b1cnt
        self.b1typ = b1typ
        self.b2cnt = b2cnt
        self.b2typ = b2typ
        self.drain_total = drain_total
        self.sim_time = sim_time
        self.sim_time_str = format_mm_ss(sim_time)

    def to_state(self):
        """Returns the minimal relative state tuple (b1rel, b2rel) comprising 9 discrete states."""
        b1rel = get_relative_type(self.b1typ, self.inc, self.b1cnt)
        b2rel = get_relative_type(self.b2typ, self.inc, self.b2cnt)
        return (b1rel, b2rel)

    def __repr__(self):
        b1rel = get_relative_type(self.b1typ, self.inc, self.b1cnt)
        b2rel = get_relative_type(self.b2typ, self.inc, self.b2cnt)
        rel_map = {0: "Empty", 1: "Match", 2: "Mismatch"}
        return (
            f"State(inc={self.inc}, B1=({self.b1cnt}x Type {self.b1typ}, {rel_map[b1rel]}), "
            f"B2=({self.b2cnt}x Type {self.b2typ}, {rel_map[b2rel]}), Drain={self.drain_total}, Time={self.sim_time_str})"
        )


class SimulationFailedError(Exception):
    """Raised when the simulation stops unexpectedly or times out."""
    pass


class PlantSimulationProblem:
    """
    Problem class for the buffer and sorting process in Plant Simulation.
    Handles handshaking and communication with Plant Simulation.
    """

    CELL_INC          = "Tab_State[1,1]"
    CELL_B1_COUNT     = "Tab_State[2,1]"
    CELL_B1_TYPE      = "Tab_State[3,1]"
    CELL_B2_COUNT     = "Tab_State[4,1]"
    CELL_B2_TYPE      = "Tab_State[5,1]"
    CELL_STATE_READY  = "Tab_State[6,1]"

    CELL_G_STATE_1    = "Tab_g_State[1,1]"
    CELL_G_STATE_2    = "Tab_g_State[2,1]"
    CELL_G_STATE_3    = "Tab_g_State[3,1]"
    CELL_G_TIME       = "Tab_g_State[4,1]"

    CELL_ACTION       = "Tab_Action[1,1]"
    CELL_ACTION_READY = "Tab_Action[2,1]"

    def __init__(self, plantsim, target_drain_count=1000, poll_interval=0.002, timeout=30.0):
        self.ps = plantsim
        self.target_drain_count = target_drain_count
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.last_state = None

        # Generate all 9 discrete relative states (3 x 3)
        self.states = []
        for b1rel in [0, 1, 2]:
            for b2rel in [0, 1, 2]:
                self.states.append((b1rel, b2rel))

        # Actions: 1 = Buffer 1, 2 = Buffer 2, 3 = Return (loop)
        self.actions = [1, 2, 3]

    def get_all_actions(self):
        """Returns the list of all available actions."""
        return self.actions

    def get_all_states(self):
        """Returns the list of all 9 discrete relative states."""
        return self.states

    def get_applicable_actions(self, current_state=None):
        """Returns applicable actions for the current state."""
        return self.actions

    def _state_ready(self):
        """Checks if Plant Simulation is waiting at a decision point."""
        return bool(self.ps.get_value(self.CELL_STATE_READY))

    def _is_running(self):
        """Checks if the simulation is running."""
        return self.ps.plantsim.IsSimulationRunning()

    def wait_for_decision(self):
        """
        Waits until Plant Simulation sets StateReady == True.
        """
        t0 = time.time()
        time.sleep(0.05)  # Short initial wait time

        while True:
            if self._state_ready():
                return

            if not self._is_running() and not self._state_ready():
                # If target drain count was already reached, do not raise an error
                curr = self.get_current_state()
                if self.is_goal_state(curr):
                    return
                raise SimulationFailedError("Simulation stopped without StateReady.")

            if time.time() - t0 > self.timeout:
                raise SimulationFailedError(
                    f"Timeout ({self.timeout} s) waiting for decision point."
                )

            time.sleep(self.poll_interval)

    def get_current_state(self):
        """Reads the current state from Plant Simulation."""
        g1 = int(self.ps.get_value(self.CELL_G_STATE_1))
        g2 = int(self.ps.get_value(self.CELL_G_STATE_2))
        g3 = int(self.ps.get_value(self.CELL_G_STATE_3))
        raw_time = self.ps.get_value(self.CELL_G_TIME)
        sim_time = parse_plantsim_time(raw_time)

        inc = int(self.ps.get_value(self.CELL_INC))
        b1cnt = int(self.ps.get_value(self.CELL_B1_COUNT))
        b1typ = int(self.ps.get_value(self.CELL_B1_TYPE))
        b2cnt = int(self.ps.get_value(self.CELL_B2_COUNT))
        b2typ = int(self.ps.get_value(self.CELL_B2_TYPE))

        state = State(
            inc=inc,
            b1cnt=b1cnt,
            b1typ=b1typ,
            b2cnt=b2cnt,
            b2typ=b2typ,
            drain_total=g1 + g2 + g3,
            sim_time=sim_time,
        )
        self.last_state = state
        return state

    def act(self, action):
        """
        Executes the handshake with Plant Simulation:
        1. Acknowledge state (StateReady = False)
        2. Set chosen action (Action = action, ActionReady = True)
        3. Resume simulation
        4. Wait for next decision point
        """
        self.ps.set_value(self.CELL_STATE_READY, False)
        self.ps.set_value(self.CELL_ACTION, action)
        self.ps.set_value(self.CELL_ACTION_READY, True)

        # Resume simulation run
        self.ps.start_simulation()

        # Wait for next halt
        self.wait_for_decision()

    def reset(self):
        """Resets the simulation and returns the initial state."""
        self.ps.reset_simulation()
        self.ps.start_simulation()
        self.wait_for_decision()
        return self.get_current_state()

    def is_goal_state(self, state):
        """Checks whether the target goal (e.g., 1000 parts in drain) is reached."""
        if self.target_drain_count is None:
            return False
        return state.drain_total >= self.target_drain_count

    def get_reward(self, state, next_state):
        """
        Reward function for Reinforcement Learning: R(s, s')
        Computes the reward based on the state transition from state -> next_state.
        """
        reward = 0.0
        # 1. Goal not yet reached: small step penalty to encourage faster completion
        if not self.is_goal_state(next_state):
            reward -= 1
        # 2. Check if parts entered the drain (throughput progress)
        drain_diff = next_state.drain_total - state.drain_total
        drain_flow = drain_diff > 0
        if drain_flow:
            reward += 100
        # 3. Check buffer fill levels (part successfully stored)
        if next_state.b1cnt > state.b1cnt:
            reward += next_state.b1cnt
        if next_state.b2cnt > state.b2cnt:
            reward += next_state.b2cnt
        # 4. If no part reached the drain but parts were removed from buffer
        #    (reversal of accumulated Gauss sum: n * (n + 1) / 2)
        if not drain_flow:
            if next_state.b1cnt < state.b1cnt:
                n1 = state.b1cnt
                reward -= (n1 * (n1 + 1)) // 2
            if next_state.b2cnt < state.b2cnt:
                n2 = state.b2cnt
                reward -= (n2 * (n2 + 1)) // 2
        return reward
