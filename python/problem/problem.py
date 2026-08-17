"""
problem.py
----------
Kapselt die Verbindung und die Kommunikation mit Tecnomatix Plant Simulation.
Implementiert das Problem-Interface (Environment) für den Agenten.
"""

import time


def parse_plantsim_time(time_val):
    """
    Konvertiert Plant-Simulation Zeitstrings (z.B. '31:07.2186' oder '01:30:15.5')
    oder int/float-Werte sauber in Sekunden (float).
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
    """Formatiert Sekunden sauber in 'MM:SS' ohne Millisekunden (z.B. '21:04')."""
    total_sec = int(round(sim_seconds))
    minutes = total_sec // 60
    seconds = total_sec % 60
    return f"{minutes:02d}:{seconds:02d}"


class State:
    """Repräsentiert einen konkreten Simulationszustand aus Plant Simulation."""

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
        """Gibt das diskrete Zustandstupel (inc, b1typ, b2typ) zurück."""
        return (self.inc, self.b1typ, self.b2typ)

    def __repr__(self):
        return (
            f"State(inc={self.inc}, B1=({self.b1cnt}x Typ {self.b1typ}), "
            f"B2=({self.b2cnt}x Typ {self.b2typ}), Drain={self.drain_total}, Time={self.sim_time_str})"
        )


class SimulationFailedError(Exception):
    """Wird geworfen, wenn die Simulation steht oder ein Timeout auftritt."""
    pass


class PlantSimulationProblem:
    """
    Problem-Klasse für das Hochregallager / Sortierproblem in Plant Simulation.
    Übernimmt den Handshake und die Kommunikation mit Plant Simulation.
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

        # Erzeuge alle 48 diskreten Zustände
        self.states = []
        for inc in [1, 2, 3]:
            for b1 in [0, 1, 2, 3]:
                for b2 in [0, 1, 2, 3]:
                    self.states.append((inc, b1, b2))

        # Aktionen: 1 = Puffer 1, 2 = Puffer 2, 3 = Return (Schleife)
        self.actions = [1, 2, 3]

    def get_all_actions(self):
        """Liefert die Liste aller verfügbaren Aktionen."""
        return self.actions

    def get_all_states(self):
        """Liefert die Liste aller 48 diskreten Zustände."""
        return self.states

    def get_applicable_actions(self, current_state=None):
        """Liefert anwendbare Aktionen für den aktuellen Zustand."""
        return self.actions

    def _state_ready(self):
        """Prüft, ob Plant Simulation am Entscheidungspunkt steht."""
        return bool(self.ps.get_value(self.CELL_STATE_READY))

    def _is_running(self):
        """Prüft, ob die Simulation läuft."""
        return self.ps.plantsim.IsSimulationRunning()

    def wait_for_decision(self):
        """
        Wartet, bis Plant Simulation StateReady == True setzt.
        """
        t0 = time.time()
        time.sleep(0.05)  # Kurze Anlaufzeit

        while True:
            if self._state_ready():
                return

            if not self._is_running() and not self._state_ready():
                # Falls Ziel bereits erreicht wurde, kein Fehler
                curr = self.get_current_state()
                if self.is_goal_state(curr):
                    return
                raise SimulationFailedError("Simulation steht ohne StateReady.")

            if time.time() - t0 > self.timeout:
                raise SimulationFailedError(
                    f"Timeout ({self.timeout} s) beim Warten auf Entscheidungspunkt."
                )

            time.sleep(self.poll_interval)

    def get_current_state(self):
        """Liest den aktuellen Zustand aus Plant Simulation."""
        g1 = int(self.ps.get_value(self.CELL_G_STATE_1))
        g2 = int(self.ps.get_value(self.CELL_G_STATE_2))
        g3 = int(self.ps.get_value(self.CELL_G_STATE_3))
        raw_time = self.ps.get_value(self.CELL_G_TIME)
        sim_time = parse_plantsim_time(raw_time)

        state = State(
            inc=int(self.ps.get_value(self.CELL_INC)),
            b1cnt=int(self.ps.get_value(self.CELL_B1_COUNT)),
            b1typ=int(self.ps.get_value(self.CELL_B1_TYPE)),
            b2cnt=int(self.ps.get_value(self.CELL_B2_COUNT)),
            b2typ=int(self.ps.get_value(self.CELL_B2_TYPE)),
            drain_total=g1 + g2 + g3,
            sim_time=sim_time,
        )
        self.last_state = state
        return state

    def act(self, action):
        """
        Führt den Handshake mit Plant Simulation durch:
        1. Quittiert Zustand (StateReady = False)
        2. Schreibt Aktion (Action = action, ActionReady = True)
        3. Startet Simulation wieder
        4. Wartet auf den nächsten Entscheidungspunkt
        """
        self.ps.set_value(self.CELL_STATE_READY, False)
        self.ps.set_value(self.CELL_ACTION, action)
        self.ps.set_value(self.CELL_ACTION_READY, True)

        # Simulation wieder anstarten
        self.ps.start_simulation()

        # Auf nächsten Halt warten
        self.wait_for_decision()

    def reset(self):
        """Setzt die Simulation zurück und wartet auf den 1. Entscheidungspunkt."""
        self.ps.reset_simulation()
        self.ps.start_simulation()
        self.wait_for_decision()
        return self.get_current_state()

    def is_goal_state(self, state):
        """Prüft, ob das Ziel (z. B. 1000 Teile im Drain) erreicht ist."""
        if self.target_drain_count is None:
            return False
        return state.drain_total >= self.target_drain_count

    def get_reward(self, next_state):
        """
        Reward-Funktion für Reinforcement Learning.
        """
        if self.last_state is None:
            return 0.0

        # Positiver Reward bei Zunahme produzierter Teile
        drain_diff = next_state.drain_total - self.last_state.drain_total
        if drain_diff > 0:
            return 10.0 * drain_diff

        # Zeitschritt-Kosten
        return -0.1


