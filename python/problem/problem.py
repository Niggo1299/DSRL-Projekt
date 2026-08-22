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


def get_relative_type(ptyp, inc, cnt):
    """
    Kategorisiert den Puffertyp relativ zum ankommenden Teil:
    - 0: Leer (ptyp == 0 oder cnt == 0)
    - 1: Match (ptyp == inc)
    - 2: Dismatch (ptyp != inc und ptyp != 0)
    """
    if ptyp == 0 or cnt == 0:
        return 0
    elif ptyp == inc:
        return 1
    else:
        return 2


def get_fill_category(count):
    """
    Kategorisiert den Pufferfüllstand in 3 Stufen:
    - 1: 0 bis 3 Teile (niedrig/leer)
    - 2: 4 bis 6 Teile (mittel)
    - 3: 7 bis 10 Teile (fast voll / voll)
    """
    if count <= 3:
        return 1
    elif count <= 6:
        return 2
    else:
        return 3


class State:
    """Repräsentiert einen konkreten Simulationszustand aus Plant Simulation."""

    def __init__(self, inc, b1cnt, b1typ, b2cnt, b2typ, lastcollect=0, drain_total=0, sim_time=0.0):
        self.inc = inc
        self.b1cnt = b1cnt
        self.b1typ = b1typ
        self.b2cnt = b2cnt
        self.b2typ = b2typ
        self.lastcollect = lastcollect
        self.drain_total = drain_total
        self.sim_time = sim_time
        self.sim_time_str = format_mm_ss(sim_time)

    def to_state(self):
        """Gibt das minimale relative Zustandstupel (b1rel, b2rel) mit 9 Zuständen zurück."""
        b1rel = get_relative_type(self.b1typ, self.inc, self.b1cnt)
        b2rel = get_relative_type(self.b2typ, self.inc, self.b2cnt)
        return (b1rel, b2rel)

    def __repr__(self):
        b1rel = get_relative_type(self.b1typ, self.inc, self.b1cnt)
        b2rel = get_relative_type(self.b2typ, self.inc, self.b2cnt)
        rel_map = {0: "Leer", 1: "Match", 2: "Dismatch"}
        return (
            f"State(inc={self.inc}, B1=({self.b1cnt}x Typ {self.b1typ}, {rel_map[b1rel]}), "
            f"B2=({self.b2cnt}x Typ {self.b2typ}, {rel_map[b2rel]}), Drain={self.drain_total}, Time={self.sim_time_str})"
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

        # Erzeuge alle 9 diskreten relativen Zustände (3 x 3)
        self.states = []
        for b1rel in [0, 1, 2]:
            for b2rel in [0, 1, 2]:
                self.states.append((b1rel, b2rel))

        # Aktionen: 1 = Puffer 1, 2 = Puffer 2, 3 = Return (Schleife)
        self.actions = [1, 2, 3]

    def get_all_actions(self):
        """Liefert die Liste aller verfügbaren Aktionen."""
        return self.actions

    def get_all_states(self):
        """Liefert die Liste aller 9 diskreten relativen Zustände."""
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
        """Liest den aktuellen Zustand aus Plant Simulation und aktualisiert das Gedächtnis."""
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

        # --- UPDATE-REGEL FÜR INTERNEN SPEICHER (Gedächtnis) ---
        # Der Speicher wird NUR aktualisiert, wenn ein echter Typ im Puffer liegt.
        # Ein Leerlaufen des Puffers (Drain) löscht das Gedächtnis nicht!
        if b1typ != 0 and b1cnt > 0:
            self.mem_b1 = b1typ
        if b2typ != 0 and b2cnt > 0:
            self.mem_b2 = b2typ

        # --- BERECHNUNG VON LASTCOLLECT ---
        if self.mem_b1 == 0 and self.mem_b2 == 0:
            lastcollect = 0  # 0 = Aufwärmphase (noch kein Puffer je belegt)
        elif inc == self.mem_b1 or inc == self.mem_b2:
            lastcollect = 1  # 1 = collected (Typ lag jüngst in Puffer 1 oder 2)
        else:
            lastcollect = 2  # 2 = notcollected (Typ lag jüngst NICHT in Puffern -> staut sich auf Schleife!)

        state = State(
            inc=inc,
            b1cnt=b1cnt,
            b1typ=b1typ,
            b2cnt=b2cnt,
            b2typ=b2typ,
            lastcollect=lastcollect,
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
        """Setzt die Simulation und den internen Speicher zurück."""
        self.mem_b1 = 0
        self.mem_b2 = 0
        self.ps.reset_simulation()
        self.ps.start_simulation()
        self.wait_for_decision()
        return self.get_current_state()

    def is_goal_state(self, state):
        """Prüft, ob das Ziel (z. B. 1000 Teile im Drain) erreicht ist."""
        if self.target_drain_count is None:
            return False
        return state.drain_total >= self.target_drain_count

    def get_reward(self, state, next_state):
        """
        Reward-Funktion für Reinforcement Learning: R(s, s')
        Berechnet die Belohnung anhand des Zustandsübergangs von state -> next_state.
        """
        reward = 0.0
        # 1. Zielzustand noch nicht erreicht: kleiner Schritt-Abzug (fördert schnelles Lösen)
        if not self.is_goal_state(next_state):
            reward -= 1
        # 2. Prüfen, ob Teile im Drain gelandet sind (Fortschritt/Durchsatz)
        drain_diff = next_state.drain_total - state.drain_total
        drain_flow = drain_diff > 0
        if drain_flow:
            reward += 100
        # 3. Puffer-Füllstände prüfen
        # Zunahme im Puffer (Teil erfolgreich zwischengespeichert)
        if next_state.b1cnt > state.b1cnt:
            reward += next_state.b1cnt
        if next_state.b2cnt > state.b2cnt:
            reward += next_state.b2cnt
        # 4. Wenn kein Teil zum Drain geflossen ist, aber Teile aus Puffer verloren gingen
        #    (Rückabwicklung der exakt angesammelten Gauß-Summe: n * (n + 1) / 2)
        if not drain_flow:
            if next_state.b1cnt < state.b1cnt:
                n1 = state.b1cnt
                reward -= (n1 * (n1 + 1)) // 2
            if next_state.b2cnt < state.b2cnt:
                n2 = state.b2cnt
                reward -= (n2 * (n2 + 1)) // 2
        return reward
    


