"""
agent.py
--------
Reflex-Agent für Plant Simulation.
Liest Aktionen für Zustände aus einer vorgefertigten Nachschlagetabelle (q_table.json).
Enthält KEINE Q-Learning-Trainingslogik mehr.
"""

import os
import json


class SimulationFailedError(Exception):
    """Wird geworfen, wenn die Simulation steht, aber kein StateReady vorliegt."""
    pass


class Agent:
    # ---- Tabellen-/Zellenadressen zentral gehalten -------------------------
    CELL_INC          = "Tab_State[1,1]"
    CELL_B1_COUNT     = "Tab_State[2,1]"
    CELL_B1_TYPE      = "Tab_State[3,1]"
    CELL_B2_COUNT     = "Tab_State[4,1]"
    CELL_B2_TYPE      = "Tab_State[5,1]"
    CELL_STATE_READY  = "Tab_State[6,1]"

    CELL_G_STATE_1    = "Tab_g_State[1,1]"
    CELL_G_STATE_2    = "Tab_g_State[2,1]"
    CELL_G_STATE_3    = "Tab_g_State[3,1]"

    CELL_ACTION       = "Tab_Action[1,1]"
    CELL_ACTION_READY = "Tab_Action[2,1]"

    def __init__(self, plantsim, manual_control=False, json_file="q_table.json"):
        """
        :param plantsim:       Instanz der Plantsim-Wrapper-Klasse
        :param manual_control: True  -> Aktionen kommen vom User (user_input)
                               False -> Aktionen kommen aus der Reflex-Tabelle
        :param json_file:      Name der Tabellendatei im 'agent'-Ordner
        """
        self.plantsim = plantsim
        self.manual_control = manual_control
        self.actions = [1, 2, 3]          # 1 = buffer1, 2 = buffer2, 3 = return

        self.table = {}                   # Dict: (inc, b1typ, b2typ) -> action
        self.json_path = self._resolve_path(json_file)

        self.last_state = None
        self.last_raw_state = None
        self.last_action = None

        # Versuchen, die Tabelle aus JSON zu laden; falls nicht vorhanden, erstelle Standard-Tabelle
        if not self.load_table(self.json_path):
            print("Erstelle vollständige Standard-Reflex-Tabelle für alle 48 Zustände...")
            self.generate_default_table()
            self.save_table(self.json_path)

    def _resolve_path(self, filename):
        """Stellt sicher, dass Dateipfade im 'agent'-Ordner liegen."""
        agent_dir = os.path.dirname(__file__)
        if not os.path.isabs(filename):
            base_name = os.path.basename(filename)
            return os.path.join(agent_dir, base_name)
        return filename

    # ------------------------------------------------------------------ State
    def get_state(self):
        """
        Liest den aktuellen Zustand aus der Plant-Simulation.
        :return: (state_key, raw_state)
                 state_key = (inc, b1typ, b2typ)
        """
        g1 = int(self.plantsim.get_value(self.CELL_G_STATE_1))
        g2 = int(self.plantsim.get_value(self.CELL_G_STATE_2))
        g3 = int(self.plantsim.get_value(self.CELL_G_STATE_3))

        raw_state = {
            "inc":         int(self.plantsim.get_value(self.CELL_INC)),
            "b1cnt":       int(self.plantsim.get_value(self.CELL_B1_COUNT)),
            "b1typ":       int(self.plantsim.get_value(self.CELL_B1_TYPE)),
            "b2cnt":       int(self.plantsim.get_value(self.CELL_B2_COUNT)),
            "b2typ":       int(self.plantsim.get_value(self.CELL_B2_TYPE)),
            "drain_total": g1 + g2 + g3,
        }
        state_key = (raw_state["inc"], raw_state["b1typ"], raw_state["b2typ"])

        self.last_raw_state = raw_state
        self.last_state = state_key
        return state_key, raw_state

    # ----------------------------------------------------------------- Action
    def select_action(self, state):
        """
        Reflex-Aktionsauswahl: Liest die passende Aktion aus der geladenen Tabelle.

        :param state: Tuple (inc, b1typ, b2typ)
        :return: int (1, 2 oder 3) oder None (Episode beenden)
        """
        if self.manual_control:
            action = self.user_input(state)
        else:
            if state in self.table:
                action = self.table[state]
            else:
                # Fallback Reflex-Regel falls Zustand unbekannt ist
                action = self._fallback_rule(state)

        self.last_action = action
        return action

    def _fallback_rule(self, state):
        """Experten-Reflexregel als Fallback."""
        inc, b1typ, b2typ = state
        if b1typ == inc:
            return 1
        elif b2typ == inc:
            return 2
        elif b1typ == 0:
            return 1
        elif b2typ == 0:
            return 2
        else:
            return 3  # Return / Ablehnen

    def set_action(self, action):
        """
        Schreibt die Aktion zurueck und gibt die Simulation frei (Handshake).
        1) StateReady  = False  (Zustand quittiert)
        2) Action      = action
        3) ActionReady = True   (Plant Simulation darf weiterlaufen)
        """
        self.plantsim.set_value(self.CELL_STATE_READY, False)
        self.plantsim.set_value(self.CELL_ACTION, action)
        self.plantsim.set_value(self.CELL_ACTION_READY, True)

    def user_input(self, state):
        """Fragt die Aktion interaktiv ab. 'q' beendet die Episode regulaer."""
        print(f"\n  Zustand (inc, b1typ, b2typ): {state}")
        while True:
            cmd = input(
                "  Aktion [1=buffer1  2=buffer2  3=return  q=Episode beenden]: "
            ).strip().lower()
            if cmd == "q":
                return None
            if cmd in ("1", "2", "3"):
                return int(cmd)
            print("  Ungueltige Eingabe.")

    # ----------------------------------------------- Tabellen-Verwaltung
    def generate_default_table(self):
        """
        Generiert eine vollständige Reflex-Tabelle für alle 48 Zustände.
        Regel:
        1. Passt zu Puffer 1 (b1typ == inc) -> Aktion 1
        2. Passt zu Puffer 2 (b2typ == inc) -> Aktion 2
        3. Puffer 1 ist frei (b1typ == 0)   -> Aktion 1
        4. Puffer 2 ist frei (b2typ == 0)   -> Aktion 2
        5. Sonst (beide mit anderen Typen voll) -> Aktion 3 (Return)
        """
        self.table = {}
        for inc in [1, 2, 3]:
            for b1typ in [0, 1, 2, 3]:
                for b2typ in [0, 1, 2, 3]:
                    state = (inc, b1typ, b2typ)
                    if b1typ == inc:
                        act = 1
                    elif b2typ == inc:
                        act = 2
                    elif b1typ == 0:
                        act = 1
                    elif b2typ == 0:
                        act = 2
                    else:
                        act = 3
                    self.table[state] = act

    def load_table(self, path=None):
        """Lädt die Nachschlagetabelle aus einer JSON-Datei."""
        filepath = path or self.json_path
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.table = {}
            for key, val in data.items():
                # Format: "Zustand (inc=1, b1typ=0, b2typ=2)" oder "(1, 0, 2)"
                if "inc=" in key:
                    parts = key.replace("Zustand (inc=", "").replace("b1typ=", "").replace("b2typ=", "").replace(")", "").split(",")
                    state = (int(parts[0]), int(parts[1]), int(parts[2]))
                else:
                    state = tuple(map(int, key.strip("()").split(",")))

                # Wenn val ein Dict ist (z. B. aus vorherigem Speicherformat):
                if isinstance(val, dict):
                    # Nimm Aktion aus Key if vorhanden, sonst fallback
                    action = val.get("Aktion", val.get("action", 1))
                else:
                    action = int(val)
                self.table[state] = action

            print(f"Reflex-Tabelle mit {len(self.table)} Zuständen geladen aus '{filepath}'.")
            return True
        except (FileNotFoundError, Exception) as e:
            print(f"Hinweis beim Laden von '{filepath}': {e}")
            return False

    def save_table(self, path=None):
        """Speichert die Reflex-Tabelle lesbar als JSON-Datei."""
        filepath = path or self.json_path
        readable_data = {}
        for (inc, b1, b2), act in sorted(self.table.items()):
            state_str = f"Zustand (inc={inc}, b1typ={b1}, b2typ={b2})"
            readable_data[state_str] = act

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(readable_data, f, indent=2, ensure_ascii=False)
        print(f"Reflex-Tabelle für {len(self.table)} Zustände gespeichert in '{filepath}'.")

    def print_table(self):
        """Gibt die Nachschlagetabelle formatiert auf der Konsole aus."""
        print("\n=================== REFLEX TABELLE ===================")
        print(f"{'ZUSTAND (inc, b1, b2)':<25} | {'REFLEX AKTION':<15}")
        print("-" * 45)
        for state, act in sorted(self.table.items()):
            act_str = {1: "1 (Puffer 1)", 2: "2 (Puffer 2)", 3: "3 (Return)"}.get(act, str(act))
            print(f"{str(state):<25} | {act_str:<15}")
        print("======================================================\n")

    def reset(self):
        """Setzt den Episodenzustand des Agenten zurueck."""
        self.last_state = None
        self.last_raw_state = None
        self.last_action = None
