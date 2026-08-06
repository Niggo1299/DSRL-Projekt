"""
agent.py
--------
Enthaelt die Agent-Klasse (RL-Platzhalter) sowie die Exception,
die ein fehlgeschlagenes Simulationsende signalisiert.
"""


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
    CELL_ACTION       = "Tab_Action[1,1]"
    CELL_ACTION_READY = "Tab_Action[2,1]"

    def __init__(self, plantsim, manual_control=True):
        """
        :param plantsim:       Instanz der Plantsim-Wrapper-Klasse
        :param manual_control: True  -> Aktionen kommen vom User (user_input)
                               False -> Aktionen kommen vom Agenten (Policy)
        """
        self.plantsim = plantsim
        self.manual_control = manual_control

        self.actions = [1, 2, 3]          # 1 = buffer1, 2 = buffer2, 3 = return

        self.last_state = None
        self.last_action = None

        self.q_table = {}                 # Platzhalter fuer spaeteres RL

    # ------------------------------------------------------------------ State
    def get_state(self):
        """Liest den aktuellen Zustand aus der Plant-Simulation."""
        state = {
            "inc":   int(self.plantsim.get_value(self.CELL_INC)),
            "b1cnt": int(self.plantsim.get_value(self.CELL_B1_COUNT)),
            "b1typ": int(self.plantsim.get_value(self.CELL_B1_TYPE)),
            "b2cnt": int(self.plantsim.get_value(self.CELL_B2_COUNT)),
            "b2typ": int(self.plantsim.get_value(self.CELL_B2_TYPE)),
        }
        self.last_state = state
        return state

    # ----------------------------------------------------------------- Action
    def select_action(self, state):
        """
        Waehlt eine Aktion.
        manual_control == True  -> Usereingabe
        manual_control == False -> (spaeter) Policy des Agenten

        :return: int aus self.actions oder None (Episode beenden)
        """
        if self.manual_control:
            action = self.user_input(state)
        else:
            # TODO: Policy implementieren (z. B. epsilon-greedy auf self.q_table)
            action = self.actions[0]

        self.last_action = action
        return action

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
        print(f"\n  Zustand: {state}")
        while True:
            cmd = input(
                "  Aktion [1=buffer1  2=buffer2  3=return  q=Episode beenden]: "
            ).strip().lower()
            if cmd == "q":
                return None
            if cmd in ("1", "2", "3"):
                return int(cmd)
            print("  Ungueltige Eingabe.")

    # ----------------------------------------------------------- RL-Platzhalter
    def get_reward(self):
        """Platzhalter: Reward aus der Simulation lesen/berechnen."""
        return 0.0

    def train(self):
        """Platzhalter: Q-Update / Lernschritt."""
        pass

    def save_q_table(self, path):
        """Platzhalter: Q-Table speichern."""
        pass

    def load_q_table(self, path):
        """Platzhalter: Q-Table laden."""
        pass

    # ------------------------------------------------------------------ Helper
    def reset(self):
        """Setzt den Episodenzustand des Agenten zurueck."""
        self.last_state = None
        self.last_action = None