"""
agent.py
--------
Enthält die Agenten-Klassen gemäß den Vorlesungs-Vorlagen:
- Agent (Basisklasse)
- ReflexAgent (Simple Reflex Agent mit Nachschlagetabelle)
- TrainingTestAgent (Ausführen trainierter Q-Tables)
- QLearningAgent (Tabellarisches Q-Learning mit GLIE-Exploration)
"""

import os
import json
import cProfile
import numpy as np
from problem.problem import SimulationFailedError, get_relative_type


# =====================================================================
# 1. BASISKLASSE (1:1 aus Vorlesungs-Vorlage)
# =====================================================================
class Agent:
    def __init__(self, problem):
        self.problem = problem
        self.action_plan = []
        self.planning_index = 0

    def plan(self, current_state):
        actions = []
        return actions

    def act(self):
        if len(self.action_plan) == 0:
            # percept
            current_state = self.problem.get_current_state()
            # search
            pr = cProfile.Profile()
            pr.enable()

            self.action_plan = self.plan(current_state)

            pr.disable()
            # after your program ends
            pr.print_stats(sort="calls")

        if len(self.action_plan) > 0:
            action = self.action_plan[0]
            current_state_hash = self.problem.to_state()
            if type(action) == dict:
                for k in action.keys():
                    if current_state_hash in k:
                        self.action_plan = action[k]
                        action = self.action_plan[0]
                        break
            self.action_plan = self.action_plan[1:]
            return action

    def to_state(self, state):
        if type(state) == list:
            states = []
            for s in state:
                states.append(s.to_state())
            return frozenset(states)
        else:
            return state.to_state()


# =====================================================================
# 2. REFLEX-AGENT (Relativer Reflex Agent mit Gedächtnis / LastCollect)
# =====================================================================
class ReflexAgent(Agent):
    """
    Reflex-Agent mit relativer Zustandskodierung und Gedächtnis:
    Zustand: (b1rel, b2rel, lastcollect)
    - b1rel: 0 = Leer, 1 = Match, 2 = Dismatch
    - b2rel: 0 = Leer, 1 = Match, 2 = Dismatch
    - lastcollect: 0 = Init, 1 = Collected (selten auf Schleife), 2 = NotCollected (Stautyp auf Schleife!)
    Insgesamt 3 x 3 x 3 = 27 Zustände.
    """

    def __init__(self, problem, json_file="q_table_reflex.json", manual_control=False):
        super().__init__(problem)
        self.actions = problem.get_all_actions()
        self.manual_control = manual_control
        self.table = {}
        self.json_path = self._resolve_path(json_file)

        if not self.load_table(self.json_path):
            print("Erstelle vollständige Standard-Reflex-Tabelle für alle 27 Zustände (mit Gedächtnis)...")
            self.generate_default_table()
            self.save_table(self.json_path)

    def _resolve_path(self, filename):
        agent_dir = os.path.dirname(__file__)
        if not os.path.isabs(filename):
            base_name = os.path.basename(filename)
            return os.path.join(agent_dir, base_name)
        return filename

    def get_state_key(self, current_state):
        """Berechnet das relative Zustandstupel (b1rel, b2rel, lastcollect)."""
        b1rel = get_relative_type(current_state.b1typ, current_state.inc, current_state.b1cnt)
        b2rel = get_relative_type(current_state.b2typ, current_state.inc, current_state.b2cnt)
        lc = getattr(current_state, "lastcollect", 0)
        return (b1rel, b2rel, lc)

    def act(self):
        """Wählt die Aktion für den aktuellen relativen Zustand aus."""
        current_state = self.problem.get_current_state()
        state_key = self.get_state_key(current_state)

        if self.manual_control:
            return self.user_input(state_key, current_state)

        if state_key in self.table:
            action = self.table[state_key]
        else:
            action = self._fallback_rule(state_key)

        return action

    def _fallback_rule(self, state_key):
        """
        Experten-Reflexregel mit Gedächtnis-Nutzung:
        1. Match auf P1 oder P2 -> immer sofort bedienen.
        2. Beide belegt & Dismatch -> immer Return.
        3. Beide Puffer leer -> Puffer 1.
        4. Ein Puffer frei & Gegenpuffer Dismatch:
           - LC=2 (NotCollected): Stautyp! Freien Puffer sofort belegen.
           - LC=1 (Collected): Seltener Typ! In Return schicken und auf Stautyp warten!
           - LC=0 (Init): Freien Puffer belegen.
        """
        b1rel, b2rel, lc = state_key

        # 1. Match hat oberste Priorität
        if b1rel == 1:
            return 1
        elif b2rel == 1:
            return 2

        # 2. Beide belegt und Dismatch -> Return-Schleife
        elif b1rel == 2 and b2rel == 2:
            return 3

        # 3. Beide Puffer leer -> Puffer 1 starten
        elif b1rel == 0 and b2rel == 0:
            return 1

        # 4. P1 leer, P2 Dismatch:
        elif b1rel == 0 and b2rel == 2:
            if lc == 2 or lc == 0:
                return 1  # Stautyp -> Puffer 1 belegen!
            else:
                return 3  # Seltener Typ -> Return (auf Stautyp warten)

        # 5. P1 Dismatch, P2 leer:
        elif b1rel == 2 and b2rel == 0:
            if lc == 2 or lc == 0:
                return 2  # Stautyp -> Puffer 2 belegen!
            else:
                return 3  # Seltener Typ -> Return (auf Stautyp warten)

        # 6. Fallback
        elif b1rel == 0:
            return 1
        elif b2rel == 0:
            return 2
        else:
            return 3

    def user_input(self, state_key, current_state):
        """Fragt die Aktion interaktiv ab."""
        rel_map = {0: "leer", 1: "match", 2: "dismatch"}
        lc_map = {0: "Init", 1: "Collected", 2: "NotCollected"}
        b1_str = rel_map.get(state_key[0], str(state_key[0]))
        b2_str = rel_map.get(state_key[1], str(state_key[1]))
        lc_str = lc_map.get(state_key[2], str(state_key[2]))
        print(f"\n  Zustand: inc={current_state.inc}, B1={b1_str}, B2={b2_str}, LastCollect={lc_str} (Key: {state_key})")
        while True:
            cmd = input("  Aktion [1=Puffer1  2=Puffer2  3=Return  q=Beenden]: ").strip().lower()
            if cmd == "q":
                return None
            if cmd in ("1", "2", "3"):
                return int(cmd)
            print("  Ungültige Eingabe.")

    def generate_default_table(self):
        """Generiert die 27 Standard-Regeln für alle Kombinationen von (b1rel, b2rel, lastcollect)."""
        self.table = {}
        for b1rel in [0, 1, 2]:
            for b2rel in [0, 1, 2]:
                for lc in [0, 1, 2]:
                    self.table[(b1rel, b2rel, lc)] = self._fallback_rule((b1rel, b2rel, lc))

    def load_table(self, path=None):
        filepath = path or self.json_path
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.table = {}
            for key, val in data.items():
                if isinstance(val, dict) and "b1rel" in val and "b2rel" in val:
                    lc = int(val.get("lastcollect", 0))
                    state = (int(val["b1rel"]), int(val["b2rel"]), lc)
                    action = int(val.get("Aktion", 1))
                else:
                    clean = key.strip("()").split(",")
                    state = (int(clean[0]), int(clean[1]), int(clean[2]) if len(clean) > 2 else 0)
                    action = val.get("Aktion", 1) if isinstance(val, dict) else int(val)

                self.table[state] = action

            if len(self.table) == 27:
                print(f"Reflex-Tabelle mit {len(self.table)} Zuständen geladen aus '{filepath}'.")
                return True
            return False
        except (FileNotFoundError, Exception):
            return False

    def save_table(self, path=None):
        filepath = path or self.json_path
        rel_map = {0: "leer", 1: "match", 2: "dismatch"}
        lc_map = {0: "Init", 1: "Collected", 2: "NotCollected"}
        readable_data = {}
        for (b1, b2, lc), act in sorted(self.table.items()):
            state_str = f"Zustand (B1={b1} [{rel_map[b1]}], B2={b2} [{rel_map[b2]}], LC={lc} [{lc_map[lc]}])"
            readable_data[state_str] = {
                "b1rel": b1,
                "b2rel": b2,
                "lastcollect": lc,
                "Aktion": act,
                "Beschreibung": f"Puffer {act}" if act < 3 else "Return-Schleife"
            }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(readable_data, f, indent=2, ensure_ascii=False)
        print(f"Reflex-Tabelle für {len(self.table)} Zustände gespeichert in '{filepath}'.")

    def print_table(self):
        rel_map = {0: "leer", 1: "match", 2: "dismatch"}
        lc_map = {0: "Init", 1: "Collected", 2: "NotCollected"}
        print("\n=================== RELATIVE REFLEX TABELLE (mit Gedächtnis) ===================")
        print(f"{'ZUSTAND (B1, B2, LastCollect)':<45} | {'REFLEX AKTION':<15}")
        print("-" * 65)
        for (b1, b2, lc), act in sorted(self.table.items()):
            state_str = f"B1={rel_map[b1]} ({b1}), B2={rel_map[b2]} ({b2}), LC={lc_map[lc]} ({lc})"
            act_str = {1: "1 (Puffer 1)", 2: "2 (Puffer 2)", 3: "3 (Return)"}.get(act, str(act))
            print(f"{state_str:<45} | {act_str:<15}")
        print("=================================================================================\n")


# =====================================================================
# 3. TRAINING TEST AGENT
# =====================================================================
class TrainingTestAgent(Agent):

    def __init__(self, problem, q_table=None, q_table_file="q_table.npy"):
        super().__init__(problem)
        self.actions = problem.get_all_actions()
        self.states = problem.get_all_states()
        self.file = q_table_file

        if q_table is not None:
            self.q_table = q_table
        else:
            self.load()

    def act(self):
        # perception
        current_state = self.problem.get_current_state()
        s_idx = self.states.index(current_state.to_state())
        # lookup best action in q_table
        action_idx = int(np.argmax(self.q_table[s_idx]))
        return self.actions[action_idx]

    def save(self):
        np.save(self.file, self.q_table)

    def load(self):
        filepath = self.file
        if not os.path.exists(filepath):
            # Fallback search in project root
            agent_dir = os.path.dirname(os.path.abspath(__file__))
            proj_dir = os.path.dirname(os.path.dirname(agent_dir))
            candidate = os.path.join(proj_dir, self.file)
            if os.path.exists(candidate):
                filepath = candidate

        self.q_table = np.load(filepath)
        print(f"[TEST AGENT] Q-Tabelle erfolgreich geladen aus '{filepath}' (Shape: {self.q_table.shape}).")


# =====================================================================
# 4. Q-LEARNING AGENT (1:1 aus Vorlesungs-Vorlage)
# =====================================================================
class QLearningAgent(Agent):

    def __init__(self, problem, q_table=None, N_sa=None, gamma=0.99, max_N_exploration=3, R_Max=100):
        super().__init__(problem)
        self.actions = problem.get_all_actions()
        self.states = problem.get_all_states()
        if q_table is not None:
            self.q_table = q_table
        else:
            self.q_table = np.zeros((len(self.states), (len(self.actions))))
        if N_sa is not None:
            self.N_sa = N_sa
        else:
            self.N_sa = np.zeros((len(self.states), (len(self.actions))))
        self.gamma = gamma
        self.max_N_exploration = max_N_exploration
        self.R_Max = R_Max

    def act(self):
        # perception
        current_state = self.problem.get_current_state()
        s = self.states.index(current_state.to_state())
        # lookup in q_table
        action = self.actions[np.argmax(self.q_table[s])]
        return action

    def train(self, episodes=1000, alpha=0.1, max_steps=100, gamma=None, max_N_exploration=None, R_Max=None):
        if gamma is not None:
            self.gamma = gamma
        if max_N_exploration is not None:
            self.max_N_exploration = max_N_exploration
        if R_Max is not None:
            self.R_Max = R_Max
        
        Steps_needed = np.zeros(episodes)

        for episode in range(episodes):
            self.problem.reset()  # Reset environment to initial state
            current_state = self.problem.get_current_state()
            
            for step in range(max_steps):
                s = self.states.index(current_state.to_state())
                
                # Calculate GLIE exploration function f(u, n)
                f_values = np.zeros(len(self.actions))
                # Assign R_Max to unexplored actions, otherwise use the current q-value
                for a_idx in range(len(self.actions)):
                    u = self.q_table[s, a_idx]
                    n = self.N_sa[s, a_idx]
                    if n < self.max_N_exploration:
                        f_values[a_idx] = self.R_Max
                    else:
                        f_values[a_idx] = u
                
                # Choose action with the highest f(u,n) value; break ties randomly
                max_f = np.max(f_values)
                best_actions_indices = np.where(f_values == max_f)[0]
                a_idx = np.random.choice(best_actions_indices)
                action = self.actions[a_idx]
                
                # Execute action (environment transitions to the next state)
                try:
                    self.problem.act(action)
                    next_state = self.problem.get_current_state()
                    s_next = self.states.index(next_state.to_state())
                    reward = self.problem.get_reward(current_state, next_state)
                except SimulationFailedError as e:
                    print(f"  [STAU / TIMEOUT] Episode {episode + 1} bei Step {step + 1} abgebrochen: {e}")
                    print("  -> Setze Simulation zurueck und starte naechste Episode...")
                    Steps_needed[episode] = max_steps
                    break
                
                # 1. Normales Q-Update für (s, a)
                self.N_sa[s, a_idx] += 1
                best_next_q = np.max(self.q_table[s_next])
                td_error = (reward + self.gamma * best_next_q) - self.q_table[s, a_idx]
                self.q_table[s, a_idx] += alpha * td_error

                # 2. Symmetrisches Q-Update für (s_sym, a_sym) - gespiegelte Puffer
                s_cur = self.states[s]           # (b1rel, b2rel)
                s_nxt = self.states[s_next]      # (b1rel', b2rel')

                # Puffer 1 und Puffer 2 vertauschen:
                s_sym = (s_cur[1], s_cur[0])
                s_nxt_sym = (s_nxt[1], s_nxt[0])

                # Aktion spiegeln (1 <-> 2, 3 bleibt 3):
                a_sym = 2 if action == 1 else (1 if action == 2 else 3)
                a_sym_idx = self.actions.index(a_sym)
                s_sym_idx = self.states.index(s_sym)
                s_nxt_sym_idx = self.states.index(s_nxt_sym)

                # Gespiegelten Q-Wert updaten (nur wenn nicht identischer Zustand):
                if s_sym_idx != s or a_sym_idx != a_idx:
                    self.N_sa[s_sym_idx, a_sym_idx] += 1
                    best_next_q_sym = np.max(self.q_table[s_nxt_sym_idx])
                    td_error_sym = (reward + self.gamma * best_next_q_sym) - self.q_table[s_sym_idx, a_sym_idx]
                    self.q_table[s_sym_idx, a_sym_idx] += alpha * td_error_sym
                
                # Overwrite current state for the next step
                current_state = next_state
                
                # End episode if the goal is reached (everything cleaned)
                if self.problem.is_goal_state(current_state):
                    Steps_needed[episode] = step + 1
                    print(f"  [ERFOLG] Episode {episode + 1} abgeschlossen in {step + 1} Schritten (Drain={current_state.drain_total}, Zeit={current_state.sim_time_str})")
                    break
                
                # Record max_steps if the goal wasn't reached within the limit
                if step == max_steps - 1:
                    Steps_needed[episode] = max_steps
        
        return Steps_needed

    def save_q_table(self, file):
        np.save(file, self.q_table)

    def load_q_table(self, file):
        self.q_table = np.load(file)
