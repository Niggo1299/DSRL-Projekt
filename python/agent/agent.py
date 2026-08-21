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
from problem.problem import SimulationFailedError


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
# 2. REFLEX-AGENT (Simple Reflex Agent für das Hochregallager)
# =====================================================================
class ReflexAgent(Agent):
    """
    Reflex-Agent für das Warehouse-Problem.
    Liest Aktionen für Zustände (inc, b1typ, b2typ) aus einer Nachschlagetabelle (q_table.json).
    """

    def __init__(self, problem, json_file="q_table_reflex.json", manual_control=False):
        super().__init__(problem)
        self.actions = problem.get_all_actions()
        self.manual_control = manual_control
        self.table = {}
        self.json_path = self._resolve_path(json_file)

        if not self.load_table(self.json_path):
            print("Erstelle vollständige Standard-Reflex-Tabelle für alle 48 Zustände...")
            self.generate_default_table()
            self.save_table(self.json_path)

    def _resolve_path(self, filename):
        agent_dir = os.path.dirname(__file__)
        if not os.path.isabs(filename):
            base_name = os.path.basename(filename)
            return os.path.join(agent_dir, base_name)
        return filename

    def act(self):
        """Wählt die Aktion für den aktuellen Zustand aus."""
        current_state = self.problem.get_current_state()
        state_key = (current_state.inc, current_state.b1typ, current_state.b2typ)

        if self.manual_control:
            return self.user_input(state_key)

        if state_key in self.table:
            action = self.table[state_key]
        else:
            action = self._fallback_rule(state_key)

        return action

    def _fallback_rule(self, state_key):
        """Experten-Reflexregel als Fallback."""
        inc, b1typ, b2typ = state_key[:3]
        if b1typ == inc:
            return 1
        elif b2typ == inc:
            return 2
        elif b1typ == 0:
            return 1
        elif b2typ == 0:
            return 2
        else:
            return 3  # Return

    def user_input(self, state_key):
        """Fragt die Aktion interaktiv ab."""
        print(f"\n  Zustand (inc, b1typ, b2typ): {state_key}")
        while True:
            cmd = input("  Aktion [1=buffer1  2=buffer2  3=return  q=Episode beenden]: ").strip().lower()
            if cmd == "q":
                return None
            if cmd in ("1", "2", "3"):
                return int(cmd)
            print("  Ungültige Eingabe.")

    def generate_default_table(self):
        """Generiert eine vollständige Reflex-Tabelle für alle 48 Zustände."""
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
        filepath = path or self.json_path
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.table = {}
            for key, val in data.items():
                if "inc=" in key:
                    parts = key.replace("Zustand (inc=", "").replace("b1typ=", "").replace("b2typ=", "").replace(")", "").split(",")
                    state = (int(parts[0]), int(parts[1]), int(parts[2]))
                else:
                    state = tuple(map(int, key.strip("()").split(",")))

                if isinstance(val, dict):
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
        filepath = path or self.json_path
        readable_data = {}
        for (inc, b1, b2), act in sorted(self.table.items()):
            state_str = f"Zustand (inc={inc}, b1typ={b1}, b2typ={b2})"
            readable_data[state_str] = act

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(readable_data, f, indent=2, ensure_ascii=False)
        print(f"Reflex-Tabelle für {len(self.table)} Zustände gespeichert in '{filepath}'.")

    def print_table(self):
        print("\n=================== REFLEX TABELLE ===================")
        print(f"{'ZUSTAND (inc, b1, b2)':<25} | {'REFLEX AKTION':<15}")
        print("-" * 45)
        for state, act in sorted(self.table.items()):
            act_str = {1: "1 (Puffer 1)", 2: "2 (Puffer 2)", 3: "3 (Return)"}.get(act, str(act))
            print(f"{str(state):<25} | {act_str:<15}")
        print("======================================================\n")


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
                s_cur = self.states[s]           # (b1rel, b2rel, b1cat, b2cat)
                s_nxt = self.states[s_next]      # (b1rel', b2rel', b1cat', b2cat')

                # Puffer 1 und Puffer 2 vertauschen:
                s_sym = (s_cur[1], s_cur[0], s_cur[3], s_cur[2])
                s_nxt_sym = (s_nxt[1], s_nxt[0], s_nxt[3], s_nxt[2])

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
