"""
agent.py
--------
Contains agent classes according to lecture templates:
- Agent (base class)
- QLearningAgent (tabular Q-learning with GLIE exploration)
"""

import os
import csv
import cProfile
import numpy as np
from problem.problem import SimulationFailedError


# =====================================================================
# 1. BASE CLASS (1:1 from lecture template)
# =====================================================================
class Agent:
    def __init__(self, problem):
        self.problem = problem
        self.action_plan = []
        self.planning_index = 0
        self.q_history = []

    def plan(self, current_state):
        actions = []
        return actions

    def act(self):
        if len(self.action_plan) == 0:
            # perception
            current_state = self.problem.get_current_state()
            # search
            pr = cProfile.Profile()
            pr.enable()

            self.action_plan = self.plan(current_state)

            pr.disable()
            # print profiling stats after search completes
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
# 2. Q-LEARNING AGENT (1:1 from lecture template)
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
        # lookup best greedy action in q_table
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
                    print(f"  [CONGESTION / TIMEOUT] Episode {episode + 1} aborted at step {step + 1}: {e}")
                    print("  -> Resetting simulation and starting next episode...")
                    Steps_needed[episode] = max_steps
                    break
                
                # Q-update for (s, a)
                self.N_sa[s, a_idx] += 1
                best_next_q = np.max(self.q_table[s_next])
                td_error = (reward + self.gamma * best_next_q) - self.q_table[s, a_idx]
                self.q_table[s, a_idx] += alpha * td_error
                
                # Record full step-by-step history
                b1rel, b2rel = current_state.to_state()
                self.q_history.append({
                    "episode": episode + 1,
                    "step": step + 1,
                    "b1": b1rel,
                    "b2": b2rel,
                    "action": action,
                    "q1": self.q_table[s, 0],
                    "q2": self.q_table[s, 1],
                    "q3": self.q_table[s, 2],
                })

                # Overwrite current state for next step
                current_state = next_state
                
                # End episode if goal is reached
                if self.problem.is_goal_state(current_state):
                    Steps_needed[episode] = step + 1
                    print(f"  [SUCCESS] Episode {episode + 1} completed in {step + 1} steps (Drain={current_state.drain_total}, Time={current_state.sim_time_str})")
                    break
                
                # Record max_steps if goal was not reached within step limit
                if step == max_steps - 1:
                    Steps_needed[episode] = max_steps
        
        return Steps_needed

    def save_q_table(self, file):
        np.save(file, self.q_table)
    
    def save_q_history(self, filepath):
        import csv
        if not self.q_history:
            return
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["episode", "step", "b1", "b2", "action", "q1", "q2", "q3"])
            writer.writeheader()
            writer.writerows(self.q_history)
        print(f"[DATA EXPORT] Full Q-history ({len(self.q_history)} steps) saved to '{filepath}'.")

    def load_q_table(self, file):
        self.q_table = np.load(file)
