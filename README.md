# Discrete Simulation and Reinforcement Learning – Project

**Topic:** Autonomous sorting robot in a high-bay storage system with two buffer stores and return loop  
**Environment:** Siemens Tecnomatix Plant Simulation (16.1) via Python COM interface  
**Institution:** Bielefeld University of Applied Sciences and Arts (HSBI)

---

## 1. Overview

This project implements an autonomous sorting robot using **tabular Q-learning with GLIE exploration** (*Greedy in the Limit with Infinite Exploration*) to sort incoming parts into two buffer stores in Tecnomatix Plant Simulation.

Parts of three types (1, 2, 3) arrive randomly. When a buffer collects **10 matching parts**, it empties into the **Drain**. Mismatched parts are redirected into the **Return Loop**. The objective is to sort and drain **1,000 parts** with minimal steps and simulation time.

```text
                  ┌────────────────┐
                  │ Source / Store │ (Type 1, 2, 3)
                  └───────┬────────┘
                          │
                          ▼
                    ┌───────────┐
                    │   Robot   │ ◄─────── Return Loop
                    └─────┬─────┘                   ▲
           ┌──────────────┼──────────────┐          │
           ▼              ▼              ▼          │
     ┌───────────┐  ┌───────────┐  ┌───────────┐    │
     │ Buffer 1  │  │ Buffer 2  │  │  Return   │────┘
     │  (max 10) │  │  (max 10) │  └───────────┘
     └─────┬─────┘  └─────┬─────┘
           └──────┬───────┘
                  ▼
              ┌───────┐
              │ Drain │ (Goal: 1,000 parts)
              └───────┘
```

---

## 2. Problem Formulation (MDP)

- **State Space (9 relative states):**
  0 = Empty, 1 = Match, 2 = Mismatch relative to the incoming part.
- **Action Space (3 actions):**
  1 = Buffer 1, 2 = Buffer 2, 3 = Return Loop.
- **Reward Function:**
  - Step penalty: -1 per step until goal is reached.
  - Throughput reward: +100 per part entering the drain.
  - Buffer placement: +n for buffering the n-th matching part.
  - Buffer loss: n(n+1)/2 if parts leave the buffer without draining.

## 3. Project Structure

```text
Projekt/
├── Doku/                             # Project assignment & presentation slides
│   ├── Project.pdf
│   ├── presentation.pdf
│   └── presentation.pptx
│
├── plant/                            # Simulation model
│   └── plantmodel.spp
│
├── python/                           # Source code
│   ├── main.py                       # Main execution script (train & test)
│   ├── agent/                        # Base Agent & QLearningAgent (q_table.npy)
│   ├── problem/                      # Environment interface & COM handshake
│   ├── plantsim/                     # COM wrapper for Plant Simulation
│   ├── data/                         # CSV logs (training_steps, q_history, throughput)
│   ├── graph/                        # Generated evaluation plots (PNG)
│   └── visualization/                # Plotting script (visualization.py)
│
└── README.md                         # Project documentation
```

---

## 4. Quickstart

### Running the Simulation (`main.py`)

Select mode at the top of `python/main.py`:

```python
MODE = "train"  # "train" = run Q-learning, "test" = evaluate learned Q-table
SAVE_CSV_DATA = True
```

Run:
```powershell
python python/main.py
```

- **`"train"` mode:** Runs tabular Q-learning, saves the trained policy to `python/agent/q_table.npy`, and exports `python/data/training_steps.csv` and `python/data/q_history.csv`.
- **`"test"` mode:** Loads `python/agent/q_table.npy`, runs a greedy evaluation episode (to 1,000 parts), and exports `python/data/throughput.csv`.

---

### Generating Visualizations (`visualization.py`)

```powershell
python python/visualization/visualization.py
```

Generates four figures in `python/graph/`:
1. **`steps_per_episode.png`:** Learning curve (steps required per episode).
2. **`q_convergence_state_1_0.png`:** Q-value convergence curves for state `(1, 0)`.
3. **`q_table.png`:** Complete formatted Q-table showing values and optimal actions.
4. **`simulation_throughput.png`:** Drained parts over simulation time ($hh:mm$).
