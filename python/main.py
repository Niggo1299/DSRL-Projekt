"""
main.py
-------
Initializes Tecnomatix Plant Simulation and the problem environment.
Executes Q-learning training or evaluates a test run using the learned Q-table.
"""

import os
import sys
import csv

from plantsim.plantsim import Plantsim
from problem.problem import PlantSimulationProblem, SimulationFailedError
from agent.agent import QLearningAgent

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
AGENT_DIR = os.path.join(BASE_DIR, "agent")

# Fixed file paths (overwritten on each training or test run)
Q_TABLE_FILE   = os.path.join(AGENT_DIR, "q_table.npy")
STEPS_CSV      = os.path.join(DATA_DIR, "training_steps.csv")
HISTORY_CSV    = os.path.join(DATA_DIR, "q_history.csv")
THROUGHPUT_CSV = os.path.join(DATA_DIR, "throughput.csv")

# ============================ Configuration ============================
MODEL_PATH = (
    r"C:\Users\Niko\OneDrive - Fachhochschule Bielefeld"
    r"\Diskrete Simulation und Reinforcement Learning\Projekt\plant\plantmodel.spp"
)
PLANTSIM_VERSION   = "16.1"
CONTEXT            = ".Modelle.Modell"
POLL_INTERVAL      = 0.002   # s - Polling interval for StateReady
TIMEOUT            = 30.0    # s - Maximum waiting time for decision point

# Mode: "train" (run Q-learning training) or "test" (evaluate learned Q-table)
MODE               = "test"

# Enable CSV data export for training and test results (overwrites existing files)
SAVE_CSV_DATA      = True

# Hyperparameters for training
EPISODES           = 10
MAX_STEPS          = 3000
ALPHA              = 0.1
GAMMA              = 0.99
MAX_N_EXPLORATION  = 10
R_MAX              = 2000

TARGET_DRAIN_COUNT = 1000    # Target number of parts in drain
# =======================================================================


def save_training_steps_csv(steps_needed, filepath):
    """Saves steps per episode to CSV (overwrites existing file)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "steps_needed"])
        for ep, steps in enumerate(steps_needed, start=1):
            writer.writerow([ep, int(steps)])
    print(f"[DATA EXPORT] Training steps saved to '{filepath}'.")


def run_test_episode(problem, agent, save_csv=True, csv_path=THROUGHPUT_CSV):
    """Executes a test episode with the learned Q-table (greedy) and logs throughput."""
    print("\n--- Starting Test Episode with Learned Q-Table ---")
    state = problem.reset()
    step = 0
    act_names = {1: "Buffer 1", 2: "Buffer 2", 3: "Return"}

    records = []
    if save_csv:
        records.append({
            "step": step,
            "sim_time": float(state.sim_time),
            "sim_time_str": state.sim_time_str,
            "drain_count": int(state.drain_total)
        })

    while not problem.is_goal_state(state):
        action = agent.act()
        step += 1
        act_str = act_names.get(action, str(action))
        print(
            f"  -> Step {step}: State={state.to_state()} => Action={act_str} "
            f"(SimTime={state.sim_time_str}, Drain={state.drain_total}/{TARGET_DRAIN_COUNT})"
        )
        problem.act(action)
        state = problem.get_current_state()

        if save_csv:
            records.append({
                "step": step,
                "sim_time": float(state.sim_time),
                "sim_time_str": state.sim_time_str,
                "drain_count": int(state.drain_total)
            })

    if problem.is_goal_state(state):
        print(f"\n[GOAL REACHED] {state.drain_total} parts produced in {state.sim_time_str} ({step} steps)!")

    if save_csv and records:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "sim_time", "sim_time_str", "drain_count"])
            writer.writeheader()
            writer.writerows(records)
        print(f"[DATA EXPORT] Throughput data ({len(records)} data points) saved to '{csv_path}'.")


def main():
    # 1. Initialize Plant Simulation
    ps = Plantsim(
        path_context=CONTEXT,
        model=MODEL_PATH,
        version=PLANTSIM_VERSION,
        visible=True,
        trust_models=True,
        license_type="Educational",
    )
    ps.set_event_controller()

    # 2. Create problem environment
    problem = PlantSimulationProblem(
        plantsim=ps,
        target_drain_count=TARGET_DRAIN_COUNT,
        poll_interval=POLL_INTERVAL,
        timeout=TIMEOUT,
    )

    # 3. Instantiate Q-learning agent
    agent = QLearningAgent(problem)
    exit_code = 0

    try:
        if MODE == "train":
            print(f"\n[RL-TRAINING] Starting Q-Learning ({EPISODES} episodes, max. {MAX_STEPS} steps)...")
            steps = agent.train(
                episodes=EPISODES,
                alpha=ALPHA,
                max_steps=MAX_STEPS,
                gamma=GAMMA,
                max_N_exploration=MAX_N_EXPLORATION,
                R_Max=R_MAX,
            )
            # Save Q-table to agent directory
            agent.save_q_table(Q_TABLE_FILE)
            print(f"[RL-TRAINING] Q-table successfully saved to '{Q_TABLE_FILE}'.")

            # Overwrite CSV files if flag is set
            if SAVE_CSV_DATA:
                save_training_steps_csv(steps, STEPS_CSV)
                agent.save_q_history(HISTORY_CSV)

        elif MODE == "test":
            if not os.path.exists(Q_TABLE_FILE):
                print(f"[ERROR] Q-table not found at '{Q_TABLE_FILE}'. Please train first!")
                return
            agent.load_q_table(Q_TABLE_FILE)
            print(f"[INFO] Q-table loaded from '{Q_TABLE_FILE}'.")
            run_test_episode(problem, agent, save_csv=SAVE_CSV_DATA, csv_path=THROUGHPUT_CSV)

        else:
            print(f"[ERROR] Unknown mode: '{MODE}'. Allowed modes are 'train' or 'test'.")

    except SimulationFailedError as e:
        print(f"\n[ABORT] Simulation failed: {e}")
        exit_code = 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user (Ctrl+C).")
        if MODE == "train":
            agent.save_q_table(Q_TABLE_FILE)
            print(f"[INFO] Current Q-table saved to '{Q_TABLE_FILE}'.")
            if SAVE_CSV_DATA and agent.q_history:
                agent.save_q_history(HISTORY_CSV)

    finally:
        ps.quit()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()