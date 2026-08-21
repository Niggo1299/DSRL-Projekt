"""
main.py
-------
Initialisiert Plant Simulation und das Problem-Environment.
Führt Episoden mit dem gewählten Agenten aus (Standard: Simple Reflex Agent).
"""

from win32com.client import gencache
import sys

from plantsim.plantsim import Plantsim
from problem.problem import PlantSimulationProblem, SimulationFailedError
from agent.agent import ReflexAgent, QLearningAgent, TrainingTestAgent
from visualization.visualization import SimulationPlotter, save_and_plot_training

# ============================ Konfiguration ============================
MODEL_PATH = (
    r"C:\Users\Niko\OneDrive - Fachhochschule Bielefeld"
    r"\Diskrete Simulation und Reinforcement Learning\Projekt\plant\plantmodel.spp"
)
PLANTSIM_VERSION   = "16.1"
CONTEXT            = ".Modelle.Modell"
POLL_INTERVAL      = 0.002   # s - Abfrageintervall für StateReady
TIMEOUT            = 30.0    # s - max. Wartezeit auf einen Entscheidungspunkt

# Agenten-Auswahl: "reflex", "manual", "q_learning", "training_test"
AGENT_TYPE         = "reflex"

SAVE_CSV_DATA      = True    # True = Ergebnisse nach Episode als CSV speichern
TARGET_DRAIN_COUNT = 1000    # Zielanzahl Teile im Drain
# =======================================================================


def create_agent(agent_type, problem):
    """Erzeugt den gewünschten Agenten basierend auf der Konfiguration."""
    if agent_type == "reflex":
        return ReflexAgent(problem, manual_control=False)
    elif agent_type == "manual":
        return ReflexAgent(problem, manual_control=True)
    elif agent_type == "q_learning":
        return QLearningAgent(problem)
    elif agent_type == "training_test":
        return TrainingTestAgent(problem)
    else:
        raise ValueError(f"Unbekannter Agenten-Typ: {agent_type}")


def run_episode(problem, agent, plotter=None):
    """Führt eine Episode mit dem konfigurierten Agenten aus."""
    print(f"\n--- Starte Episode mit Agent '{AGENT_TYPE}' ---")
    state = problem.reset()

    # Initialer Datenpunkt
    if plotter:
        plotter.record(state.sim_time, state.drain_total, step=0)

    step = 0
    act_names = {1: "Buffer 1", 2: "Buffer 2", 3: "Return"}

    while not problem.is_goal_state(state):
        # 1. Aktion durch Agenten bestimmen
        action = agent.act()
        if action is None:
            print("  Episode durch Benutzer beendet.")
            break

        step += 1
        act_str = act_names.get(action, str(action))
        print(
            f"  -> Step {step}: State={state.to_state()} => Aktion={act_str} "
            f"(SimTime={state.sim_time_str}, Drain={state.drain_total}/{TARGET_DRAIN_COUNT if TARGET_DRAIN_COUNT else 'inf'})"
        )

        # 2. Aktion an Plant Simulation übergeben (Handshake)
        problem.act(action)

        # 3. Neuen Zustand abfragen
        state = problem.get_current_state()

        # 4. Datenpunkt für Plotter/CSV speichern
        if plotter:
            plotter.record(state.sim_time, state.drain_total, step=step)

    if problem.is_goal_state(state):
        print(f"\n[ZIEL ERREICHT] {state.drain_total} / {TARGET_DRAIN_COUNT} Teile produziert in {state.sim_time_str}!")


def main():
    # 1. Plant Simulation initialisieren
    ps = Plantsim(
        path_context=CONTEXT,
        model=MODEL_PATH,
        version=PLANTSIM_VERSION,
        visible=False,
        trust_models=True,
        license_type="Educational",
    )
    ps.set_event_controller()

    # 2. Problem-Environment erzeugen
    problem = PlantSimulationProblem(
        plantsim=ps,
        target_drain_count=TARGET_DRAIN_COUNT,
        poll_interval=POLL_INTERVAL,
        timeout=TIMEOUT,
    )

    # 3. Agenten instanziieren
    agent = create_agent(AGENT_TYPE, problem)
    if isinstance(agent, ReflexAgent) and not agent.manual_control:
        agent.print_table()

    plotter = SimulationPlotter() if SAVE_CSV_DATA else None
    exit_code = 0

    try:
        if AGENT_TYPE == "q_learning":
            print("\n[RL-TRAINING] Starte Q-Learning Trainingslauf...")
            steps = agent.train(episodes=100, alpha=0.1, max_steps=3000, gamma = 0.99, max_N_exploration = 3   , R_Max = 2000)   
            agent.save_q_table("q_table.npy")
            print(f"[RL-TRAINING] Training abgeschlossen. Schritte je Episode: {steps}")
            save_and_plot_training(steps, agent_type=AGENT_TYPE)
        else:
            while True:
                cmd = input("\n[ENTER] = neue Episode starten, 'q' = Programm beenden: ").strip().lower()
                if cmd == "q":
                    break
                run_episode(problem, agent, plotter)
                if plotter:
                    plotter.save_csv()

    except SimulationFailedError as e:
        print(f"\n[ABBRUCH] Simulation fehlgeschlagen: {e}")
        exit_code = 1

    except KeyboardInterrupt:
        print("\n[INFO] Programm durch Benutzer mit Strg+C unterbrochen.")
        if AGENT_TYPE == "q_learning" and 'agent' in locals():
            agent.save_q_table("q_table.npy")
            print("[INFO] Aktuelle Q-Tabelle wurde gesichert.")

    finally:
        if plotter and plotter.records:
            plotter.save_csv()
        ps.quit()  # Plant Simulation sauber beenden

    sys.exit(exit_code)


if __name__ == "__main__":
    main()