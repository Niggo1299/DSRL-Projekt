"""
main.py
-------
Initialisiert Plant Simulation und den Reflex-Agenten und führt Episoden aus.
"""

import sys
import time

from plantsim.plantsim import Plantsim
from agent.agent import Agent, SimulationFailedError
from visualization.visualization import LivePlotter

# ---------------- Konfiguration ----------------
MODEL_PATH = (
    r"C:\Users\Niko\OneDrive - Fachhochschule Bielefeld"
    r"\Diskrete Simulation und Reinforcement Learning\Projekt\plant\plantmodel.spp"
)
PLANTSIM_VERSION   = "16.1"
CONTEXT            = ".Modelle.Modell"
POLL_INTERVAL      = 0.002   # s - billiger COM-Call waehrend die Sim pausiert
TIMEOUT            = 30.0    # s - max. Wartezeit auf einen Entscheidungspunkt

MANUAL_CONTROL     = False   # True = User steuert per Tastatur, False = Reflex-Agent steuert
USE_LIVE_PLOT      = True    # True = Live-Plotter (Teile im Drain / Timestep) anzeigen
TARGET_DRAIN_COUNT = 1000    # Abbrechen, wenn 1000 Teile im Drain sind (None = deaktiviert)
# -----------------------------------------------


def is_running(ps):
    """Nutzt die zugrunde liegende COM-Schnittstelle der Library."""
    return ps.plantsim.IsSimulationRunning()


def state_ready(ps):
    return bool(ps.get_value(Agent.CELL_STATE_READY))


def wait_for_decision(ps, timeout=TIMEOUT):
    """
    Wartet, bis StateReady == True (echter Entscheidungspunkt).

    :raises SimulationFailedError: Sim steht ohne StateReady oder Timeout.
    """
    t0 = time.time()
    time.sleep(0.05)                       # kurze Anlaufzeit fuer die Sim

    while True:
        if state_ready(ps):
            return

        if not is_running(ps) and not state_ready(ps):
            raise SimulationFailedError(
                "Simulation failed"
            )

        if time.time() - t0 > timeout:
            raise SimulationFailedError(
                f"Timeout ({timeout} s) beim Warten auf einen Entscheidungspunkt."
            )

        time.sleep(POLL_INTERVAL)


def run_episode(ps, agent, plotter=None):
    """Fuehrt eine Episode mit dem Reflex-Agenten aus."""
    ps.reset_simulation()
    agent.reset()
    ps.start_simulation()

    step = 0
    # Ersten Entscheidungspunkt abwarten
    wait_for_decision(ps)
    state, raw_state = agent.get_state()

    # Initialer Plot-Punkt
    if plotter:
        plotter.update(raw_state["sim_time"], raw_state["drain_total"])

    while True:
        # 1. Aktion aus Nachschlagetabelle bestimmen
        action = agent.select_action(state)
        if action is None:
            print("  Episode durch Benutzer beendet.")
            break

        # 2. Aktion schreiben + Simulation freigeben (Handshake)
        agent.set_action(action)
        ps.start_simulation()

        step += 1

        # 3. Warten, bis der naechste Entscheidungspunkt erreicht ist
        wait_for_decision(ps)
        next_state, next_raw_state = agent.get_state()

        act_name = {1: "Buffer 1", 2: "Buffer 2", 3: "Return"}.get(action, str(action))
        print(
            f"  -> Step {step}: State={state} => Reflex-Aktion={act_name} "
            f"(SimTime={next_raw_state['sim_time_str']}, Drain={next_raw_state['drain_total']}/{TARGET_DRAIN_COUNT if TARGET_DRAIN_COUNT else 'inf'})"
        )

        # 4. Live-Plot mit echter Plant-Simulation-Zeit aktualisieren
        if plotter:
            plotter.update(next_raw_state["sim_time"], next_raw_state["drain_total"])

        state, raw_state = next_state, next_raw_state

        # 5. Abbruchbedingung prüfen: Zielanzahl produzierter Teile erreicht?
        if TARGET_DRAIN_COUNT is not None and next_raw_state["drain_total"] >= TARGET_DRAIN_COUNT:
            print(f"\n[ZIEL ERREICHT] {next_raw_state['drain_total']} / {TARGET_DRAIN_COUNT} Teile produziert.")
            if plotter:
                plotter.save_plot()
            break


def main():
    ps = Plantsim(
        path_context=CONTEXT,
        model=MODEL_PATH,
        version=PLANTSIM_VERSION,
        visible=True,
        trust_models=True,
        license_type="Educational",
    )
    ps.set_event_controller()

    agent = Agent(ps, manual_control=MANUAL_CONTROL)
    agent.print_table()

    plotter = LivePlotter() if USE_LIVE_PLOT else None

    exit_code = 0
    try:
        while True:
            cmd = input(
                "\n[ENTER] = neue Episode starten, 'q' = Programm beenden: "
            ).strip().lower()
            if cmd == "q":
                break
            run_episode(ps, agent, plotter)
            if plotter:
                plotter.save_plot()

    except SimulationFailedError as e:
        print(f"\n[ABBRUCH] simulation failed: {e}")
        exit_code = 1

    finally:
        if plotter:
            plotter.save_plot()
        ps.quit()          # Plant Simulation schliessen

    sys.exit(exit_code)


if __name__ == "__main__":
    main()