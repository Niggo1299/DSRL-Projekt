"""
main.py
-------
Initialisiert Plant Simulation und den Agenten und fuehrt Episoden aus.
"""

import sys
import time

from plantsim.plantsim import Plantsim
from agent.agent import Agent, SimulationFailedError

# ---------------- Konfiguration ----------------
MODEL_PATH = (
    r"C:\Users\Niko\OneDrive - Fachhochschule Bielefeld"
    r"\Diskrete Simulation und Reinforcement Learning\Projekt\plant\plantmodel.spp"
)
PLANTSIM_VERSION = "16.1"
CONTEXT          = ".Modelle.Modell"
POLL_INTERVAL    = 0.002   # s - billiger COM-Call waehrend die Sim pausiert
TIMEOUT          = 30.0    # s - max. Wartezeit auf einen Entscheidungspunkt
MANUAL_CONTROL   = True    # True = User steuert, False = Agent steuert
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


def run_episode(ps, agent):
    """Fuehrt eine Episode aus. Reguläres Ende: Usereingabe 'q'."""
    ps.reset_simulation()
    agent.reset()
    ps.start_simulation()

    step = 0
    while True:
        # 1. Warten, bis ein neuer Zustand ansteht
        wait_for_decision(ps)

        # 2. Zustand auslesen
        state = agent.get_state()

        # 3. Aktion bestimmen (User oder Agent)
        action = agent.select_action(state)
        if action is None:
            print("  Episode durch Benutzer beendet.")
            break

        # 4. Aktion schreiben + Simulation freigeben (Handshake)
        agent.set_action(action)

        # 5. Simulation fortsetzen (RESUME)
        ps.start_simulation()

        step += 1
        print(f"  -> Step {step}: Action = {action} freigegeben.")


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

    exit_code = 0
    try:
        while True:
            cmd = input(
                "\n[ENTER] = neue Episode starten, 'q' = Programm beenden: "
            ).strip().lower()
            if cmd == "q":
                break
            run_episode(ps, agent)

    except SimulationFailedError as e:
        print(f"\n[ABBRUCH] simulation failed: {e}")
        exit_code = 1

    finally:
        ps.quit()          # Plant Simulation schliessen

    sys.exit(exit_code)


if __name__ == "__main__":
    main()