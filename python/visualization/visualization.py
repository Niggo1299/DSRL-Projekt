"""
visualization.py
----------------
Visualizes results from Q-learning training and evaluation test runs:
1. Learning curve (steps per episode)
2. Q-value convergence for a predefined state
3. Full Q-table rendered as a formatted graphic table
4. Simulation throughput (parts in drain over simulation time)

All graphics are saved to the 'graph/' directory and overwritten on each execution.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ============================ Paths & Settings ============================
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data")
AGENT_DIR = os.path.join(BASE_DIR, "agent")
GRAPH_DIR = os.path.join(BASE_DIR, "graph")

STEPS_CSV      = os.path.join(DATA_DIR, "training_steps.csv")
HISTORY_CSV    = os.path.join(DATA_DIR, "q_history.csv")
THROUGHPUT_CSV = os.path.join(DATA_DIR, "throughput.csv")
Q_TABLE_FILE   = os.path.join(AGENT_DIR, "q_table.npy")

# Predefined state for Q-value convergence analysis (0=Empty, 1=Match, 2=Mismatch)
PREDEFINED_STATE = (1, 0)
# ==========================================================================


def plot_learning_curve(csv_path, output_path):
    """Plot 1: Steps per episode (learning curve)."""
    if not os.path.exists(csv_path):
        print(f"[WARNING] Training steps file '{csv_path}' not found. Skipping Plot 1.")
        return

    data = pd.read_csv(csv_path)
    if "episode" not in data.columns or "steps_needed" not in data.columns:
        print(f"[WARNING] Columns 'episode'/'steps_needed' missing in '{csv_path}'.")
        return

    plt.figure(figsize=(9, 5))
    plt.plot(data["episode"], data["steps_needed"], color="#1f77b4", linewidth=1.8, marker="o", markersize=3)
    plt.title("Learning Curve: Steps per Episode", fontsize=13, fontweight="bold")
    plt.xlabel("Episode", fontsize=11)
    plt.ylabel("Steps needed to reach goal", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[PLOT] Learning curve successfully saved to '{output_path}'.")


def plot_q_convergence(csv_path, output_path, target_state=(1, 0)):
    """Plot 2: Q-value development for a predefined state."""
    if not os.path.exists(csv_path):
        print(f"[WARNING] Q-history file '{csv_path}' not found. Skipping Plot 2.")
        return

    df = pd.read_csv(csv_path)
    required_cols = {"b1", "b2", "q1", "q2", "q3"}
    if not required_cols.issubset(df.columns):
        print(f"[WARNING] Required columns {required_cols} missing in '{csv_path}'.")
        return

    # Filter for predefined state
    sub_df = df[(df["b1"] == target_state[0]) & (df["b2"] == target_state[1])].copy()
    if sub_df.empty:
        print(f"[WARNING] No data points found for state {target_state} in '{csv_path}'.")
        return

    sub_df["trial"] = range(1, len(sub_df) + 1)
    rel_map = {0: "Empty", 1: "Match", 2: "Mismatch"}
    state_label = f"({target_state[0]}, {target_state[1]}) [B1={rel_map.get(target_state[0])}, B2={rel_map.get(target_state[1])}]"

    plt.figure(figsize=(9, 5.5))
    plt.plot(sub_df["trial"], sub_df["q1"], label="Action 1: Buffer 1", color="#1f77b4", linewidth=1.8)
    plt.plot(sub_df["trial"], sub_df["q2"], label="Action 2: Buffer 2", color="#ff7f0e", linewidth=1.8, linestyle="--")
    plt.plot(sub_df["trial"], sub_df["q3"], label="Action 3: Return Loop", color="#2ca02c", linewidth=2.0)

    plt.title(f"Q-Value Convergence for State {state_label}", fontsize=12, fontweight="bold")
    plt.xlabel(f"State Visits / Decisions in State {target_state}", fontsize=11)
    plt.ylabel("Q-Value", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best", framealpha=0.9, fontsize=10)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[PLOT] Q-value convergence successfully saved to '{output_path}'.")


def plot_q_table_graphic(q_table_path, output_path):
    """Plot 3: Full Q-table for all 9 states as a formatted graphic table."""
    if not os.path.exists(q_table_path):
        print(f"[WARNING] Q-table file '{q_table_path}' not found. Skipping Plot 3.")
        return

    q_table = np.load(q_table_path)
    rel_names = {0: "Empty", 1: "Match", 2: "Mismatch"}
    act_names = {0: "Buffer 1 (1)", 1: "Buffer 2 (2)", 2: "Return (3)"}

    # 9 relative states: (b1rel, b2rel) with b1, b2 in [0, 1, 2]
    states = [(b1, b2) for b1 in [0, 1, 2] for b2 in [0, 1, 2]]

    headers = [
        "State Index",
        "Buffer 1",
        "Buffer 2",
        "Q(Buffer 1)",
        "Q(Buffer 2)",
        "Q(Return Loop)",
        "Best Action"
    ]

    table_content = []
    cell_colors = []

    for idx, (b1, b2) in enumerate(states):
        q0 = float(q_table[idx, 0])
        q1 = float(q_table[idx, 1])
        q2 = float(q_table[idx, 2])
        best_act_idx = int(np.argmax([q0, q1, q2]))

        row_text = [
            str(idx),
            f"{rel_names[b1]} ({b1})",
            f"{rel_names[b2]} ({b2})",
            f"{q0:.2f}",
            f"{q1:.2f}",
            f"{q2:.2f}",
            act_names[best_act_idx]
        ]
        table_content.append(row_text)

        # Zebra striping with subtle background tints
        base_color = "#f9fbfd" if idx % 2 == 1 else "#ffffff"
        row_colors = [base_color] * len(row_text)
        # Highlight best action in light green
        row_colors[6] = "#d4edda"
        cell_colors.append(row_colors)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axis("off")
    ax.set_title("Trained Q-Table (9 States x 3 Actions)", fontsize=13, fontweight="bold", pad=15)

    col_widths = [0.12, 0.16, 0.16, 0.14, 0.14, 0.14, 0.16]
    tab = ax.table(
        cellText=table_content,
        colLabels=headers,
        colWidths=col_widths,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center"
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    tab.scale(1.0, 1.8)

    # Style header row (dark blue with white text)
    for (i, j), cell in tab.get_celld().items():
        if i == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor((21/255, 96/255, 130/255))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Q-table graphic successfully saved to '{output_path}'.")


def plot_simulation_throughput(csv_path, output_path):
    """Plot 4: Simulation throughput (parts in drain over simulation time)."""
    if not os.path.exists(csv_path):
        print(f"[WARNING] Throughput file '{csv_path}' not found. Skipping throughput plot.")
        return

    data = pd.read_csv(csv_path)
    if "sim_time" not in data.columns or "drain_count" not in data.columns:
        print(f"[WARNING] Required columns 'sim_time'/'drain_count' missing in '{csv_path}'.")
        return

    plt.figure(figsize=(9.5, 5))
    plt.plot(data["sim_time"], data["drain_count"], color="#1f77b4", linewidth=2.0, label="Parts in Drain")

    # Format X-axis in HH:MM
    formatter = FuncFormatter(lambda x, _: f"{int(max(0, x)) // 3600:02d}:{(int(max(0, x)) % 3600) // 60:02d}")
    plt.gca().xaxis.set_major_formatter(formatter)

    plt.title("Simulation Throughput: Parts in Drain over Time", fontsize=13, fontweight="bold")
    plt.xlabel("Simulation Time (hh:mm)", fontsize=11)
    plt.ylabel("Parts in Drain", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper left", framealpha=0.9, fontsize=10)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[PLOT] Simulation throughput successfully saved to '{output_path}'.")


def main():
    os.makedirs(GRAPH_DIR, exist_ok=True)

    out_curve      = os.path.join(GRAPH_DIR, "steps_per_episode.png")
    out_q_conv     = os.path.join(GRAPH_DIR, f"q_convergence_state_{PREDEFINED_STATE[0]}_{PREDEFINED_STATE[1]}.png")
    out_q_tab      = os.path.join(GRAPH_DIR, "q_table.png")
    out_throughput = os.path.join(GRAPH_DIR, "simulation_throughput.png")

    print("\n--- Generating Visualizations ---")
    plot_learning_curve(STEPS_CSV, out_curve)
    plot_q_convergence(HISTORY_CSV, out_q_conv, target_state=PREDEFINED_STATE)
    plot_q_table_graphic(Q_TABLE_FILE, out_q_tab)
    plot_simulation_throughput(THROUGHPUT_CSV, out_throughput)
    print("--- All visualizations completed ---\n")


if __name__ == "__main__":
    main()
