"""
visualization.py
----------------
Data recording and post-simulation plotting for Plant Simulation.
Records throughput data silently during simulation, exports data to CSV in 'data/',
and renders English graphs in 'graph/'. Supports offline plotting from CSV.
"""

import os
import csv
import glob
import datetime
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Default directory structure under python/
PYTHON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PYTHON_DIR, "data")
GRAPH_DIR = os.path.join(PYTHON_DIR, "graph")


def format_time_ticks(x, pos):
    """
    Formats seconds on the X-axis strictly as Hours:Minutes (hh:mm).
    """
    if x < 0:
        return "00:00"
    total_sec = int(round(x))
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    return f"{hours:02d}:{minutes:02d}"


class SimulationPlotter:
    """
    Collects simulation data silently without live GUI overhead.
    Generates high-resolution English plots and exports data to CSV after simulation.
    """

    def __init__(self, title="Simple Reflex Agent",
                 xlabel="Time (hh:mm)",
                 ylabel="Parts in Drain"):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

        self.records = []  # List of dicts: {"step": int, "sim_time": float, "drain_count": int}

    def record(self, sim_time, drain_count, step=None):
        """
        Records a single simulation data point silently.
        :param sim_time: Simulation time in seconds (float or int)
        :param drain_count: Total parts in drain (int)
        :param step: Optional decision step count (int)
        """
        if step is None:
            step = len(self.records)

        self.records.append({
            "step": step,
            "sim_time": float(sim_time),
            "drain_count": int(drain_count)
        })

    def update(self, sim_time, drain_count, step=None):
        """
        Alias for record() to maintain compatibility with existing simulation loops.
        """
        self.record(sim_time, drain_count, step=step)

    def save_csv(self, filepath=None):
        """
        Exports collected data to a CSV file in the 'data' folder using creation date timestamp.
        """
        if filepath is None:
            os.makedirs(DATA_DIR, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(DATA_DIR, f"{timestamp}.csv")
        else:
            dirname = os.path.dirname(filepath)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)

        fieldnames = ["step", "sim_time", "sim_time_str", "drain_count"]
        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in self.records:
                sim_time_str = format_time_ticks(rec["sim_time"], None)
                writer.writerow({
                    "step": rec["step"],
                    "sim_time": rec["sim_time"],
                    "sim_time_str": sim_time_str,
                    "drain_count": rec["drain_count"]
                })

        print(f"[DATA EXPORT] Saved {len(self.records)} records to CSV: '{filepath}'")
        return filepath

    def generate_plot(self, filepath=None, show=False):
        """
        Generates and saves a high-resolution figure in the 'graph' folder with English annotations.
        """
        if not self.records:
            print("[VISUALIZATION] Warning: No data records available to plot.")
            return None

        timesteps = [r["sim_time"] for r in self.records]
        drain_counts = [r["drain_count"] for r in self.records]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title(self.title, fontsize=14, fontweight='bold', pad=12)
        ax.set_xlabel(self.xlabel, fontsize=11)
        ax.set_ylabel(self.ylabel, fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.xaxis.set_major_formatter(FuncFormatter(format_time_ticks))

        ax.plot(
            timesteps, drain_counts,
            color="#1f77b4", linewidth=2.5, marker="o", markersize=4
        )

        if filepath is None:
            os.makedirs(GRAPH_DIR, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(GRAPH_DIR, f"{timestamp}.png")
        else:
            dirname = os.path.dirname(filepath)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)

        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"[VISUALIZATION] Plot successfully saved to: '{filepath}'")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return filepath

    def save_plot(self, filepath=None):
        """
        Alias for generate_plot() for backward compatibility.
        """
        return self.generate_plot(filepath=filepath)


# Backward compatibility alias
LivePlotter = SimulationPlotter


def plot_from_csv(csv_filepath=None,
                  output_image_path=None,
                  title="Simple Reflex Agent",
                  xlabel="Time (hh:mm)",
                  ylabel="Parts in Drain",
                  show=True):
    """
    Reads throughput data from a CSV file (defaults to latest in 'data/' folder)
    and generates a formatted plot saved in the 'graph/' folder.
    """
    # Auto-detect CSV file if not specified
    if csv_filepath is None:
        csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
        if not csv_files:
            # Fallback search in script directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            csv_files = glob.glob(os.path.join(script_dir, "*.csv"))

        if not csv_files:
            print(f"[ERROR] No CSV files found in '{DATA_DIR}'. Please run simulation first.")
            return

        # Select most recently modified CSV file
        csv_filepath = max(csv_files, key=os.path.getmtime)
        print(f"[OFFLINE PLOT] Selected latest CSV file: '{csv_filepath}'")

    if not os.path.exists(csv_filepath):
        print(f"[ERROR] CSV file not found: '{csv_filepath}'")
        return

    # Auto-derive output image path in 'graph/' folder if not specified
    if output_image_path is None:
        os.makedirs(GRAPH_DIR, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(csv_filepath))[0]
        output_image_path = os.path.join(GRAPH_DIR, f"{base_name}.png")
    else:
        dirname = os.path.dirname(output_image_path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

    sim_times = []
    drain_counts = []

    with open(csv_filepath, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sim_times.append(float(row["sim_time"]))
            drain_counts.append(int(row["drain_count"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.xaxis.set_major_formatter(FuncFormatter(format_time_ticks))

    ax.plot(
        sim_times, drain_counts,
        color="#1f77b4", linewidth=2.5, marker="o", markersize=4
    )

    fig.savefig(output_image_path, dpi=300, bbox_inches="tight")
    print(f"[OFFLINE PLOT] Plot saved to: '{output_image_path}'")

    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plot from offline simulation CSV data.")
    parser.add_argument("--csv", default=None, help="Path to input CSV file (default: latest in data/ folder)")
    parser.add_argument("--out", default=None, help="Path to output PNG image (default: inside graph/ folder)")
    parser.add_argument("--no-show", action="store_true", help="Do not display interactive plot window")
    args = parser.parse_args()

    show_window = not args.no_show
    plot_from_csv(csv_filepath=args.csv, output_image_path=args.out, show=show_window)



