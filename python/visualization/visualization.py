"""
visualization.py
----------------
Live-Visualisierung für Plant Simulation.
Plottet in Echtzeit die Anzahl der produzierten Teile im Drain (Y-Achse)
über die Timesteps / Schritte (X-Achse).
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def format_time_ticks(x, pos):
    """
    Formatiert Sekunden auf der X-Achse strikt im Format Minute:Sekunde (MM:SS) ohne Millisekunden.
    """
    if x < 0:
        return "00:00"
    total_sec = int(round(x))
    minutes = total_sec // 60
    seconds = total_sec % 60
    return f"{minutes:02d}:{seconds:02d}"


class LivePlotter:
    def __init__(self, title="Plant Simulation - Live Durchsatz", 
                 xlabel="Simulationszeit (Minute:Sekunde)", ylabel="Anzahl Teile im Drain"):
        """
        Initialisiert das interaktive Matplotlib-Diagramm.
        """
        self.timesteps = []
        self.drain_counts = []

        # Interaktiven Modus aktivieren
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        
        # Design & Layout
        self.fig.canvas.manager.set_window_title(title)
        self.ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
        self.ax.set_xlabel(xlabel, fontsize=11)
        self.ax.set_ylabel(ylabel, fontsize=11)
        self.ax.grid(True, linestyle="--", alpha=0.6)

        # X-Achsen Formatter auf MM:SS setzen
        self.ax.xaxis.set_major_formatter(FuncFormatter(format_time_ticks))

        # Plot-Linie initialisieren
        (self.line,) = self.ax.plot(
            [], [], color="#1f77b4", linewidth=2.5, marker="o", markersize=4, label="Teile im Drain"
        )
        self.ax.legend(loc="upper left")

    def update(self, timestep, drain_count):
        """
        Fügt einen neuen Datenpunkt hinzu und aktualisiert den Plot live.
        :param timestep: Aktueller Schritt / Timestep (int)
        :param drain_count: Aktuelle Gesamtzahl der Teile im Drain (int)
        """
        self.timesteps.append(timestep)
        self.drain_counts.append(drain_count)

        # Daten der Linie aktualisieren
        self.line.set_data(self.timesteps, self.drain_counts)

        # Achsen automatisch anpassen
        self.ax.relim()
        self.ax.autoscale_view()

        # Zeichnen ohne zu blockieren
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def save_plot(self, filepath="python/visualization/live_throughput.png"):
        """
        Speichert das aktuelle Diagramm als hochauflösende Bilddatei (PNG).
        :param filepath: Pfad zum Speichern des Bildes
        """
        import os
        dirname = os.path.dirname(filepath)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        self.fig.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Diagramm erfolgreich gespeichert unter: '{filepath}'")

    def close(self):
        """Schließt das Diagramm-Fenster."""
        plt.ioff()
        plt.close(self.fig)

    def keep_open(self):
        """Hält das Fenster nach Ablauf geöffnet."""
        plt.ioff()
        plt.show()
