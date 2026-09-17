import matplotlib.pyplot as plt
import numpy as np 
import csv

# ==============================================================================
# 1. Plot: Q-Learning Basis Konfiguration
# ==============================================================================
path = r"02_Experimente\01_QLearning_Basis\20260820_231448_q_learning.csv"
data = np.genfromtxt(path, delimiter=",", names=True)

plt.figure()
plt.plot(data["episode"], data["steps_needed"])
plt.xlabel("Episodes")
plt.ylabel("Steps needed to reach goal")
plt.title("Learning Curve Q-Learning Basis Konfiguration")
plt.grid(True)

save_path = path.replace(".csv", ".png")
plt.savefig(save_path, dpi=300)
plt.close()


# ==============================================================================
# 2. Plot: Q-Learning Symmetrie Konfiguration
# ==============================================================================
path = r"02_Experimente\02_QLearning_Symmetrie\20260821_125057_q_learning.csv"
data = np.genfromtxt(path, delimiter=",", names=True)

plt.figure()
plt.plot(data["episode"], data["steps_needed"])
plt.xlabel("Episodes")
plt.ylabel("Steps needed to reach goal")
plt.title("Learning Curve Q-Learning Symmetrie Konfiguration")
plt.grid(True)

save_path = path.replace(".csv", ".png")
plt.savefig(save_path, dpi=300)
plt.close()


# ==============================================================================
# 3. Plot: Q-Learning Relative Typen Konfiguration
# ==============================================================================
path = r"02_Experimente\03_QLearning_RelativeTypen\20260821_200047_q_learning.csv"
data = np.genfromtxt(path, delimiter=",", names=True)

plt.figure()
plt.plot(data["episode"], data["steps_needed"])
plt.xlabel("Episodes")
plt.ylabel("Steps needed to reach goal")
plt.title("Learning Curve Q-Learning Relative Typen Konfiguration")
plt.grid(True)

save_path = path.replace(".csv", ".png")
plt.savefig(save_path, dpi=300)
plt.close()


# ==============================================================================
# 4. Plot: Q-Learning LastCollect Konfiguration
# ==============================================================================
path = r"02_Experimente\04_QLearning_LastCollect\20260821_235156_q_learning.csv"
data = np.genfromtxt(path, delimiter=",", names=True)

plt.figure()
plt.plot(data["episode"], data["steps_needed"])
plt.xlabel("Episodes")
plt.ylabel("Steps needed to reach goal")
plt.title("Learning Curve Q-Learning LastCollect Konfiguration")
plt.grid(True)

save_path = path.replace(".csv", ".png")
plt.savefig(save_path, dpi=300)
plt.close()


# ==============================================================================
# 5. Plot: Q-Learning Minimal Konfiguration
# ==============================================================================
path = r"02_Experimente\05_QLearning_Minimal\20260822_125541_q_learning.csv"
data = np.genfromtxt(path, delimiter=",", names=True)

plt.figure()
plt.plot(data["episode"], data["steps_needed"])
plt.xlabel("Episodes")
plt.ylabel("Steps needed to reach goal")
plt.title("Learning Curve Q-Learning Minimal Konfiguration")
plt.grid(True)

save_path = path.replace(".csv", ".png")
plt.savefig(save_path, dpi=300)
plt.close()

# ==============================================================================
# 6. Vergleich: ReflexAgent mit vs. ohne LastCollect (Endphase / Zoom)
# ==============================================================================
from matplotlib.ticker import FuncFormatter

path_mit = r"02_Experimente\00_Baseline_ReflexAgent\20260822_mit_last_collect.csv"
path_ohne = r"02_Experimente\00_Baseline_ReflexAgent\20260822_ohne_last_collect.csv"

data_mit = np.genfromtxt(path_mit, delimiter=",", names=True)
data_ohne = np.genfromtxt(path_ohne, delimiter=",", names=True)

plt.figure(figsize=(10, 5))
plt.plot(data_mit["sim_time"], data_mit["drain_count"], label="With LastCollect (End: 432:12)", color="#1f77b4", linewidth=2.0, marker="o", markersize=4)
plt.plot(data_ohne["sim_time"], data_ohne["drain_count"], label="Without LastCollect (End: 429:02)", color="#d62728", linewidth=2.0, linestyle="--", marker="s", markersize=4)

plt.xlabel("Simulationtime (hh:mm)")
plt.ylabel("Items in Drain")
plt.title("Performance Compare ReflexAgent",fontsize=14, fontweight="bold")
plt.grid(True)
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(max(0, x)) // 3600:02d}:{(int(max(0, x)) % 3600) // 60:02d}"))

# Zoom: Die letzten 50 Stunden vor Zielerreichung (ca. 379:00 bis zum Ende)
first_finish = min(data_mit["sim_time"].max(), data_ohne["sim_time"].max())
start_time = first_finish - (50 * 3600)  # 50 Stunden vor dem ersten Finish
max_time = max(data_mit["sim_time"].max(), data_ohne["sim_time"].max()) + 1800

plt.xlim(start_time, max_time)
plt.ylim(880, 1005)

plt.legend()

save_path_comp = r"02_Experimente\00_Baseline_ReflexAgent\reflex_agent_comparison_zoom.png"
plt.savefig(save_path_comp, dpi=300, bbox_inches="tight")
plt.close()

# ==============================================================================
# 7. Säulendiagramm-Vergleich: ReflexAgent (Steps & Simulationszeit)
# ==============================================================================
steps_mit = int(data_mit["step"][-1])
steps_ohne = int(data_ohne["step"][-1])

time_sec_mit = float(data_mit["sim_time"][-1])
time_sec_ohne = float(data_ohne["sim_time"][-1])

time_hrs_mit = time_sec_mit / 3600
time_hrs_ohne = time_sec_ohne / 3600

time_str_mit = f"{int(time_sec_mit)//3600:02d}:{(int(time_sec_mit)%3600)//60:02d}"
time_str_ohne = f"{int(time_sec_ohne)//3600:02d}:{(int(time_sec_ohne)%3600)//60:02d}"

agents = ["Without LastCollect", "With LastCollect"]
colors = ["#d62728", "#1f77b4"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

# 1. Subplot: Benötigte Schritte (Y-Achse ab Step 1920)
bars1 = ax1.bar(agents, [steps_ohne, steps_mit], color=colors, width=0.45)
ax1.set_title("Required Decision Steps", fontsize=12, fontweight="bold")
ax1.set_ylabel("Steps")
ax1.set_ylim(1920, 1985)
ax1.grid(axis="y", linestyle="--", alpha=0.7)
ax1.bar_label(bars1, fmt="%d Steps", padding=3, fontweight="bold")

# 2. Subplot: Benötigte Simulationszeit (Y-Achse ab Stunde 420)
bars2 = ax2.bar(agents, [time_hrs_ohne, time_hrs_mit], color=colors, width=0.45)
ax2.set_title("Required Simulation Time", fontsize=12, fontweight="bold")
ax2.set_ylabel("Time in Hours (h)")
ax2.set_ylim(420, 436)
ax2.grid(axis="y", linestyle="--", alpha=0.7)
ax2.bar_label(bars2, labels=[f"{time_str_ohne} ({time_hrs_ohne:.1f}h)", f"{time_str_mit} ({time_hrs_mit:.1f}h)"], padding=3, fontweight="bold")

fig.suptitle("ReflexAgent Comparison: Steps & Simulation Time", fontsize=14, fontweight="bold")
fig.tight_layout()

save_path_bar = r"02_Experimente\00_Baseline_ReflexAgent\reflex_agent_bar_comparison.png"
fig.savefig(save_path_bar, dpi=300)
plt.close(fig)

# ==============================================================================
# 8. Q-Table Excerpt: Suboptimal Policy / Reward Farming (Baseline Model)
# ==============================================================================
path_q = r"02_Experimente\01_QLearning_Basis\q_table.csv"
with open(path_q, mode="r", encoding="utf-8") as f:
    q_rows = list(csv.DictReader(f))

# Definition of state dimensions: (inc, b1typ, b2typ, b1cat, b2cat)
incs = [1, 2, 3]
b1typs = [0, 1, 2, 3]
b2typs = [0, 1, 2, 3]
lang_map = {0: "Empty", 1: "DE", 2: "EN", 3: "ES"}
cat_labels = ["0-3", "4-6", "7-9"]

def decode_state(idx):
    rem = idx
    b2c = cat_labels[rem % 3]; rem //= 3
    b1c = cat_labels[rem % 3]; rem //= 3
    b2t = lang_map[b2typs[rem % 4]]; rem //= 4
    b1t = lang_map[b1typs[rem % 4]]; rem //= 4
    inc = lang_map[incs[rem % 3]]
    return inc, b1t, b2t, b1c, b2c

selected_indices = [60, 61, 62, 353, 356, 359]
table_content = []
cell_colors = []

for row_idx, idx in enumerate(selected_indices):
    r = q_rows[idx]
    inc, b1t, b2t, b1c, b2c = decode_state(idx)
    q0 = float(r["Action_0"])
    q1 = float(r["Action_1"])
    q2 = float(r["Action_2"])
    
    best_idx = max(range(3), key=lambda i: [q0, q1, q2][i])
    act_names = ["Buffer 1 (A0)", "Buffer 2 (A1)", "Return Loop (A2)"]
    chosen_act = act_names[best_idx]
    
    is_anomaly = (idx in [61, 353, 356, 359])
    status = "Reward Farming" if is_anomaly else "Optimal"
    
    b1_str = f"{b1t} ({b1c})" if b1t != "Empty" else "Empty"
    b2_str = f"{b2t} ({b2c})" if b2t != "Empty" else "Empty"
    
    row_text = [
        str(idx),
        inc,
        b1_str,
        b2_str,
        f"{q0:.2f}",
        f"{q1:.2f}",
        f"{q2:.2f}",
        chosen_act,
        status
    ]
    table_content.append(row_text)
    row_color = "#ffcccc" if is_anomaly else ("#f2f2f2" if row_idx % 2 == 1 else "#ffffff")
    cell_colors.append([row_color] * len(row_text))

headers = [
    "State Index",
    "Incoming Part",
    "Buffer 1 (Type / Fill)",
    "Buffer 2 (Type / Fill)",
    "Q(Buffer 1)",
    "Q(Buffer 2)",
    "Q(Return)",
    "Selected Action",
    "Evaluation"
]

fig, ax = plt.subplots(figsize=(15.5, 3.8))
ax.axis("off")
ax.set_title("Q-Table Excerpt: Learned Suboptimal Policy / Reward Farming (Baseline Model)", fontsize=13, fontweight="bold", pad=15)

tab = ax.table(
    cellText=table_content,
    colLabels=headers,
    cellColours=cell_colors,
    loc="center",
    cellLoc="center"
)
tab.auto_set_font_size(False)
tab.set_fontsize(9.5)
tab.scale(1.1, 1.8)

# Style header row (RGB 21, 96, 130)
for (i, j), cell in tab.get_celld().items():
    if i == 0:
        cell.set_text_props(weight="bold", color="white")
        cell.set_facecolor((21/255, 96/255, 130/255))

save_path_table = r"02_Experimente\01_QLearning_Basis\q_table_reward_farming.png"
fig.tight_layout()
fig.savefig(save_path_table, dpi=300, bbox_inches="tight")
plt.close(fig)

# ==============================================================================
# 9. Q-Table: Complete Table for Minimal State Representation (Experiment 05)
# ==============================================================================
path_q_min = r"02_Experimente\05_QLearning_Minimal\q_table.csv"
with open(path_q_min, mode="r", encoding="utf-8") as f:
    q_min_rows = list(csv.DictReader(f))

rel_states = ["Empty", "Match", "Dismatch"]
action_labels = ["Buffer 1 (A0)", "Buffer 2 (A1)", "Return Loop (A2)"]

table_min_content = []
cell_min_colors = []

for idx, r in enumerate(q_min_rows):
    b1_rel = rel_states[idx // 3]
    b2_rel = rel_states[idx % 3]
    
    q0 = float(r["Action_0"])
    q1 = float(r["Action_1"])
    q2 = float(r["Action_2"])
    
    best_idx = max(range(3), key=lambda i: [q0, q1, q2][i])
    chosen_act = action_labels[best_idx]
    
    row_text = [
        b1_rel,
        b2_rel,
        f"{q0:.2f}",
        f"{q1:.2f}",
        f"{q2:.2f}",
        chosen_act
    ]
    table_min_content.append(row_text)
    row_color = "#f2f2f2" if idx % 2 == 1 else "#ffffff"
    cell_min_colors.append([row_color] * len(row_text))

min_headers = [
    "Buffer 1 (Relative)",
    "Buffer 2 (Relative)",
    "Q(Buffer 1)",
    "Q(Buffer 2)",
    "Q(Return)",
    "Optimal Action"
]

fig, ax = plt.subplots(figsize=(11, 4.8))
ax.axis("off")
ax.set_title("Complete Q-Table: Minimal State Representation", fontsize=13, fontweight="bold", pad=15)

col_widths = [0.18, 0.18, 0.14, 0.14, 0.14, 0.22]

tab_min = ax.table(
    cellText=table_min_content,
    colLabels=min_headers,
    colWidths=col_widths,
    cellColours=cell_min_colors,
    loc="center",
    cellLoc="center"
)
tab_min.auto_set_font_size(False)
tab_min.set_fontsize(10)
tab_min.scale(1.0, 1.8)

# Style header row (RGB 21, 96, 130)
for (i, j), cell in tab_min.get_celld().items():
    if i == 0:
        cell.set_text_props(weight="bold", color="white")
        cell.set_facecolor((21/255, 96/255, 130/255))

save_path_table_min = r"02_Experimente\05_QLearning_Minimal\q_table_minimal.png"
fig.tight_layout()
fig.savefig(save_path_table_min, dpi=300, bbox_inches="tight")
plt.close(fig)


# ==============================================================================
# 10. Q-Value Convergence for Exemplary State (over Trials & Global Steps)
# ==============================================================================
import pandas as pd

hist_path = r"02_Experimente\05_QLearning_Minimal\20260909_175953_q_history.csv"
df_hist = pd.read_csv(hist_path)

df_hist["global_step"] = range(1, len(df_hist) + 1)

# Exemplarischer Zustand: z.B. (2, 2) = Beide Puffer Dismatch
b1_target, b2_target = 2, 2
rel_map = {0: "Empty", 1: "Match", 2: "Dismatch"}
state_label = f"({b1_target}, {b2_target}) [B1={rel_map[b1_target]}, B2={rel_map[b2_target]}]"

# Nach Zielzustand filtern und auf die ersten 100 Trials begrenzen
sub_hist = df_hist[(df_hist["b1"] == b1_target) & (df_hist["b2"] == b2_target)].copy()
sub_hist["trial"] = range(1, len(sub_hist) + 1)
sub_hist = sub_hist.head(100)

plt.figure(figsize=(9, 5.5))

# Plot: Q-Werte über die ersten 100 Versuche / Entscheidungen in diesem Zustand (Trials)
plt.plot(sub_hist["trial"], sub_hist["q1"], label="Buffer 1 (Action 1)", color="#1f77b4", linewidth=1.8)
plt.plot(sub_hist["trial"], sub_hist["q2"], label="Buffer 2 (Action 2)", color="#ff7f0e", linewidth=1.8, linestyle="--")
plt.plot(sub_hist["trial"], sub_hist["q3"], label="Return Loop (Action 3)", color="#2ca02c", linewidth=2.0)

plt.title(f"Q-Value Convergence (First 100 Trials)\nState: {state_label}", fontsize=12, fontweight="bold")
plt.xlabel(f"Trials / Visits in State ({b1_target}, {b2_target})", fontsize=11)
plt.ylabel("Q-Value", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="best", framealpha=0.9, fontsize=10)

save_path_q_conv = r"02_Experimente\05_QLearning_Minimal\q_convergence_exemplary_state.png"
plt.tight_layout()
plt.savefig(save_path_q_conv, dpi=300, bbox_inches="tight")
plt.close()
print(f"Plot für Q-Werte-Konvergenz erfolgreich gespeichert unter: {save_path_q_conv}")


# ==============================================================================
# 11. Q-Table Excerpt: Evaluation of LastCollect Memory Vector (Experiment 04)
# ==============================================================================
path_q_lc = r"02_Experimente\04_QLearning_LastCollect\q_table.csv"
with open(path_q_lc, mode="r", encoding="utf-8") as f:
    q_lc_rows = list(csv.DictReader(f))

rel_names = {0: "Empty", 1: "Match", 2: "Dismatch"}
cat_names = {1: "0-2", 2: "3-6", 3: "7-9"}
lc_names = {0: "Init (0)", 1: "Collected (1)", 2: "NotCollected (2)"}

def decode_exp04(idx):
    rem = idx
    lc = rem % 3; rem //= 3
    b2c = (rem % 3) + 1; rem //= 3
    b1c = (rem % 3) + 1; rem //= 3
    b2r = rem % 3; rem //= 3
    b1r = rem % 3
    return b1r, b2r, b1c, b2c, lc

# Selected state indices comparing LC=1 (Collected) vs LC=2 (NotCollected)
# when Buffer 1 is empty and Buffer 2 is occupied (Dismatch)
selected_lc_indices = [55, 56, 58, 59]

table_lc_content = []
cell_lc_colors = []

for row_idx, idx in enumerate(selected_lc_indices):
    r = q_lc_rows[idx]
    b1r, b2r, b1c, b2c, lc = decode_exp04(idx)
    q0 = float(r["Action_0"])
    q1 = float(r["Action_1"])
    q2 = float(r["Action_2"])
    
    best_idx = max(range(3), key=lambda i: [q0, q1, q2][i])
    act_names = ["Buffer 1 (A0)", "Buffer 2 (A1)", "Return Loop (A2)"]
    chosen_act = act_names[best_idx]
    
    b1_str = f"{rel_names[b1r]} ({cat_names[b1c]})" if b1r != 0 else "Empty"
    b2_str = f"{rel_names[b2r]} ({cat_names[b2c]})" if b2r != 0 else "Empty"
    lc_str = lc_names[lc]
    eval_text = "Send to empty buffer"
    
    row_text = [
        str(idx),
        b1_str,
        b2_str,
        lc_str,
        f"{q0:.2f}",
        f"{q1:.2f}",
        f"{q2:.2f}",
        chosen_act,
        eval_text
    ]
    table_lc_content.append(row_text)
    row_color = "#f2f2f2" if row_idx % 2 == 1 else "#ffffff"
    cell_lc_colors.append([row_color] * len(row_text))

headers_lc = [
    "State Index",
    "Buffer 1 (Rel. / Fill)",
    "Buffer 2 (Rel. / Fill)",
    "LastCollect (Memory)",
    "Q(Buffer 1)",
    "Q(Buffer 2)",
    "Q(Return Loop)",
    "Selected Action",
    "Evaluation"
]

fig, ax = plt.subplots(figsize=(14.5, 3.2))
ax.axis("off")
ax.set_title("Q-Table Excerpt: Evaluation of LastCollect Memory Vector (Experiment 04)\n"
             "Agent refuses to reject 'Collected' parts to the loop: Immediate buffering avoids transport latency",
             fontsize=12, fontweight="bold", pad=15)

col_widths = [0.09, 0.15, 0.15, 0.14, 0.10, 0.10, 0.11, 0.14, 0.16]

tab_lc = ax.table(
    cellText=table_lc_content,
    colLabels=headers_lc,
    colWidths=col_widths,
    cellColours=cell_lc_colors,
    loc="center",
    cellLoc="center"
)
tab_lc.auto_set_font_size(False)
tab_lc.set_fontsize(10)
tab_lc.scale(1.0, 1.8)

# Style header row (RGB 21, 96, 130)
for (i, j), cell in tab_lc.get_celld().items():
    if i == 0:
        cell.set_text_props(weight="bold", color="white")
        cell.set_facecolor((21/255, 96/255, 130/255))

save_path_table_lc = r"02_Experimente\04_QLearning_LastCollect\q_table_lastcollect.png"
fig.tight_layout()
fig.savefig(save_path_table_lc, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Plot für LastCollect-Tabelle erfolgreich gespeichert unter: {save_path_table_lc}")


# ==============================================================================
# 12. Decision Tables: Simple Reflex Agent Comparison (With vs. Without LastCollect)
# ==============================================================================

# --- 12.1 Reflex Agent OHNE LastCollect (9 Zustände) ---
rel_states = ["Empty", "Match", "Dismatch"]
rows_without = []

for b1 in [0, 1, 2]:
    for b2 in [0, 1, 2]:
        idx = b1 * 3 + b2
        b1_str = rel_states[b1]
        b2_str = rel_states[b2]
        
        if b1 == 1:
            act = "Buffer 1 (A1)"
            rule = "Match Buffer 1 Priorität"
            color = "#e8f4f8"
        elif b2 == 1:
            act = "Buffer 2 (A2)"
            rule = "Match Buffer 2 Priorität"
            color = "#e8f4f8"
        elif b1 == 0 and b2 == 0:
            act = "Buffer 1 (A1)"
            rule = "Initiale Belegung (Puffer 1)"
            color = "#f4f6f7"
        elif b1 == 0 and b2 == 2:
            act = "Buffer 1 (A1)"
            rule = "Freien Puffer 1 sofort belegen (Parallelität)"
            color = "#e8f8f0"  # Optimal green
        elif b1 == 2 and b2 == 0:
            act = "Buffer 2 (A2)"
            rule = "Freien Puffer 2 sofort belegen (Parallelität)"
            color = "#e8f8f0"  # Optimal green
        else:  # b1 == 2 and b2 == 2
            act = "Return Loop (A3)"
            rule = "Beide Puffer blockiert -> Schleife"
            color = "#fef9e7"
            
        rows_without.append([str(idx), b1_str, b2_str, act, rule, color])

headers_without = [
    "Index",
    "Buffer 1 (Relative)",
    "Buffer 2 (Relative)",
    "Reflex Action",
    "Strategy / Rule"
]

fig1, ax1 = plt.subplots(figsize=(12, 4.8))
ax1.axis("off")
ax1.set_title("Lookup Table: Simple Reflex Agent WITHOUT LastCollect (9 States)\n"
              "Strategy: Immediate Parallel Buffering without Wait Latency (1,956 Steps | 429:02 min)",
              fontsize=12, fontweight="bold", pad=15)

tab_data1 = [r[:5] for r in rows_without]
tab_colors1 = [[r[5]] * 5 for r in rows_without]
col_widths1 = [0.08, 0.20, 0.20, 0.18, 0.34]

tab1 = ax1.table(
    cellText=tab_data1,
    colLabels=headers_without,
    colWidths=col_widths1,
    cellColours=tab_colors1,
    loc="center",
    cellLoc="center"
)
tab1.auto_set_font_size(False)
tab1.set_fontsize(9.5)
tab1.scale(1.0, 1.8)

for (i, j), cell in tab1.get_celld().items():
    if i == 0:
        cell.set_text_props(weight="bold", color="white")
        cell.set_facecolor("#236e4b")  # Green theme for the faster agent

save_path_without = r"02_Experimente\00_Baseline_ReflexAgent\reflex_table_without_lastcollect.png"
fig1.tight_layout()
fig1.savefig(save_path_without, dpi=300, bbox_inches="tight")
plt.close(fig1)
print(f"Plot für Reflex-Tabelle OHNE LastCollect gespeichert unter: {save_path_without}")


# --- 12.2 Reflex Agent MIT LastCollect (27 Zustände) ---
lc_labels = ["Init (0)", "Collected (1)", "NotCollected (2)"]
rows_with = []

idx = 0
for b1 in [0, 1, 2]:
    for b2 in [0, 1, 2]:
        for lc in [0, 1, 2]:
            b1_str = rel_states[b1]
            b2_str = rel_states[b2]
            lc_str = lc_labels[lc]
            
            if b1 == 1:
                act = "Buffer 1 (A1)"
                rule = "Match Buffer 1 Priorität"
                color = "#f4f6f7"
            elif b2 == 1:
                act = "Buffer 2 (A2)"
                rule = "Match Buffer 2 Priorität"
                color = "#f4f6f7"
            elif b1 == 0 and b2 == 0:
                act = "Buffer 1 (A1)"
                rule = "Initiale Belegung (Puffer 1)"
                color = "#f4f6f7"
            elif b1 == 0 and b2 == 2:
                if lc == 1:
                    act = "Return Loop (A3)"
                    rule = "Auf Stautyp warten -> erzeugt +15 Schritte Latenz!"
                    color = "#ffd6d6"  # Red highlight for anomaly rule
                else:
                    act = "Buffer 1 (A1)"
                    rule = "Stautyp / Init -> Puffer 1 belegen"
                    color = "#eaf2f8"
            elif b1 == 2 and b2 == 0:
                if lc == 1:
                    act = "Return Loop (A3)"
                    rule = "Auf Stautyp warten -> erzeugt +15 Schritte Latenz!"
                    color = "#ffd6d6"  # Red highlight for anomaly rule
                else:
                    act = "Buffer 2 (A2)"
                    rule = "Stautyp / Init -> Puffer 2 belegen"
                    color = "#eaf2f8"
            else:  # b1 == 2 and b2 == 2
                act = "Return Loop (A3)"
                rule = "Beide Puffer blockiert -> Schleife"
                color = "#fef9e7"
                
            rows_with.append([str(idx), b1_str, b2_str, lc_str, act, rule, color])
            idx += 1

headers_with = [
    "Index",
    "Buffer 1 (Rel.)",
    "Buffer 2 (Rel.)",
    "LastCollect",
    "Reflex Action",
    "Strategy / Rule (Hypothesis)"
]

fig2, ax2 = plt.subplots(figsize=(14.5, 11.5))
ax2.axis("off")
ax2.set_title("Lookup Table: Simple Reflex Agent WITH LastCollect Memory (27 States)\n"
              "Red Highlight: The 2 Anomaly Rules where LC=1 sends parts to Return (1,971 Steps | 432:12 min)",
              fontsize=12, fontweight="bold", pad=15)

tab_data2 = [r[:6] for r in rows_with]
tab_colors2 = [[r[6]] * 6 for r in rows_with]
col_widths2 = [0.07, 0.16, 0.16, 0.14, 0.16, 0.31]

tab2 = ax2.table(
    cellText=tab_data2,
    colLabels=headers_with,
    colWidths=col_widths2,
    cellColours=tab_colors2,
    loc="center",
    cellLoc="center"
)
tab2.auto_set_font_size(False)
tab2.set_fontsize(9)
tab2.scale(1.0, 1.6)

for (i, j), cell in tab2.get_celld().items():
    if i == 0:
        cell.set_text_props(weight="bold", color="white")
        cell.set_facecolor("#8b2500")  # Warm crimson theme

save_path_with = r"02_Experimente\00_Baseline_ReflexAgent\reflex_table_with_lastcollect.png"
fig2.tight_layout()
fig2.savefig(save_path_with, dpi=300, bbox_inches="tight")
plt.close(fig2)
print(f"Plot für Reflex-Tabelle MIT LastCollect gespeichert unter: {save_path_with}")



