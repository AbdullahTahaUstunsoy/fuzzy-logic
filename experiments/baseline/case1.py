"""
Case Study 1'in yeniden uretimi ve klasik APF ile karsilastirma.

Calistirma:  python experiments/baseline/case1.py
Ciktilar:
  results/figures/case1_paths.png        FPPA vs klasik APF yollari (makale Fig. 6 karsiligi)
  results/figures/case1_kinematics.png   hiz, yon, ivme, yaw rate (makale Fig. 7-10 karsiligi)
  results/figures/case1_alpha.png        makaledeki alpha=0.99 ile bizim alpha=0.90
  results/tables/case1_metrics.csv       sayisal karsilastirma tablosu
"""
import os, sys, csv, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from scenarios import case_study_1
from fppa import FPPA
from apf import APF
from metrics import evaluate, kinematics

START = [0.0, 19.0]
FIG = os.path.join(ROOT, "results", "figures")
TAB = os.path.join(ROOT, "results", "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)


def timed_run(planner):
    t0 = time.perf_counter()
    traj, logs = planner.run(START)
    return traj, logs, 1000 * (time.perf_counter() - t0) / len(traj)


def draw_env(ax, env, arc):
    gx, gy = np.meshgrid(np.linspace(-5, 210, 300), np.linspace(-45, 50, 160))
    cs = ax.contourf(gx, gy, env.mu_env(np.stack([gx, gy], -1)), levels=20, cmap="viridis")
    ax.plot(arc[:, 0], arc[:, 1], "w.", ms=3, label="engel")
    ax.plot([0, 200], [20, 20], "w--", lw=1, label="global rota")
    ax.set_aspect("equal")
    return cs


env, arc = case_study_1()

# Ayni senaryoda iki planlayici (ayni arayuz: run -> traj, logs)
planners = {
    "FPPA (alpha=0.90)": FPPA(env, alpha=0.90),
    "FPPA (alpha=0.99, makale)": FPPA(env, alpha=0.99),
    "Klasik APF": APF(env.goal, arc),
}
results = {name: timed_run(p) for name, p in planners.items()}

# ---------------- tablo ----------------
rows = [(name, evaluate(tr, env.goal, arc, 0.05, ms)) for name, (tr, _, ms) in results.items()]
with open(os.path.join(TAB, "case1_metrics.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method"] + list(rows[0][1].keys()))
    for name, m in rows:
        w.writerow([name] + [round(v, 3) if isinstance(v, float) else v for v in m.values()])
for name, m in rows:
    print(f"{name:28s}", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})

# ---------------- yollar: FPPA vs APF ----------------
fppa_tr = results["FPPA (alpha=0.90)"][0]
apf_tr = results["Klasik APF"][0]
fig, ax = plt.subplots(figsize=(11, 5))
fig.colorbar(draw_env(ax, env, arc), ax=ax, label=r"$\mu_E$")
ax.plot(fppa_tr[:, 0], fppa_tr[:, 1], "r-", lw=2, label="FPPA")
ax.plot(apf_tr[:, 0], apf_tr[:, 1], "c-", lw=2, label="Klasik APF")
ax.plot(*apf_tr[-1], "cx", ms=12, mew=3)
ax.annotate("yerel minimum", apf_tr[-1], xytext=(apf_tr[-1][0] - 35, apf_tr[-1][1] - 12),
            color="c", arrowprops=dict(color="c", arrowstyle="->"))
ax.legend(loc="lower right"); ax.set_title("Case Study 1: FPPA vs klasik APF")
fig.savefig(os.path.join(FIG, "case1_paths.png"), dpi=130, bbox_inches="tight")

# ---------------- kinematik (FPPA) ----------------
v, psi, acc, yaw = kinematics(fppa_tr, 0.05)
t = np.arange(len(v)) * 0.05
fig, axs = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
for a, y, lab in zip(axs, [v, np.degrees(psi), acc, yaw],
                     ["hiz [m/s]", "yon [deg]", "ivme [m/s²]", "yaw rate [rad/s]"]):
    a.plot(t[:len(y)], y); a.set_ylabel(lab); a.grid(alpha=0.3)
axs[-1].set_xlabel("zaman [s]")
axs[0].set_title("Case Study 1: FPPA kinematikleri (makale Fig. 7-10 karsiligi)")
fig.savefig(os.path.join(FIG, "case1_kinematics.png"), dpi=130, bbox_inches="tight")

# ---------------- alpha karsilastirmasi ----------------
fig, axs = plt.subplots(1, 2, figsize=(15, 4.5))
for a, name in zip(axs, ["FPPA (alpha=0.99, makale)", "FPPA (alpha=0.90)"]):
    tr = results[name][0]
    draw_env(a, env, arc)
    a.plot(tr[:, 0], tr[:, 1], "r-", lw=1.2)
    a.set_title(name)
fig.savefig(os.path.join(FIG, "case1_alpha.png"), dpi=130, bbox_inches="tight")
print("grafikler ve tablo kaydedildi.")
