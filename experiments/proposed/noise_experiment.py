"""
Ozgun katki deneyi: gurultulu engel algilama altinda baseline FPPA (k=0) ile
belirsizlige duyarli uyelik fonksiyonu (k=1, k=2) karsilastirmasi.

Calistirma:  python experiments/proposed/noise_experiment.py           (tam deney, birkac dakika)
             python experiments/proposed/noise_experiment.py --quick   (hizli deneme)
Ciktilar:
  results/tables/noise_raw.csv          her kosunun olcutleri
  results/tables/noise_summary.csv      sigma x k icin ortalama ve standart sapma
  results/figures/noise_metrics.png     yaw rate, engel mesafesi, yol uzunlugu - sigma
  results/figures/noise_paths.png       ornek yorungeler (sigma = 1 m, k = 0 ve k = 2)
"""
import os, sys, csv, time, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from scenarios import case_study_1
from noise import simulate_noisy
from metrics import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true", help="az tekrar ile hizli deneme")
args = ap.parse_args()

SIGMAS = [0.0, 0.5, 1.0, 2.0] if args.quick else [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
KS = [0, 1, 2]
SEEDS = range(2 if args.quick else 10)
SAFE_DIST = 1.25                                  # makaledeki guvenlik mesafesi r
TS = 0.05
FIG = os.path.join(ROOT, "results", "figures")
TAB = os.path.join(ROOT, "results", "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)

env, arc = case_study_1()

# ---------------- deney dongusu ----------------
raw, example = [], {}
t_start = time.perf_counter()
for sigma in SIGMAS:
    for k in KS:
        for seed in SEEDS:
            t0 = time.perf_counter()
            traj = simulate_noisy(env, arc, sigma, k, seed)
            ms = 1000 * (time.perf_counter() - t0) / len(traj)
            m = evaluate(traj, env.goal, arc, TS, ms)    # olcutler GERCEK engele gore
            m["safe"] = m["min_clearance_m"] >= SAFE_DIST
            m["success"] = m["reached_goal"] and m["min_clearance_m"] > 0
            raw.append({"sigma": sigma, "k": k, "seed": seed, **m})
            if sigma == 1.0 and seed == 0:
                example[k] = traj
    print(f"sigma={sigma:<4} tamam  ({time.perf_counter() - t_start:.0f} s)")

with open(os.path.join(TAB, "noise_raw.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(raw[0].keys()))
    w.writeheader()
    for r in raw:
        w.writerow({c: (round(v, 4) if isinstance(v, float) else v) for c, v in r.items()})

# ---------------- ozet tablo ----------------
COLS = ["rms_yaw_rate_rad_s", "min_clearance_m", "path_length_m", "travel_time_s", "step_time_ms"]
summary = []
for sigma in SIGMAS:
    for k in KS:
        rs = [r for r in raw if r["sigma"] == sigma and r["k"] == k]
        row = {"sigma": sigma, "k": k,
               "success_rate": np.mean([r["success"] for r in rs]),
               "safe_rate": np.mean([r["safe"] for r in rs])}
        for c in COLS:
            vals = np.array([r[c] for r in rs])
            row[c + "_mean"], row[c + "_std"] = vals.mean(), vals.std()
        summary.append(row)

with open(os.path.join(TAB, "noise_summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    w.writeheader()
    for r in summary:
        w.writerow({c: (round(float(v), 3) if not isinstance(v, int) else v) for c, v in r.items()})

print(f"\n{'sigma':>5} {'k':>2} {'basari':>7} {'yaw rms':>8} {'min mesafe':>11} {'yol':>7}")
for r in summary:
    print(f"{r['sigma']:>5} {r['k']:>2} {r['success_rate']:>7.2f} "
          f"{r['rms_yaw_rate_rad_s_mean']:>8.2f} {r['min_clearance_m_mean']:>11.2f} "
          f"{r['path_length_m_mean']:>7.0f}")

# ---------------- grafik 1: olcutler - sigma ----------------
COLORS = {0: "#2a78d6", 1: "#eb6834", 2: "#1baf7a"}
MARKERS = {0: "o", 1: "s", 2: "^"}
LABELS = {0: "k=0 (baseline)", 1: "k=1", 2: "k=2"}
panels = [("rms_yaw_rate_rad_s", "RMS yaw rate [rad/s]\n(dusuk = puruzsuz)"),
          ("min_clearance_m", "engel yuzeyine min. mesafe [m]"),
          ("path_length_m", "yol uzunlugu [m]")]

fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, (col, ylab) in zip(axs, panels):
    for k in KS:
        rs = [r for r in summary if r["k"] == k]
        ys = [r[col + "_mean"] for r in rs]
        ax.errorbar(SIGMAS, ys, yerr=[r[col + "_std"] for r in rs], color=COLORS[k],
                    marker=MARKERS[k], ms=7, lw=2, capsize=3, label=LABELS[k])
        ax.annotate(LABELS[k].split(" ")[0], (SIGMAS[-1], ys[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color="0.25")
    if col == "min_clearance_m":
        ax.axhline(SAFE_DIST, color="0.5", ls="--", lw=1)
        ax.annotate("guvenlik mesafesi r", (SIGMAS[0], SAFE_DIST), xytext=(0, 4),
                    textcoords="offset points", fontsize=9, color="0.35")
    ax.set_xlabel(r"olcum gurultusu $\sigma$ [m]"); ax.set_ylabel(ylab)
    ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
axs[0].legend(frameon=False)
fig.suptitle(f"Gurultulu engel algilama: baseline vs belirsizlige duyarli uyelik "
             f"({len(SEEDS)} tekrar, ortalama ± std)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "noise_metrics.png"), dpi=130, bbox_inches="tight")

# ---------------- grafik 2: ornek yorungeler ----------------
if 0 in example and 2 in example:
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.5), sharex=True, sharey=True)
    for ax, k in zip(axs, [0, 2]):
        ax.plot(arc[:, 0], arc[:, 1], ".", color="0.3", ms=3, label="engel (gercek)")
        ax.plot([0, 200], [20, 20], "--", color="0.6", lw=1, label="global rota")
        ax.plot(example[k][:, 0], example[k][:, 1], color=COLORS[k], lw=1.2, label="robot")
        ax.set_title(rf"$\sigma$ = 1 m, {LABELS[k]}"); ax.set_aspect("equal")
        ax.legend(loc="lower right", frameon=False)
    fig.savefig(os.path.join(FIG, "noise_paths.png"), dpi=130, bbox_inches="tight")
print("tablolar ve grafikler kaydedildi.")
