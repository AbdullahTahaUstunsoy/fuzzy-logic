"""
FPPA'yi Case Study 1 uzerinde calistirir ve robotun yolunu cizer.

Calistirma:  python experiments/baseline/run_fppa.py
             python experiments/baseline/run_fppa.py --alpha 0.99   (makaledeki deger)
Cikti:       results/figures/case1_fppa_path.png
"""
import os, sys, time, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from scenarios import case_study_1
from fppa import FPPA

ap = argparse.ArgumentParser()
ap.add_argument("--alpha", type=float, default=0.90)
ap.add_argument("--eta_n", type=float, default=0.75)
ap.add_argument("--start_y", type=float, default=19.0)
args = ap.parse_args()

env, arc = case_study_1()
planner = FPPA(env, alpha=args.alpha, eta_n=args.eta_n)

t0 = time.perf_counter()
traj, logs = planner.run([0.0, args.start_y])
elapsed = time.perf_counter() - t0

reached = np.linalg.norm(traj[-1] - env.goal) < 1.01
clearance = np.min(np.linalg.norm(traj[:, None] - arc[None], axis=-1))
print(f"alpha={args.alpha}  eta_n={args.eta_n}")
print(f"hedefe ulasti mi : {reached}")
print(f"adim sayisi      : {len(traj) - 1}  ({(len(traj) - 1) * planner.ts:.1f} s simulasyon)")
print(f"adim basi sure   : {1000 * elapsed / len(traj):.3f} ms")
print(f"engele min mesafe: {clearance:.2f} m (atomik merkezlere)")

# --- cizim: ortam haritasi + yol, altta hiz ve hedef kontur ---
gx, gy = np.meshgrid(np.linspace(-5, 210, 300), np.linspace(-45, 50, 160))
mu = env.mu_env(np.stack([gx, gy], axis=-1))
v = np.linalg.norm(np.diff(traj, axis=0), axis=1) / planner.ts
t = np.arange(len(v)) * planner.ts

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1]})
cs = ax1.contourf(gx, gy, mu, levels=20, cmap="viridis")
fig.colorbar(cs, ax=ax1, label=r"$\mu_E$")
ax1.plot(arc[:, 0], arc[:, 1], "w.", ms=3, label="engel")
ax1.plot([0, 200], [20, 20], "w--", lw=1, label="global rota")
ax1.plot(traj[:, 0], traj[:, 1], "r-", lw=2, label="FPPA")
ax1.set_aspect("equal"); ax1.legend(loc="lower right")
ax1.set_title(rf"FPPA, Case Study 1  ($\alpha$={args.alpha}, $\eta_n$={args.eta_n})")

ax2.plot(t, v, label="hiz [m/s]")
ax2.plot(t, [l["mu_th"] * 5 for l in logs], "--", label=r"hedef kontur $\mu_{th}$ (x5)")
ax2.set_xlabel("zaman [s]"); ax2.grid(alpha=0.3); ax2.legend()

os.makedirs(os.path.join(ROOT, "results", "figures"), exist_ok=True)
out = os.path.join(ROOT, "results", "figures", "case1_fppa_path.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("kaydedildi:", out)
