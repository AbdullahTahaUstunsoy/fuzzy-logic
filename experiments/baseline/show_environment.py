"""
Case Study 1 icin bulanik ortam haritalarini cizer:
mu_O (engel), mu_P (rota) ve birlesik mu_E.

Calistirma:  python experiments/baseline/show_environment.py
Cikti:       results/figures/case1_environment.png
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from scenarios import case_study_1

env, arc = case_study_1()
gx, gy = np.meshgrid(np.linspace(-5, 210, 300), np.linspace(-45, 50, 160))
grid = np.stack([gx, gy], axis=-1)

maps = [(env.mu_obstacle(grid), r"$\mu_O$ : engel"),
        (env.mu_path(grid), r"$\mu_P$ : rota"),
        (env.mu_env(grid), r"$\mu_E = \mu_O \dot{+} \mu_P$ : ortam")]

fig, axs = plt.subplots(3, 1, figsize=(9, 11))
for ax, (mu, title) in zip(axs, maps):
    cs = ax.contourf(gx, gy, mu, levels=20, vmin=0, vmax=1, cmap="viridis")
    ax.set_title(title); ax.set_aspect("equal")
fig.colorbar(cs, ax=axs, label="uyelik derecesi", shrink=0.6)

os.makedirs(os.path.join(ROOT, "results", "figures"), exist_ok=True)
out = os.path.join(ROOT, "results", "figures", "case1_environment.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("kaydedildi:", out)
print("rota sirtinin en yuksek uyeligi: %.3f" % env.mu_path(np.array([[150.0, 20.0]]))[0])
