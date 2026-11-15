"""
Gurultulu engel algilama altinda simulasyon (ozgun katki).

Gercek bir robot engellerin yerini sensorle (LiDAR, kamera) olcer ve olcum
hatalidir. Bunu modellemek icin her ornekleme aninda (50 ms) her atomik engelin
merkezine bagimsiz Gauss gurultusu eklenir:

    c_i(olculen) = c_i(gercek) + N(0, sigma^2 I)

Onerilen yontem - belirsizlige duyarli uyelik fonksiyonu:
    makale : bulanik dairenin yaricapi = lambda + r
    bizim  : bulanik dairenin yaricapi = lambda + r + k * sigma
k = 0 makaledeki yontemle (baseline) aynidir.
"""
import numpy as np
from fppa import FPPA


def simulate_noisy(env, true_obstacles, sigma, k, seed, lam=0.25, r=1.25,
                   start=(0.0, 19.0), max_steps=6000, goal_tol=1.0, **fppa_kwargs):
    """
    env            : FuzzyEnvironment (rota tanimli)
    true_obstacles : (N, 2) engellerin GERCEK atomik merkezleri
    sigma          : olcum gurultusunun standart sapmasi (m)
    k              : belirsizlik payi katsayisi (0 = baseline)
    seed           : rastgele tohum; ayni tohum = birebir ayni gurultu dizisi
                     (baseline ve onerilen yontem adil karsilastirilir)
    Donus: robotun yorungesi (T, 2)
    """
    rng = np.random.default_rng(seed)
    env.set_obstacles(true_obstacles, lam=lam, r=r + k * sigma)   # genisletilmis uyelik
    planner = FPPA(env, **fppa_kwargs)

    x = np.array(start, float)
    traj = [x.copy()]
    for _ in range(max_steps):
        measured = true_obstacles + rng.normal(0.0, sigma, true_obstacles.shape)
        env.move_obstacles(measured)                 # robot sadece olculeni gorur
        dx, _ = planner.step(x)
        x = x + dx
        traj.append(x.copy())
        if np.linalg.norm(x - env.goal) < goal_tol:
            break

    env.set_obstacles(true_obstacles, lam=lam, r=r)   # ortami eski haline getir
    return np.array(traj)
