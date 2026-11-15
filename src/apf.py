"""
Klasik Yapay Potansiyel Alan (APF) planlayicisi - Khatib (1985) tarzi.

Karsilastirma yontemi: makale (Bolum 4.1) klasik APF'nin icbukey engelde
yerel minimuma takildigini belirtir; bunu gostermek icin kullanilir.

FPPA ile ayni arayuze sahiptir (run(x0) -> (traj, logs)), boylece deney
betikleri iki planlayiciyi ayni sekilde calistirabilir.
"""
import numpy as np


class APF:
    def __init__(self, goal, obstacles, k_att=1.0, k_rep=50.0, d0=6.0,
                 v_m=5.0, ts=0.05):
        """
        goal      : hedef nokta
        obstacles : (N, 2) engel noktalari (FPPA'daki atomik merkezlerle ayni)
        k_att     : cekici kuvvet katsayisi
        k_rep     : itici kuvvet katsayisi
        d0        : itici kuvvetin etki mesafesi (m); daha uzaktaki engeller itmez
        v_m       : azami hiz (m/s), ts: ornekleme suresi (s)
        """
        self.goal = np.asarray(goal, float)
        self.obs = np.asarray(obstacles, float)
        self.k_att, self.k_rep, self.d0 = k_att, k_rep, d0
        self.v_m, self.ts = v_m, ts

    def force(self, x):
        # cekici kuvvet: hedefe dogru, sabit buyuklukte
        to_goal = self.goal - x
        f = self.k_att * to_goal / (np.linalg.norm(to_goal) + 1e-9)
        # itici kuvvet: d0'dan yakin her engelden uzaga dogru
        diff = x - self.obs
        d = np.linalg.norm(diff, axis=1)
        near = d < self.d0
        if np.any(near):
            dn = d[near][:, None]
            f += np.sum(self.k_rep * (1 / dn - 1 / self.d0) / dn ** 2 * diff[near] / dn, axis=0)
        return f

    def step(self, x):
        f = self.force(x)
        n = np.linalg.norm(f)
        # kuvvet yonunde ilerle; adim boyu en fazla v_m * ts
        dx = f / (n + 1e-9) * min(self.v_m * self.ts, n * self.ts)
        return dx, {"force": n}

    def run(self, x0, max_steps=6000, goal_tol=1.0):
        x = np.array(x0, float)
        traj, logs = [x.copy()], []
        for _ in range(max_steps):
            dx, info = self.step(x)
            x = x + dx
            traj.append(x.copy())
            logs.append(info)
            if np.linalg.norm(x - self.goal) < goal_tol:
                break
        return np.array(traj), logs
