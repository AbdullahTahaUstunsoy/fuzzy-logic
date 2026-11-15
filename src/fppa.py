"""
FPPA planlayicisi (referans makalenin 3. bolumu, Algorithm 1).

Her adimda:
  1) mu_th = alpha * (1 - mu_O(x))^d_o      hangi kontur izlenecek   (Denklem 17)
  2) v_max = v_m   * (1 - mu_O(x))^d_v      engel yakininda hiz siniri (Denklem 18)
  3) dx = normal adim + teget adim                                    (Denklem 22)
       normal : -eta_n (mu_E - mu_th) grad(mu_E)   -> hedef kontura it/cek
       teget  : +- v_max t_s R(pi/2) grad/|grad|   -> kontur boyunca ilerle
"""
import numpy as np
from fuzzy_env import rot90


class FPPA:
    def __init__(self, env, alpha=0.90, d_o=1.25, v_m=5.0, d_v=0.5,
                 eta_n=0.75, ts=0.05, direction=-1):
        """
        env       : FuzzyEnvironment
        alpha     : engel yokken izlenecek kontur seviyesi
                    (makale: 0.99; rota sirtinin en yuksek uyeligi ~0.955 oldugu icin
                    0.99 konturu olusmuyor -> bizim uygulamada 0.90, bkz. README/Limitations)
        d_o, d_v  : engel yakininda kontur ve hizin dusus hizlari
        v_m       : rotadaki azami hiz (m/s)
        eta_n     : normal adim katsayisi (makaledeki gibi 0.75)
        ts        : ornekleme suresi (s)
        direction : teget adimin isareti; engelin hangi taraftan dolanilacagi
        """
        self.env = env
        self.alpha, self.d_o, self.v_m, self.d_v = alpha, d_o, v_m, d_v
        self.eta_n, self.ts, self.direction = eta_n, ts, direction

    def step(self, x):
        """Tek planlama adimi. Donus: (dx, bilgi sozlugu)."""
        mu_o = float(self.env.mu_obstacle(x))
        mu_th = self.alpha * (1 - mu_o) ** self.d_o            # Denklem 17
        v_max = self.v_m * (1 - mu_o) ** self.d_v              # Denklem 18

        mu_e = float(self.env.mu_env(x))
        g = self.env.grad_env(x)
        g_unit = g / (np.linalg.norm(g) + 1e-12)

        normal = -self.eta_n * (mu_e - mu_th) * g              # Denklem 22, 1. terim
        tangent = self.direction * v_max * self.ts * rot90(g_unit)  # Denklem 22, 2. terim
        dx = normal + tangent

        max_step = v_max * self.ts                             # hiz sinirini uygula
        n = np.linalg.norm(dx)
        if n > max_step:
            dx *= max_step / n
        return dx, {"mu_o": mu_o, "mu_e": mu_e, "mu_th": mu_th, "v_max": v_max}

    def run(self, x0, max_steps=20000, goal_tol=1.0):
        """Hedefe ulasana (ya da adim sinirina) kadar calistir."""
        x = np.array(x0, float)
        traj, logs = [x.copy()], []
        for _ in range(max_steps):
            dx, info = self.step(x)
            x = x + dx
            traj.append(x.copy())
            logs.append(info)
            if np.linalg.norm(x - self.env.goal) < goal_tol:
                break
        return np.array(traj), logs
