"""
Deney senaryolari. Parametreler referans makalenin 4. bolumunden alinmistir.
"""
import numpy as np
from fuzzy_env import FuzzyEnvironment


def case_study_1(p=1.0, b=1.25):
    """
    Case Study 1 (makale, Bolum 4.1):
      - Duz rota [0, 20] -> [200, 20], 2 m aralikli 100 waypoint
      - Merkezi [70, 0], yaricapi 30 m, acisi -3pi/4 .. 3pi/4 olan yay seklinde
        (icbukey) sabit engel, 97 atomik kume, lambda = 0.25 m, r = 1.25 m
    Donus: (env, engel_merkezleri)
    """
    xs = np.linspace(0, 200, 101)
    waypoints = np.column_stack([xs, np.full_like(xs, 20.0)])
    env = FuzzyEnvironment(waypoints, sigma=1.0, gamma=0.75, b=b, delta=0.1, p=p)

    th = np.linspace(-3 * np.pi / 4, 3 * np.pi / 4, 97)
    arc = np.column_stack([70 + 30 * np.cos(th), 30 * np.sin(th)])
    env.set_obstacles(arc, lam=0.25, r=1.25)
    return env, arc
