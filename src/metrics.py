"""
Yorunge performans olcutleri.

Makale sonuclari cogunlukla grafikle veriyor; karsilastirmayi sayisal yapmak
icin asagidaki olcutleri kullaniyoruz.
"""
import numpy as np


def kinematics(traj, ts):
    """Konum dizisinden hiz, yon, ivme ve yaw rate (makaledeki gibi sayisal turev)."""
    d = np.diff(traj, axis=0)
    v = np.linalg.norm(d, axis=1) / ts
    psi = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    acc = np.diff(v) / ts
    yaw_rate = np.diff(psi) / ts
    return v, psi, acc, yaw_rate


def evaluate(traj, goal, obstacle_pts, ts, step_time_ms=None, object_radius=0.25):
    """
    reached_goal     : hedefe 1 m'den fazla yaklasti mi
    path_length_m    : katedilen toplam yol
    travel_time_s    : simulasyon suresi
    min_clearance_m  : nesne yuzeyine en kucuk mesafe (atomik merkez - nesne yaricapi)
    rms_yaw_rate     : donus hizinin RMS'i (kucuk = puruzsuz surus)
    max_abs_acc      : en buyuk ivme (kucuk = konforlu surus)
    step_time_ms     : bir planlama adiminin ortalama hesaplama suresi
    """
    v, psi, acc, yaw = kinematics(traj, ts)
    dists = np.linalg.norm(traj[:, None] - obstacle_pts[None], axis=-1)
    out = {
        "reached_goal": bool(np.linalg.norm(traj[-1] - goal) < 1.01),
        "path_length_m": float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))),
        "travel_time_s": float((len(traj) - 1) * ts),
        "min_clearance_m": float(dists.min() - object_radius),
        "rms_yaw_rate_rad_s": float(np.sqrt(np.mean(yaw ** 2))),
        "max_abs_acc_m_s2": float(np.max(np.abs(acc))),
    }
    if step_time_ms is not None:
        out["step_time_ms"] = float(step_time_ms)
    return out
