"""
Bulanik ortam modeli (referans makalenin 2. bolumu).

Ortam tek bir bulanik kume E ile temsil edilir:
    E = O U P
    O : engellerin bulanik kumesi   -> mu_O(x)  "burasi ne kadar tehlikeli?"
    P : rotanin bulanik kumesi      -> mu_P(x)  "burasi rotaya ne kadar yakin?"
    U : Yager t-conorm (bulanik VEYA)

Denklem numaralari makaledeki numaralardir.
"""
import numpy as np


def rot90(v):
    """R(pi/2) v : 2B vektoru 90 derece saat yonunun tersine dondurur."""
    return np.array([-v[1], v[0]])


def yager_union(memberships, p=1.0, smooth=True):
    """
    Yager t-conorm ile K adet bulanik kumenin birlesimi (Denklem 1 ve 7).

        a1 + ... + aK = sat( (a1^p + ... + aK^p)^(1/p) )

    memberships : (K, ...) dizisi, her satir bir kumenin uyelik degerleri
    p           : p = 1 en temkinli birlesim, p -> sonsuz klasik max
    smooth      : True ise sat(.) yerine turevlenebilir tanh(.) kullanilir
                  (makalede de boyle yapilmistir)
    """
    s = np.sum(memberships ** p, axis=0) ** (1.0 / p)
    return np.tanh(s) if smooth else np.minimum(1.0, s)


class FuzzyEnvironment:
    """Engeller + rota -> tek bir bulanik ortam haritasi mu_E(x)."""

    def __init__(self, waypoints, sigma=1.0, gamma=0.75, b=1.25, delta=0.1, p=1.0):
        """
        waypoints : (M+1, 2) global rotanin ara noktalari
        sigma     : rotadan izin verilen yanal sapma (m)
        gamma     : komsu Gauss waypoint'lerinin ortusmesi
        b         : engel uyeliginin dusus keskinligi
        delta     : hareketli engeli hiz yonunde uzatma katsayisi
        p         : Yager t-conorm parametresi
        """
        self.b, self.delta, self.p = b, delta, p
        self.obs_c = np.zeros((0, 2))
        self.obs_invL = np.zeros((0, 2, 2))
        self._build_path(np.asarray(waypoints, float), sigma, gamma)

    # ------------------------------------------------------------------
    # Rota: bulanik waypoint'ler (Denklem 8, 9, 10)
    # ------------------------------------------------------------------
    def _build_path(self, w, sigma, gamma):
        centers, inv_spreads = [], []
        for j in range(len(w) - 1):
            seg = w[j + 1] - w[j]
            L = np.linalg.norm(seg)
            h = seg / L                                   # rota yonu
            H = np.column_stack([h, rot90(h)])            # [h, h_dik]
            # Gamma_j: rota boyunca gamma*L, yanlara sigma kadar yayilim (Denklem 10)
            G = H @ np.diag([gamma * L, sigma]) @ H.T
            centers.append(0.5 * (w[j] + w[j + 1]))      # p_j: iki waypoint'in orta noktasi
            inv_spreads.append(np.linalg.inv(G))
        self.wp_c = np.array(centers)
        self.wp_invG = np.array(inv_spreads)
        self.goal = w[-1]

    # ------------------------------------------------------------------
    # Engeller: atomik bulanik kumeler (Denklem 2, 3)
    # ------------------------------------------------------------------
    def set_obstacles(self, centers, lam, r, velocities=None):
        """
        centers    : (N, 2) atomik engellerin merkezleri c_i
        lam        : nesne boyutu lambda_i (skaler veya N'lik dizi)
        r          : guvenlik mesafesi r_i (skaler veya N'lik dizi)
        velocities : (N, 2) engel hizlari v_i (None -> hepsi sabit)
        """
        centers = np.atleast_2d(centers).astype(float)
        N = len(centers)
        lam, r = np.broadcast_to(lam, N), np.broadcast_to(r, N)
        if velocities is None:
            velocities = np.zeros((N, 2))
        inv = []
        for i in range(N):
            Lam = (lam[i] + r[i]) * np.eye(2)            # daire: boyut + guvenlik payi
            speed = np.linalg.norm(velocities[i])
            if speed > 1e-9:                              # hareketli engel: hiz yonunde uzat
                q = velocities[i] / speed
                Q = np.column_stack([q, rot90(q)])
                Lam = Lam + Q @ np.diag([self.delta * speed ** 2, 0.0]) @ Q.T
            inv.append(np.linalg.inv(Lam))
        self.obs_c = centers
        self.obs_invL = np.array(inv)

    # ------------------------------------------------------------------
    # Uyelik fonksiyonlari.  x: (..., 2) tek nokta ya da nokta izgarasi
    # ------------------------------------------------------------------
    @staticmethod
    def _local_coords(x, centers, inv_mats):
        """chi = A^-1 (x - c) donusumunu tum kumeler icin bir kerede hesaplar."""
        shape = (-1,) + (1,) * (x.ndim - 1) + (2,)
        d = x[None, ...] - centers.reshape(shape)
        return np.einsum("nij,n...j->n...i", inv_mats, d)

    def _atomic_obstacles(self, x):
        """Her atomik engelin uyeligi mu_Oi(x), sekil (N, ...)  (Denklem 2, 6)."""
        x = np.asarray(x, float)
        if len(self.obs_c) == 0:
            return np.zeros((1,) + x.shape[:-1])
        chi = self._local_coords(x, self.obs_c, self.obs_invL)
        r2 = np.maximum(np.sum(chi ** 2, axis=-1), 1e-12)
        return np.tanh(r2 ** (-self.b))                   # BMF: tanh((chi^T chi)^-b)  (Denklem 6)

    def _atomic_waypoints(self, x):
        """Her bulanik waypoint'in uyeligi mu_Wj(x), sekil (M, ...)  (Denklem 8, 9)."""
        x = np.asarray(x, float)
        chi = self._local_coords(x, self.wp_c, self.wp_invG)
        return np.exp(-0.5 * np.sum(chi ** 2, axis=-1))   # Gauss BMF (Denklem 8)

    def mu_obstacle(self, x):
        """Toplam engel uyeligi mu_O(x) (Denklem 5)."""
        return yager_union(self._atomic_obstacles(x), self.p)

    def mu_path(self, x):
        """Bulanik rota uyeligi mu_P(x) (Denklem 12)."""
        return yager_union(self._atomic_waypoints(x), self.p)

    def mu_env(self, x):
        """
        Ortam uyeligi mu_E(x): butun atomik engeller ve waypoint'ler TEK bir
        Yager birlesiminde (Remark 1, Denklem 7 ve 14).

        Not: tanh yaklasiminda birlesim birlesmeli (associative) degildir.
        Once mu_O ve mu_P'yi ayri ayri hesaplayip sonra tekrar birlestirmek
        tanh(tanh(.)) <= 0.76 sinirina yol acar; bu yuzden tek seferde birlestirilir.
        """
        allsets = np.concatenate([self._atomic_obstacles(x), self._atomic_waypoints(x)])
        return yager_union(allsets, self.p)

    def grad_env(self, x, h=1e-3):
        """mu_E'nin gradyani, merkezi farklar ile sayisal olarak."""
        x = np.asarray(x, float)
        e = np.eye(2) * h
        m = self.mu_env(np.stack([x + e[0], x - e[0], x + e[1], x - e[1]]))
        return np.array([m[0] - m[1], m[2] - m[3]]) / (2 * h)
