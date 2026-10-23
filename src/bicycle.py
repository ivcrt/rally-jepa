# src/bicycle.py
# Baseline physique : modele velo lineaire, coefficients identifies sur les donnees.
#
# On ne connait ni la masse, ni l'inertie, ni la rigidite des pneus de la voiture
# BeamNG, ni le rapport de direction (le volant est en degres, pas l'angle des roues).
# Mais les equations du velo lineaire sont LINEAIRES EN LEURS COEFFICIENTS :
#
#   lateral : dvy/dt + vx*r = a1*vy/vx + a2*r/vx + a3*delta
#   lacet   : dr/dt         = b1*vy/vx + b2*r/vx + b3*delta
#
# avec, physiquement, a1 = -(Cf+Cr)/m < 0, b2 = -(Cf*lf^2 + Cr*lr^2)/Iz < 0,
# a3 > 0 et b3 > 0 (braquer a gauche pousse vers la gauche). On les estime par
# moindres carres sur les runs d'entrainement : c'est un modele "boite grise",
# 6 parametres a sens physique, contre des milliers pour un reseau.
#
# Longitudinal (le velo ne le modelise pas) : modele simple, lui aussi identifie :
#   dvx/dt - vy*r = c0 + c1*throttle + c2*throttle/vx + c3*brake + c4*vx^2
#
# Limite connue, et c'est voulu : pneu LINEAIRE = force laterale sans saturation.
# En glisse, le vrai pneu sature ; on s'attend donc a ce que ce modele se trompe la.

import numpy as np

DT = 0.02


def _features(s, a):
    vx = np.maximum(s[:, 0], 1.0)
    vy, r = s[:, 1], s[:, 2]
    delta, thr, brk = a[:, 0], a[:, 1], a[:, 2]
    lat = np.c_[vy / vx, r / vx, delta]
    lon = np.c_[np.ones_like(vx), thr, thr / vx, brk, vx ** 2]
    return lat, lon


def fit(s, a, s_next, dt=DT):
    """Estime les coefficients a, b, c par moindres carres."""
    ds = (s_next - s) / dt
    vx, vy, r = s[:, 0], s[:, 1], s[:, 2]
    lat, lon = _features(s, a)
    coef_a = np.linalg.lstsq(lat, ds[:, 1] + vx * r, rcond=None)[0]
    coef_b = np.linalg.lstsq(lat, ds[:, 2], rcond=None)[0]
    coef_c = np.linalg.lstsq(lon, ds[:, 0] - vy * r, rcond=None)[0]
    return {"a": coef_a, "b": coef_b, "c": coef_c}


def step(params, s, a, dt=DT):
    """Un pas d'Euler du modele identifie. s : (N, 3), a : (N, 3)."""
    vx, vy, r = s[:, 0], s[:, 1], s[:, 2]
    lat, lon = _features(s, a)
    dvy = lat @ params["a"] - vx * r
    dr = lat @ params["b"]
    dvx = lon @ params["c"] + vy * r
    return s + dt * np.c_[dvx, dvy, dr]


def fit_per_surface(s, a, s_next, surface):
    """Un jeu de coefficients par surface : la terre adhere moins que l'asphalte
    (coefficient du volant environ 2 fois plus faible sur la terre dans nos donnees)."""
    return {k: fit(s[surface == k], a[surface == k], s_next[surface == k]) for k in np.unique(surface)}


def sanity(params):
    """Verifie que les signes correspondent a la physique attendue."""
    a1, a2, a3 = params["a"]
    b1, b2, b3 = params["b"]
    checks = {
        "a1 < 0 (amortissement lateral)": a1 < 0,
        "b2 < 0 (amortissement en lacet)": b2 < 0,
        "a3 > 0 (braquer pousse lateralement)": a3 > 0,
        "b3 > 0 (braquer fait tourner)": b3 > 0,
    }
    return checks
