# data/make_dataset.py
# Deux roles :
#   1. Generer des donnees SYNTHETIQUES (modele velo lineaire) pour faire tourner
#      tout le repo immediatement, sans BeamNG.
#   2. Transformer des runs bruts (raw CSV) en dataset fenetre avec id de groupe.
#
# Le modele velo sert aussi de BASELINE ANALYTIQUE (cf. src/baselines.py).
# Sur donnees synthetiques il est "parfait" (c'est le generateur). Sur vraies
# donnees BeamNG il devient une approximation, ce qui est le but de la comparaison.

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C


# --- Modele velo lineaire (repere voiture) ---
# Parametres type voiture de rallye legere. Tu pourras les recaler.
BIKE = dict(
    m=1200.0,    # masse (kg)
    Iz=1500.0,   # inertie en lacet (kg.m2)
    lf=1.2,      # distance CG -> essieu avant (m)
    lr=1.4,      # distance CG -> essieu arriere (m)
    Cf=80000.0,  # rigidite de derive avant (N/rad)
    Cr=90000.0,  # rigidite de derive arriere (N/rad)
)


def bike_step(s, a, dt=C.DT, p=BIKE):
    """Un pas du modele velo. s=[vx,vy,r], a=[delta,ax_cmd]. Retourne s_{t+1}."""
    vx, vy, r = s
    delta, ax = a
    vx = max(vx, 1.0)  # eviter la division par ~0 a l'arret

    # Angles de derive avant / arriere
    af = delta - (vy + p["lf"] * r) / vx
    ar = -(vy - p["lr"] * r) / vx

    # Forces laterales (pneu lineaire)
    Fyf = p["Cf"] * af
    Fyr = p["Cr"] * ar

    # Derivees (dynamique laterale + lacet, longi commandee)
    vx_dot = ax + vy * r
    vy_dot = (Fyf + Fyr) / p["m"] - vx * r
    r_dot = (p["lf"] * Fyf - p["lr"] * Fyr) / p["Iz"]

    vx2 = vx + dt * vx_dot
    vy2 = vy + dt * vy_dot
    r2 = r + dt * r_dot
    return np.array([vx2, vy2, r2], dtype=np.float32)


def smooth_controls(n, rng):
    """Sequence de commandes lissee (volant + accel) pour un run."""
    # Bruit blanc filtre par moyenne glissante = signal lisse et continu.
    raw_delta = rng.normal(0, 0.25, n)
    raw_ax = rng.normal(0.5, 1.5, n)
    k = 25
    ker = np.ones(k) / k
    delta = np.convolve(raw_delta, ker, mode="same") * 0.5   # rad
    ax = np.convolve(raw_ax, ker, mode="same")               # m/s2
    return np.stack([delta, ax], axis=1).astype(np.float32)


def generate_synthetic():
    """Genere des runs synthetiques et ecrit le raw CSV."""
    rng = np.random.default_rng(C.SEED)
    rows = []
    for run_id in range(C.SYN_N_RUNS):
        a_seq = smooth_controls(C.SYN_RUN_LEN, rng)
        s = np.array([rng.uniform(8, 25), 0.0, 0.0], dtype=np.float32)  # demarrage
        for t in range(C.SYN_RUN_LEN):
            a = a_seq[t]
            # un peu de bruit de process (capteurs imparfaits)
            s_noisy = s + rng.normal(0, 0.01, 3).astype(np.float32)
            rows.append([run_id, t, *s_noisy, *a])
            s = bike_step(s, a)
    arr = np.array(rows, dtype=np.float32)
    header = "run_id,t," + ",".join(C.STATE_COLS + C.ACTION_COLS)
    os.makedirs(C.DATA_DIR, exist_ok=True)
    np.savetxt(C.RAW_CSV, arr, delimiter=",", header=header, comments="")
    print(f"[synth] {C.SYN_N_RUNS} runs ecrits dans {C.RAW_CSV}")


def build_windows():
    """Lit le raw CSV et construit le dataset fenetre (.npz)."""
    import pandas as pd
    df = pd.read_csv(C.RAW_CSV)

    H = C.CONTEXT
    X_state, X_action, Y_next, groups, beta = [], [], [], [], []

    for run_id, g in df.groupby("run_id"):
        g = g.sort_values("t").reset_index(drop=True)
        s = g[C.STATE_COLS].to_numpy(np.float32)
        a = g[C.ACTION_COLS].to_numpy(np.float32)
        n = len(g)
        for i in range(H - 1, n - 1):
            # Voiture quasi a l'arret : derive non definie, on saute.
            if s[i, 0] < C.MIN_SPEED or s[i + 1, 0] < C.MIN_SPEED:
                continue
            X_state.append(s[i - H + 1: i + 1])   # contexte (H, state_dim)
            X_action.append(a[i])                 # action au temps i
            Y_next.append(s[i + 1])               # etat suivant (cible)
            groups.append(int(run_id))
            vx, vy = s[i + 1, 0], s[i + 1, 1]
            beta.append(np.arctan2(vy, max(vx, 1e-3)))  # angle de derive (probe)

    X_state = np.asarray(X_state, np.float32)
    X_action = np.asarray(X_action, np.float32)
    Y_next = np.asarray(Y_next, np.float32)
    groups = np.asarray(groups, np.int64)
    beta = np.asarray(beta, np.float32)

    # Nettoyage NaN (trous capteurs sur vraies donnees). Lecon CL-Drive.
    for arr in (X_state, X_action, Y_next):
        np.nan_to_num(arr, copy=False)

    np.savez_compressed(
        C.DATASET_NPZ,
        X_state=X_state, X_action=X_action, Y_next=Y_next,
        groups=groups, beta=beta,
    )
    print(f"[dataset] {len(groups)} fenetres, {len(np.unique(groups))} runs -> {C.DATASET_NPZ}")


if __name__ == "__main__":
    # Par defaut : genere du synthetique puis fenetre.
    # Sur vraies donnees BeamNG : commente generate_synthetic() et fais tourner
    # collect_beamng.py d'abord (il ecrit le meme format de raw CSV).
    if "--no-synth" not in sys.argv:
        generate_synthetic()
    build_windows()
