# src/real_data.py
# Chargement des 4 collectes BeamNG et decoupage train / test PAR GROUPE.
#
#   regime "asphalte" : data/raw_runs.csv        (nuit 1, west_coast_usa)
#   regime "terre"    : data/raw_runs_dirt.csv   (nuit 2, Utah, IA prudente)
#   regime "glisse"   : data/raw_runs_slide.csv + data/raw_runs_slide2.csv
#                       (nuits 3 et 4, vitesse imposee)
#
# Attention fuite : les nuits 3 et 4 ont rejoue la meme graine, donc le run k
# des deux nuits a les memes parametres et un debut de trajectoire tres proche.
# Ils partagent donc le meme groupe ("G<k>") : jamais l'un en train, l'autre en test.

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

COLS = ["run_id", "t", "vx", "vy", "r", "delta", "throttle", "brake"]
STATE = ["vx", "vy", "r"]
ACTION = ["delta", "throttle", "brake"]   # volant (degres), accelerateur, frein (0..1)

MIN_VX = 5.0          # m/s : en dessous, l'angle de derive mesure surtout des manoeuvres lentes
SKIP_START = 250      # pas : on ignore les 5 premieres secondes (demarrage, demi-tour)

SOURCES = [
    ("asphalte", "raw_runs.csv", "A"),
    ("terre", "raw_runs_dirt.csv", "D"),
    ("glisse", "raw_runs_slide.csv", "G"),
    ("glisse", "raw_runs_slide2.csv", "G"),   # meme prefixe : jumeaux de la nuit 3
]


def load_all():
    """Toutes les collectes dans un seul tableau, avec regime, groupe et run unique."""
    parts = []
    for regime, fname, prefix in SOURCES:
        path = os.path.join(C.DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"[real_data] absent, ignore : {fname}")
            continue
        df = pd.read_csv(path, usecols=COLS)
        df["regime"] = regime
        df["group"] = prefix + df["run_id"].astype(str)
        df["run"] = fname + ":" + df["run_id"].astype(str)   # trajectoire unique
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["run", "t"]).reset_index(drop=True)
    df["valid"] = (df.vx > MIN_VX) & (df.t >= SKIP_START)
    return df


def split_groups(df, seed=C.SEED, test_frac=0.2):
    """Test en distribution : 20 % des groupes asphalte + terre.
    Test hors distribution : toute la glisse (jamais vue a l'entrainement)."""
    rng = np.random.default_rng(seed)
    normal = np.array(sorted(df.loc[df.regime != "glisse", "group"].unique()))
    rng.shuffle(normal)
    n_test = int(round(test_frac * len(normal)))
    test_in = set(normal[:n_test])
    train = set(normal[n_test:])
    test_out = set(df.loc[df.regime == "glisse", "group"].unique())
    assert train.isdisjoint(test_in) and train.isdisjoint(test_out)
    return train, test_in, test_out


def transitions(df, groups):
    """Paires (s_t, a_t, s_t+1) consecutives et valides, pour les groupes donnes."""
    d = df[df.group.isin(groups)]
    s = d[STATE].to_numpy(float)
    a = d[ACTION].to_numpy(float)
    same = (d.run.to_numpy()[1:] == d.run.to_numpy()[:-1]) & (np.diff(d.t.to_numpy()) == 1)
    ok = same & d.valid.to_numpy()[:-1] & d.valid.to_numpy()[1:]
    return s[:-1][ok], a[:-1][ok], s[1:][ok]


def segments(df, groups, horizon, stride=25):
    """Departs de rollout : fenetres de horizon+1 pas consecutifs et tous valides."""
    out_s0, out_a, out_s = [], [], []
    for _, r in df[df.group.isin(groups)].groupby("run", sort=False):
        s = r[STATE].to_numpy(float)
        a = r[ACTION].to_numpy(float)
        v = r.valid.to_numpy() & np.r_[True, np.diff(r.t.to_numpy()) == 1]
        for i in range(0, len(r) - horizon, stride):
            if v[i:i + horizon + 1].all():
                out_s0.append(s[i]); out_a.append(a[i:i + horizon]); out_s.append(s[i + 1:i + horizon + 1])
    return np.array(out_s0), np.array(out_a), np.array(out_s)
