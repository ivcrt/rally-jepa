# src/eval_state.py
# Evaluation multi-pas DANS L'ESPACE D'ETAT, sur les vraies donnees BeamNG.
# Remplace l'ancien eval_rollout.py, qui comparait toujours a l'etat t+1
# et ne mesurait donc pas une vraie prediction a plusieurs pas.
#
# Protocole :
#   - entrainement : 80 % des groupes asphalte + terre ;
#   - test "en distribution" : les 20 % restants ;
#   - test "hors distribution" : toute la glisse provoquee (jamais vue) ;
#   - rollout : on part d'un etat reel, on fait avancer chaque modele avec la
#     VRAIE sequence d'actions, et on compare a la trajectoire reelle ;
#   - plusieurs decoupages (graines) pour avoir une moyenne et un ecart-type.
#
# Modeles compares (tous recoivent la surface : 0 asphalte, 1 terre) :
#   persistance : rien ne change, le plancher a battre ;
#   lineaire    : s_t+1 - s_t = A [s_t, a_t, surface] + b ;
#   mlp         : meme entree, reseau 2 x 128 (prediction directe, pas de latent) ;
#   velo        : modele velo lineaire identifie, un jeu de coefficients par surface.
#
# Lancer : python src/eval_state.py

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(HERE))
sys.path.append(HERE)
import config as C
import real_data as R
import bicycle as B

HORIZON = 50                        # 50 pas = 1 s
REPORT = [5, 25, 50]                # 0,1 s / 0,5 s / 1 s
SEEDS = [0, 1, 2]
SURFACE = {"asphalte": 0, "terre": 1, "glisse": 1}   # la glisse est sur la terre (Utah)


def add_surface(df):
    df["surface"] = df.regime.map(SURFACE).astype(float)
    return df


def transitions_with_surface(df, groups):
    d = df[df.group.isin(groups)]
    s, a, sn = R.transitions(d, groups)
    # meme filtre que R.transitions, applique a la colonne surface
    same = (d.run.to_numpy()[1:] == d.run.to_numpy()[:-1]) & (np.diff(d.t.to_numpy()) == 1)
    ok = same & d.valid.to_numpy()[:-1] & d.valid.to_numpy()[1:]
    surf = d.surface.to_numpy()[:-1][ok]
    return s, a, sn, surf


def segments_with_surface(df, groups):
    s0, acts, truth, surf = [], [], [], []
    for g in groups:
        d = df[df.group == g]
        if d.empty:
            continue
        a0, a1, a2 = R.segments(d, {g}, HORIZON)
        if len(a0):
            s0.append(a0); acts.append(a1); truth.append(a2)
            surf.append(np.full(len(a0), d.surface.iloc[0]))
    return np.concatenate(s0), np.concatenate(acts), np.concatenate(truth), np.concatenate(surf)


class Learned:
    """Prediction de l'increment d'etat a partir de [etat, action, surface]."""
    def __init__(self, kind, seed):
        self.kind, self.seed = kind, seed

    def fit(self, s, a, sn, surf):
        X = np.c_[s, a, surf]
        Y = sn - s
        self.mx, self.sx = X.mean(0), X.std(0) + 1e-9
        self.my, self.sy = Y.mean(0), Y.std(0) + 1e-9
        if self.kind == "lineaire":
            self.m = LinearRegression()
        else:
            self.m = MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=80, early_stopping=True,
                                  random_state=self.seed)
        self.m.fit((X - self.mx) / self.sx, (Y - self.my) / self.sy)
        return self

    def step(self, s, a, surf):
        X = (np.c_[s, a, surf] - self.mx) / self.sx
        return s + self.m.predict(X) * self.sy + self.my


def rollout(step_fn, s0, acts):
    """Fait avancer un modele sur tout l'horizon avec les vraies actions."""
    s, out = s0.copy(), []
    for h in range(acts.shape[1]):
        s = step_fn(s, acts[:, h])
        s[:, 0] = np.maximum(s[:, 0], 0.0)
        out.append(s.copy())
    return np.stack(out, axis=1)          # (N, H, 3)


def rmse_by_horizon(pred, truth):
    return np.sqrt(((pred - truth) ** 2).mean(axis=0))   # (H, 3)


def run_seed(df, seed):
    train, test_in, test_out = R.split_groups(df, seed=seed)
    s, a, sn, surf = transitions_with_surface(df, train)

    bike = B.fit_per_surface(s, a, sn, surf)
    models = {
        "persistance": lambda s_, a_, f_: s_,
        "lineaire": Learned("lineaire", seed).fit(s, a, sn, surf).step,
        "mlp": Learned("mlp", seed).fit(s, a, sn, surf).step,
        "velo": None,
    }
    res = {}
    for name_test, groups in [("en distribution", test_in), ("glisse (hors distribution)", test_out)]:
        s0, acts, truth, f = segments_with_surface(df, groups)
        for name, m in models.items():
            if name == "velo":
                def step_fn(x, u, f=f):
                    out = np.empty_like(x)
                    for k, p in bike.items():
                        sel = f == k
                        out[sel] = B.step(p, x[sel], u[sel])
                    return out
            else:
                step_fn = lambda x, u, m=m, f=f: m(x, u, f)
            res[(name_test, name)] = rmse_by_horizon(rollout(step_fn, s0, acts), truth)
        res[(name_test, "_n")] = len(s0)
    return res, bike


def main():
    df = add_surface(R.load_all())
    all_res = []
    for seed in SEEDS:
        res, bike = run_seed(df, seed)
        all_res.append(res)
        if seed == SEEDS[0]:
            for k, p in bike.items():
                nom = "asphalte" if k == 0 else "terre"
                ok = all(B.sanity(p).values())
                print(f"[velo] {nom} : a = {np.round(p['a'], 4)}, b = {np.round(p['b'], 4)}, signes physiques {'OK' if ok else 'INCOHERENTS'}")
        print(f"[graine {seed}] fait")

    names = ["persistance", "lineaire", "mlp", "velo"]
    rows = []
    for test in ["en distribution", "glisse (hors distribution)"]:
        n = int(np.mean([r[(test, "_n")] for r in all_res]))
        print(f"\n=== Test {test} ({n} departs de rollout en moyenne) ===")
        print("RMSE moyenne sur 3 graines (ecart-type), par horizon")
        for comp, unit, j in [("vy", "m/s", 1), ("r", "rad/s", 2), ("vx", "m/s", 0)]:
            print(f"  {comp} ({unit})")
            for name in names:
                vals = np.array([r[(test, name)][:, j] for r in all_res])     # (graines, H)
                cells = [f"{vals[:, h - 1].mean():.3f} ({vals[:, h - 1].std():.3f})" for h in REPORT]
                print(f"    {name:12s} " + "   ".join(f"{h * C.DT:.1f}s: {c}" for h, c in zip(REPORT, cells)))
                for h in range(HORIZON):
                    rows.append([test, comp, name, (h + 1) * C.DT, vals[:, h].mean(), vals[:, h].std()])

    out = pd.DataFrame(rows, columns=["test", "composante", "modele", "horizon_s", "rmse", "ecart_type"])
    os.makedirs(C.OUT_DIR, exist_ok=True)
    path = os.path.join(C.OUT_DIR, "eval_state.csv")
    out.to_csv(path, index=False)
    print(f"\n[ok] resultats complets -> {path}")


if __name__ == "__main__":
    main()
