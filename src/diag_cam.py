# src/diag_cam.py
# DIAGNOSTIC du resultat negatif de jepa_cam.py : les latents du JEPA et de la
# reconstruction ne contiennent presque rien sur la vitesse (R2 lineaire ~ 0).
# Pourquoi ? Trois tests, sur les memes donnees et les memes decoupages :
#
#   1. BORNE HAUTE SUPERVISEE : le meme encodeur CNN, entraine DIRECTEMENT a
#      predire (vx, vy, r) a partir des 3 images. Si lui y arrive, l'information
#      est dans les images et ce sont les objectifs auto-supervises qui ne
#      l'extraient pas. S'il echoue aussi, ce sont les images qui sont trop pauvres.
#   2. LE MEME, AVEC UNE SEULE IMAGE : la vitesse ne se voit que dans le mouvement
#      entre images. Si 1 image fait presque aussi bien que 3, le modele se sert
#      d'autre chose (decor, forme de la route) que du mouvement.
#   3. PROBE NON LINEAIRE (petit MLP) sur les latents geles du JEPA et de la
#      reconstruction : l'information y est peut-etre, mais pas lineairement.
#
# Les modeles JEPA et reconstruction sont re-entraines (memes reglages que
# jepa_cam.py) et cette fois SAUVEGARDES dans outputs/models/.
#
# Lancer : python -W ignore src/diag_cam.py              (3 graines, environ 1 h sur GTX 1650)
#          python -W ignore src/diag_cam.py --seeds 1    (essai, environ 20 min)

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(HERE))
sys.path.append(HERE)
import config as C
import jepa_cam as J

COMPS = ["vx", "vy", "r"]
MODEL_DIR = os.path.join(C.OUT_DIR, "models")


class SupNet(nn.Module):
    """Meme architecture que l'encodeur de jepa_cam, + une tete de regression vers (vx, vy, r)."""
    def __init__(self, n_frames):
        super().__init__()
        self.n_frames = n_frames
        self.conv = nn.Sequential(
            nn.Conv2d(3 * n_frames, 32, 4, 2, 1), nn.GELU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.GELU(),
            nn.Conv2d(128, 128, 3, 1, 1), nn.GELU())
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(128 * 6 * 8, 256), nn.GELU(),
                                nn.Linear(256, J.LATENT), nn.GELU(), nn.Linear(J.LATENT, 3))

    def forward(self, obs9):
        x = obs9[:, -3 * self.n_frames:]          # 1 image = les 3 derniers canaux (image courante)
        return self.fc(self.conv(x))


def eval_idx(D, runs):
    i = J.starts(D, runs, 0)
    return i[D["S"][i, 0] > J.MIN_VX]


def train_sup(G, D, idx, n_frames, seed, mu, sd):
    torch.manual_seed(seed)
    net = SupNet(n_frames).to(J.DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=J.LR, weight_decay=1e-5)
    Y = torch.tensor((D["S"] - mu) / sd, dtype=torch.float32, device=J.DEV)
    for ep in range(J.EPOCHS):
        t0 = time.time()
        perm = np.random.default_rng(seed + ep).permutation(idx)
        for k in range(0, len(perm) - J.BATCH + 1, J.BATCH):
            i = perm[k:k + J.BATCH]
            loss = F.mse_loss(net(G.obs(i)), Y[torch.as_tensor(i, device=J.DEV)])
            opt.zero_grad(); loss.backward(); opt.step()
        if ep in (0, J.EPOCHS - 1):
            print(f"    supervise {n_frames} image(s) ep {ep + 1:2d} : perte {loss.item():.4f}  ({time.time() - t0:.0f} s/epoque)", flush=True)
    return net.eval()


@torch.no_grad()
def predict_sup(net, G, i, mu, sd):
    out = [net(G.obs(i[k:k + 512])).cpu().numpy() for k in range(0, len(i), 512)]
    return np.concatenate(out) * sd + mu


def r2s(y, p):
    return [r2_score(y[:, c], p[:, c]) for c in range(3)]


def run_seed(D, seed):
    train_runs, test_in, test_out = J.split(D, seed)
    tr_idx = np.where(np.isin(D["run"], list(train_runs)))[0]
    a_mean, a_std = D["A"][tr_idx].mean(0), D["A"][tr_idx].std(0) + 1e-6
    G = J.GPUData(D, a_mean, a_std)
    tests = {"en distribution": test_in, "glisse": test_out}
    print(f"[graine {seed}] {len(train_runs)} runs d'entrainement", flush=True)

    ftr = eval_idx(D, train_runs)
    ftr_probe = np.random.default_rng(0).choice(ftr, min(len(ftr), 40000), replace=False)
    mu, sd = D["S"][ftr].mean(0), D["S"][ftr].std(0) + 1e-9
    res = {}

    # 1 et 2 : bornes hautes supervisees (1 image et 3 images)
    for nf in [1, 3]:
        net = train_sup(G, D, ftr, nf, seed, mu, sd)
        for name, runs in tests.items():
            i = eval_idx(D, runs)
            res[(name, f"supervise, {nf} image(s)")] = r2s(D["S"][i], predict_sup(net, G, i, mu, sd))
        del net

    # 3 : JEPA et reconstruction, probes lineaire ET non lineaire sur latents geles
    os.makedirs(MODEL_DIR, exist_ok=True)
    for kind in ["jepa", "recon"]:
        m = J.train(kind, G, J.starts(D, train_runs, J.K_TRAIN), seed)
        torch.save(m.state_dict(), os.path.join(MODEL_DIR, f"{kind}_seed{seed}.pt"))
        ztr = J.encode(m, G, ftr_probe)
        ytr = (D["S"][ftr_probe] - mu) / sd
        lin = Ridge(alpha=1.0).fit(ztr, ytr)
        zm, zs = ztr.mean(0), ztr.std(0) + 1e-6
        mlp = MLPRegressor(hidden_layer_sizes=(256, 256), max_iter=200, early_stopping=True,
                           random_state=seed).fit((ztr - zm) / zs, ytr)
        for name, runs in tests.items():
            i = eval_idx(D, runs)
            z = J.encode(m, G, i)
            res[(name, f"{kind}, probe lineaire")] = r2s(D["S"][i], lin.predict(z) * sd + mu)
            res[(name, f"{kind}, probe MLP")] = r2s(D["S"][i], mlp.predict((z - zm) / zs) * sd + mu)
        del m
    del G
    if J.DEV.type == "cuda":
        torch.cuda.empty_cache()
    return res


ORDER = ["supervise, 3 image(s)", "supervise, 1 image(s)", "jepa, probe MLP", "recon, probe MLP",
         "jepa, probe lineaire", "recon, probe lineaire"]


def summarize(all_res):
    n = len(all_res)
    print(f"\n################ DIAGNOSTIC APRES {n} GRAINE(S) : R2 (ecart-type) ################")
    rows = []
    for test in ["en distribution", "glisse"]:
        print(f"=== Test {test}")
        print(f"  {'':26s}" + "".join(f"{c:>16s}" for c in COMPS))
        for name in ORDER:
            v = np.array([r[(test, name)] for r in all_res])
            print(f"  {name:26s}" + "".join(f"{v[:, c].mean():>10.3f} ({v[:, c].std():.2f})" for c in range(3)))
            for c, comp in enumerate(COMPS):
                rows.append([test, name, comp, v[:, c].mean(), v[:, c].std(), n])
    pd.DataFrame(rows, columns=["test", "methode", "composante", "r2", "ecart_type", "graines"]).to_csv(
        os.path.join(C.OUT_DIR, "diag_cam.csv"), index=False)
    print(f"[ok] -> {os.path.join(C.OUT_DIR, 'diag_cam.csv')}", flush=True)


def main():
    n_seeds = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else 3
    print(f"[calcul] {J.DEV}" + (f" ({torch.cuda.get_device_name(0)})" if J.DEV.type == "cuda" else
                                 " : pas de GPU detecte, ce sera tres lent"), flush=True)
    D = J.load()
    all_res = []
    for seed in range(n_seeds):
        all_res.append(run_seed(D, seed))
        summarize(all_res)                   # bilan apres chaque graine


if __name__ == "__main__":
    main()
