# src/jepa_cam.py
# World models a partir de la CAMERA : JEPA contre reconstruction de pixels.
#
# QUESTION : vaut-il mieux predire le futur dans un espace de representation
# (JEPA, these de LeCun) ou predire les pixels futurs (world model generatif,
# dans l'esprit de Ha & Schmidhuber 2018) ? A architecture et budget egaux.
#
# DONNEES : data/raw_runs_cam.csv + data/frames_cam/run_XXXX.npz (collect_cam.py)
#   - 10 images/s, 64 x 64, capot rogne -> 48 x 64 ;
#   - observation o_t = 3 images consecutives (0,2 s) ;
#   - action a_t = moyenne (volant, accelerateur, frein) entre deux images.
#
# DECOUPAGE (par run, pas de fuite) :
#   - entrainement : 80 % des runs "limit" (conduite normale) ;
#   - test en distribution : les 20 % restants ;
#   - test hors distribution : tous les runs "set" (vitesse imposee, glisse provoquee).
#
# MODELES (identiques sauf la perte) :
#   encodeur CNN -> latent 64 ; predicteur MLP (latent, action) -> latent suivant ;
#   entrainement sur un rollout latent de K = 8 pas (0,8 s) : c'est ce qui avait
#   fait la difference sur les donnees d'etat.
#   JEPA  : perte = MSE(latent predit, encodeur cible EMA(o_t+k)) + VICReg.
#   RECON : un decodeur reconstruit l'image a partir du latent ; perte = MSE pixels
#           sur l'image courante et sur chaque image predite.
#
# EVALUATION (encodeurs geles, aucun reglage sur le test) :
#   - probe lineaire latent -> (vx, vy, r, derive) : R2 ;
#   - rollout latent sur 1 s avec les vraies actions, lu par le meme probe : RMSE.
#
# Lancer : python src/jepa_cam.py            (3 graines)
#          python src/jepa_cam.py --seeds 1  (essai plus rapide)

import os
import sys
import copy
import glob
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(HERE))
import config as C

CSV = os.path.join(C.DATA_DIR, "raw_runs_cam.csv")
FRAMES = os.path.join(C.DATA_DIR, "frames_cam")
CROP = 48            # on garde les 48 lignes du haut (le capot est en bas)
CTX = 3              # images par observation
STEP = 5             # pas de simulation entre deux images
SKIP_START = 250     # on ignore les 5 premieres secondes de chaque run
MIN_VX = 5.0         # pour les probes et le rollout : derive definie seulement en roulant
K_TRAIN = 8          # rollout latent a l'entrainement (0,8 s)
K_EVAL = 10          # rollout a l'evaluation (1 s)
REPORT = [1, 5, 10]  # 0,1 s / 0,5 s / 1 s
LATENT = 64
EPOCHS = 20
BATCH = 128
LR = 3e-4
EMA, W_VAR, W_COV = 0.99, 1.0, 0.04
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- donnees

def load():
    t0 = time.time()
    df = pd.read_csv(CSV, usecols=["run_id", "t", "vx", "vy", "r", "delta", "throttle", "brake", "speed_mode"])
    X, S, A, run, mode, link, ok = [], [], [], [], [], [], []
    for path in sorted(glob.glob(os.path.join(FRAMES, "run_*.npz"))):
        rid = int(os.path.basename(path)[4:8])
        d = df[df.run_id == rid].sort_values("t")
        z = np.load(path)
        fr, st = z["frames"], z["steps"]
        if len(d) == 0 or len(fr) < CTX + 2:
            continue
        tt = d.t.to_numpy()
        sel = st < tt.max() - STEP
        fr, st = fr[sel], st[sel]
        pos = np.searchsorted(tt, st)
        good = (pos < len(tt)) & (tt[np.minimum(pos, len(tt) - 1)] == st)
        fr, st, pos = fr[good], st[good], pos[good]
        sv = d[["vx", "vy", "r"]].to_numpy(np.float32)
        av = d[["delta", "throttle", "brake"]].to_numpy(np.float32)
        a = np.stack([av[p:p + STEP].mean(0) for p in pos])          # action entre deux images
        X.append(fr[:, :CROP]); S.append(sv[pos]); A.append(a)
        run.append(np.full(len(st), rid)); mode.append(d.speed_mode.iloc[0])
        link.append(np.r_[False, np.diff(st) == STEP])
        ok.append(st >= SKIP_START)
    D = dict(X=np.concatenate(X), S=np.concatenate(S), A=np.concatenate(A), run=np.concatenate(run),
             link=np.concatenate(link), ok=np.concatenate(ok))
    D["mode"] = {r: m for r, m in zip([x[0] for x in run], mode)}
    streak = np.zeros(len(D["X"]), np.int64)          # images valides consecutives finissant en i
    for i in range(len(streak)):
        if D["ok"][i]:
            streak[i] = streak[i - 1] + 1 if (i > 0 and D["link"][i]) else 1
    D["streak"] = streak
    print(f"[donnees] {len(D['X'])} images, {len(D['mode'])} runs, chargees en {time.time() - t0:.0f} s", flush=True)
    return D


def split(D, seed):
    rng = np.random.default_rng(seed)
    normal = np.array(sorted(r for r, m in D["mode"].items() if m == "limit"))
    rng.shuffle(normal)
    n_test = int(round(0.2 * len(normal)))
    test_in, train = set(normal[:n_test]), set(normal[n_test:])
    test_out = {r for r, m in D["mode"].items() if m == "set"}
    assert train.isdisjoint(test_in) and train.isdisjoint(test_out)
    return train, test_in, test_out


def starts(D, runs, K):
    """Indices i tels que o_i, ..., o_i+K soient complets et consecutifs, dans les runs donnes."""
    n = len(D["streak"])
    i = np.where(D["streak"] >= CTX + K)[0] - K
    i = i[(i >= 0) & (i + K < n)]
    return i[np.isin(D["run"][i], list(runs))]


class GPUData:
    """Images sur le GPU (uint8) pour fabriquer les observations sans aller-retour."""
    def __init__(self, D, a_mean, a_std):
        self.X = torch.tensor(D["X"]).permute(0, 3, 1, 2).contiguous().to(DEV)   # (N, 3, 48, 64) uint8
        self.A = torch.tensor((D["A"] - a_mean) / a_std, dtype=torch.float32, device=DEV)

    def obs(self, i):
        """(B, 9, 48, 64) : les CTX dernieres images, en [0, 1]."""
        i = torch.as_tensor(i, device=DEV)
        idx = i[:, None] + torch.arange(-CTX + 1, 1, device=DEV)
        return self.X[idx].flatten(1, 2).float() / 255.0

    def frame(self, i):
        return self.X[torch.as_tensor(i, device=DEV)].float() / 255.0


# ---------------------------------------------------------------- modeles

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3 * CTX, 32, 4, 2, 1), nn.GELU(),     # 24 x 32
            nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),          # 12 x 16
            nn.Conv2d(64, 128, 4, 2, 1), nn.GELU(),         # 6 x 8
            nn.Conv2d(128, 128, 3, 1, 1), nn.GELU())
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(128 * 6 * 8, 256), nn.GELU(), nn.Linear(256, LATENT))

    def forward(self, x):
        return self.fc(self.conv(x))


class Decoder(nn.Module):
    """Pour le modele par reconstruction : latent -> image courante (3, 48, 64)."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(LATENT, 256), nn.GELU(), nn.Linear(256, 128 * 6 * 8), nn.GELU())
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GELU(),   # 12 x 16
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),    # 24 x 32
            nn.ConvTranspose2d(32, 3, 4, 2, 1))                # 48 x 64

    def forward(self, z):
        return torch.sigmoid(self.deconv(self.fc(z).view(-1, 128, 6, 8)))


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(LATENT + 3, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(),
                                 nn.Linear(256, LATENT))

    def forward(self, z, a):
        return z + self.net(torch.cat([z, a], 1))      # on predit le changement de latent


class WorldModel(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.enc, self.pred = Encoder(), Predictor()
        if kind == "jepa":
            self.tgt = copy.deepcopy(self.enc)
            for p in self.tgt.parameters():
                p.requires_grad = False
        else:
            self.dec = Decoder()

    def eval_encoder(self):
        # JEPA : on lit l'espace de l'encodeur cible, celui dans lequel le predicteur predit.
        return self.tgt if self.kind == "jepa" else self.enc


def vicreg(z):
    std = torch.sqrt(z.var(0) + 1e-4)
    zc = z - z.mean(0)
    cov = (zc.T @ zc) / (len(z) - 1)
    return F.relu(1 - std).mean(), ((cov - torch.diag(torch.diag(cov))) ** 2).sum() / z.shape[1]


@torch.no_grad()
def eff_rank(z):
    s = torch.linalg.svdvals(z - z.mean(0))
    p = s / s.sum()
    return float(torch.exp(-(p * torch.log(p + 1e-12)).sum()))


def train(kind, G, idx, seed):
    torch.manual_seed(seed)
    m = WorldModel(kind).to(DEV)
    params = [p for n, p in m.named_parameters() if not n.startswith("tgt.")]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-5)
    for ep in range(EPOCHS):
        t0 = time.time()
        perm = np.random.default_rng(seed + ep).permutation(idx)
        tot = 0.0
        for k in range(0, len(perm) - BATCH + 1, BATCH):
            i = perm[k:k + BATCH]
            z0 = m.enc(G.obs(i))
            z, loss = z0, 0.0
            if kind == "recon":
                loss = F.mse_loss(m.dec(z0), G.frame(i))
            for j in range(1, K_TRAIN + 1):
                z = m.pred(z, G.A[torch.as_tensor(i + j - 1, device=DEV)])
                if kind == "jepa":
                    with torch.no_grad():
                        zt = m.tgt(G.obs(i + j))
                    loss = loss + F.mse_loss(z, zt) / K_TRAIN
                else:
                    loss = loss + F.mse_loss(m.dec(z), G.frame(i + j)) / K_TRAIN
            if kind == "jepa":
                var, cov = vicreg(z0)
                loss = loss + W_VAR * var + W_COV * cov
            opt.zero_grad(); loss.backward(); opt.step()
            if kind == "jepa":
                with torch.no_grad():
                    for pt, p in zip(m.tgt.parameters(), m.enc.parameters()):
                        pt.mul_(EMA).add_(p, alpha=1 - EMA)
            tot += float(loss)
        if ep in (0, EPOCHS // 2, EPOCHS - 1):
            with torch.no_grad():
                zz = m.eval_encoder()(G.obs(idx[:2000]))
            print(f"    {kind:5s} ep {ep + 1:2d} : perte {tot / max(1, len(perm) // BATCH):.4f}  "
                  f"rang effectif {eff_rank(zz):.1f}/{LATENT}  ({time.time() - t0:.0f} s/epoque)", flush=True)
    return m.eval()


# ---------------------------------------------------------------- evaluation

@torch.no_grad()
def encode(m, G, i):
    enc = m.eval_encoder()
    return np.concatenate([enc(G.obs(i[k:k + 512])).cpu().numpy() for k in range(0, len(i), 512)])


@torch.no_grad()
def roll(m, G, i):
    """Latents predits a 1..K_EVAL pas, a partir de o_i et des vraies actions : (N, K, LATENT)."""
    out = []
    for k in range(0, len(i), 512):
        b = i[k:k + 512]
        z = m.eval_encoder()(G.obs(b)); zs = []
        for j in range(K_EVAL):
            z = m.pred(z, G.A[torch.as_tensor(b + j, device=DEV)])
            zs.append(z.cpu().numpy())
        out.append(np.stack(zs, 1))
    return np.concatenate(out)


def targets(D, i):
    s = D["S"][i]
    return np.c_[s, np.arctan2(s[:, 1], np.maximum(s[:, 0], 1e-3))]      # vx, vy, r, derive


def evaluate(m, D, G, train, tests):
    """Probe (Ridge) entraine sur le train, puis R2 et rollout lu par ce probe."""
    tr = starts(D, train, 0)
    tr = tr[D["S"][tr, 0] > MIN_VX]
    tr = np.random.default_rng(0).choice(tr, min(len(tr), 40000), replace=False)
    ztr, ytr = encode(m, G, tr), targets(D, tr)
    mu, sd = ytr.mean(0), ytr.std(0) + 1e-9
    probe = Ridge(alpha=1.0).fit(ztr, (ytr - mu) / sd)
    read = lambda z: probe.predict(z) * sd + mu
    res = {}
    for name, runs in tests.items():
        i = starts(D, runs, 0); i = i[D["S"][i, 0] > MIN_VX]
        y = targets(D, i); yp = read(encode(m, G, i))
        res[(name, "r2")] = [r2_score(y[:, c], yp[:, c]) for c in range(4)]
        j = starts(D, runs, K_EVAL); j = j[D["S"][j, 0] > MIN_VX]
        zr = roll(m, G, j)
        pred = read(zr.reshape(-1, LATENT)).reshape(len(j), K_EVAL, 4)[:, :, :3]
        truth = np.stack([D["S"][j + k] for k in range(1, K_EVAL + 1)], 1)
        res[(name, "rollout")] = np.sqrt(((pred - truth) ** 2).mean(0))                 # (K, 3)
        p0 = read(encode(m, G, j))[:, None, :3]                                           # etat lu a t, fige
        res[(name, "lu_fige")] = np.sqrt(((p0 - truth) ** 2).mean(0))
        res[(name, "persistance")] = np.sqrt(((D["S"][j][:, None] - truth) ** 2).mean(0))  # etat VRAI a t, fige
        res[(name, "n")] = (len(i), len(j))
    return res


@torch.no_grad()
def save_recon_examples(m, D, G, runs, path):
    from PIL import Image
    i = starts(D, runs, K_EVAL)[::997][:6]
    if len(i) == 0:
        return
    z = m.enc(G.obs(i)); rows = []
    for j in [0, 5, 10]:
        if j:
            for _ in range(5):
                z = m.pred(z, G.A[torch.as_tensor(i + (j - 5) + _, device=DEV)])
        rec = (m.dec(z).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        real = D["X"][i + j]
        rows.append(np.concatenate([np.concatenate([a, b], 0) for a, b in zip(real, rec)], 1))
    Image.fromarray(np.concatenate(rows, 0)).resize((6 * 64 * 2, 3 * 96 * 2), Image.NEAREST).save(path)
    print(f"[recon] exemples (vrai en haut, reconstruit/predit en bas ; lignes : t, t+0,5 s, t+1 s) -> {path}")


# ---------------------------------------------------------------- principal

def main():
    n_seeds = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else 3
    print(f"[calcul] {DEV}" + (f" ({torch.cuda.get_device_name(0)})" if DEV.type == "cuda" else
                               " : pas de GPU detecte, ce sera tres lent"), flush=True)
    D = load()
    all_res = {"jepa": [], "recon": []}
    for seed in range(n_seeds):
        train_runs, test_in, test_out = split(D, seed)
        tr_idx = np.where(np.isin(D["run"], list(train_runs)))[0]
        a_mean, a_std = D["A"][tr_idx].mean(0), D["A"][tr_idx].std(0) + 1e-6
        G = GPUData(D, a_mean, a_std)
        idx = starts(D, train_runs, K_TRAIN)
        print(f"[graine {seed}] {len(train_runs)} runs d'entrainement, {len(idx)} sequences ; "
              f"tests : {len(test_in)} runs normaux, {len(test_out)} runs en glisse", flush=True)
        for kind in ["jepa", "recon"]:
            m = train(kind, G, idx, seed)
            all_res[kind].append(evaluate(m, D, G, train_runs, {"en distribution": test_in, "glisse": test_out}))
            if kind == "recon" and seed == 0:
                save_recon_examples(m, D, G, test_out, os.path.join(C.OUT_DIR, "recon_exemples.png"))
            del m
        del G
        if DEV.type == "cuda":
            torch.cuda.empty_cache()
        summarize(all_res, seed + 1)          # bilan enregistre apres CHAQUE graine


def summarize(all_res, n_seeds):
    print(f"\n################ BILAN APRES {n_seeds} GRAINE(S) ################")
    rows = []
    for test in ["en distribution", "glisse"]:
        print(f"\n=== Test {test}  (moyenne sur {n_seeds} graine(s), ecart-type entre parentheses)")
        print("  Probe lineaire, R2 :            vx       vy       r     derive")
        for kind in ["jepa", "recon"]:
            v = np.array([r[(test, "r2")] for r in all_res[kind]])
            print(f"    {kind:5s}                    " + "  ".join(f"{v[:, c].mean():.3f}({v[:, c].std():.2f})" for c in range(4)))
        print("  Rollout sur 1 s lu par le probe, RMSE :")
        for comp, j in [("vx (m/s)", 0), ("vy (m/s)", 1), ("r (rad/s)", 2)]:
            print(f"   {comp}")
            for name, key, src in [("persistance (etat vrai)", "persistance", "jepa"),
                                   ("jepa : etat lu a t, fige", "lu_fige", "jepa"),
                                   ("jepa : rollout latent", "rollout", "jepa"),
                                   ("recon : etat lu a t, fige", "lu_fige", "recon"),
                                   ("recon : rollout latent", "rollout", "recon")]:
                v = np.array([r[(test, key)][:, j] for r in all_res[src]])
                print(f"     {name:28s} " + "  ".join(f"{h / 10:.1f}s: {v[:, h - 1].mean():.3f} ({v[:, h - 1].std():.3f})" for h in REPORT))
                for h in range(K_EVAL):
                    rows.append([test, comp.split()[0], name, (h + 1) / 10, v[:, h].mean(), v[:, h].std()])
    pd.DataFrame(rows, columns=["test", "composante", "modele", "horizon_s", "rmse", "ecart_type"]).to_csv(
        os.path.join(C.OUT_DIR, "eval_cam.csv"), index=False)
    print(f"\n[ok] bilan sur {n_seeds} graine(s) -> {os.path.join(C.OUT_DIR, 'eval_cam.csv')}", flush=True)


if __name__ == "__main__":
    main()
