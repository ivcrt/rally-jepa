# src/jepa_real.py
# World model JEPA sur les vraies donnees BeamNG, evalue dans le meme cadre que
# les baselines (src/eval_state.py), sur les memes departs de rollout.
#
# IDEE (a relier au cours JEPA) :
#   - l'OBSERVATION o_t n'est plus l'etat seul (3 nombres) mais une fenetre :
#     les H derniers etats, les H dernieres actions, et la surface. Il y a donc
#     quelque chose a resumer (dynamique recente, tendance, bruit) ;
#   - l'ENCODEUR E resume o_t en un latent z_t de 32 nombres ;
#   - le PREDICTEUR P predit z_t+1 a partir de z_t et de l'action a_t ;
#   - la CIBLE est E_cible(o_t+1), ou E_cible est une copie lente de E (EMA)
#     qu'on ne modifie jamais par gradient (stop-gradient) ;
#   - VICReg empeche l'EFFONDREMENT : sans lui, E pourrait sortir la meme
#     constante pour tout, et P la predirait parfaitement (perte nulle, modele inutile).
#   On ne revient JAMAIS aux grandeurs physiques pendant l'entrainement.
#
# EVALUATION :
#   - un DECODEUR D (latent -> etat physique) est entraine APRES coup, encodeur gele ;
#   - rollout : z = E(o_t) une seule fois, puis z = P(z, a) a chaque pas avec les
#     vraies actions, et on decode chaque z pour comparer a la vraie trajectoire ;
#   - baseline equitable "mlp_hist" : meme fenetre en entree, mais prediction
#     directe de l'etat suivant (pas de latent). Si le JEPA gagne, on saura si
#     c'est grace au latent ou seulement grace a l'historique ;
#   - probe : le latent contient-il l'angle de derive ? compare a un probe sur
#     l'entree brute (si le latent fait moins bien que l'entree, il perd de l'info).
#
# Lancer : python src/jepa_real.py --select   (choisit K sur la validation, puis evalue)
#          python src/jepa_real.py --k 8      (K impose)
#          python src/jepa_real.py --k 8 --control   (+ controle : MLP entraines sur rollout)
#          python src/jepa_real.py            (K = 1 seulement, version precedente)

import os
import sys
import copy
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
sys.path.append(HERE)
import config as C
import real_data as R
import bicycle as B
from eval_state import Learned, add_surface, rollout as rollout_state, transitions_with_surface, SEEDS

H = 10              # historique : 10 pas = 0,2 s
HORIZON = 50        # rollout sur 1 s
REPORT = [5, 25, 50]
LATENT, HID = 32, 128
EPOCHS, BATCH, LR = 15, 512, 1e-3
EMA, W_VAR, W_COV = 0.99, 1.0, 0.04
torch.set_num_threads(max(1, os.cpu_count() or 1))


# ---------------------------------------------------------------- donnees

class Block:
    """Les runs d'un ensemble de groupes, a plat, avec de quoi decouper des fenetres."""
    def __init__(self, df, groups):
        d = df[df.group.isin(groups)]
        self.S = d[R.STATE].to_numpy(np.float32)
        self.A = d[R.ACTION].to_numpy(np.float32)
        self.surf = d.surface.to_numpy(np.float32)
        run, t, ok = d.run.to_numpy(), d.t.to_numpy(), d.valid.to_numpy()
        link = np.r_[False, (run[1:] == run[:-1]) & (np.diff(t) == 1)]
        streak = np.zeros(len(d), np.int64)          # nb de pas valides consecutifs finissant en i
        for i in range(len(d)):
            if ok[i]:
                streak[i] = streak[i - 1] + 1 if (i > 0 and link[i]) else 1
        self.streak = streak

    def obs(self, idx):
        """o_i = [s_(i-H+1..i), a_(i-H..i-1), surface] aplati : (N, 6H+1)."""
        st = self.S[idx[:, None] + np.arange(-H + 1, 1)]
        ac = self.A[idx[:, None] + np.arange(-H, 0)]
        return np.concatenate([st.reshape(len(idx), -1), ac.reshape(len(idx), -1), self.surf[idx, None]], 1)

    def transition_idx(self):
        return np.where(self.streak >= H + 2)[0] - 1     # o_i et o_i+1 complets

    def segment_idx(self, stride=25):
        cand = np.where(self.streak >= H + 1 + HORIZON)[0] - HORIZON
        out = []
        for i in cand:                                     # un depart tous les `stride` pas au plus
            if not out or i - out[-1] >= stride:
                out.append(i)
        return np.array(out)


class Norm:
    def __init__(self, x):
        self.m, self.s = x.mean(0), x.std(0) + 1e-6
    def __call__(self, x):
        return (x - self.m) / self.s
    def inv(self, x):
        return x * self.s + self.m


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, o))


# ---------------------------------------------------------------- JEPA

class JEPA(nn.Module):
    def __init__(self, d_obs):
        super().__init__()
        self.enc = mlp(d_obs, HID, LATENT)
        self.act = mlp(4, 32, 16)                 # action a_t (3) + surface
        self.pred = mlp(LATENT + 16, HID, LATENT)
        self.tgt = copy.deepcopy(self.enc)
        for p in self.tgt.parameters():
            p.requires_grad = False

    def step(self, z, a):
        return self.pred(torch.cat([z, self.act(a)], 1))


def vicreg(z):
    std = torch.sqrt(z.var(0) + 1e-4)
    var = F.relu(1 - std).mean()
    zc = z - z.mean(0)
    cov = (zc.T @ zc) / (len(z) - 1)
    cov = ((cov - torch.diag(torch.diag(cov))) ** 2).sum() / z.shape[1]
    return var, cov


def eff_rank(z):
    """Rang effectif du latent : proche de 1 = effondrement, proche de LATENT = latent bien utilise."""
    s = torch.linalg.svdvals(z - z.mean(0))
    p = s / s.sum()
    return float(torch.exp(-(p * torch.log(p + 1e-12)).sum()))


def rollout_idx(blk, K):
    """Departs t tels que o_t, ..., o_t+K soient tous complets et consecutifs."""
    return np.where(blk.streak >= H + 1 + K)[0] - K


def train_jepa(blk, nobs, nact, seed, K=1):
    """K = 1 : le predicteur apprend un seul pas (version precedente).
    K > 1 : PERTE SUR UN ROLLOUT LATENT. Le predicteur enchaine ses propres
    predictions sur K pas, comme a l'evaluation, et chaque pas est compare a la
    cible E_cible(o_t+k). Il apprend ainsi a ne pas deriver quand on l'enchaine."""
    torch.manual_seed(seed)
    idx = rollout_idx(blk, K)
    T = lambda x: torch.tensor(x, dtype=torch.float32)
    m = JEPA(blk.obs(idx[:1]).shape[1])
    opt = torch.optim.AdamW([p for n, p in m.named_parameters() if not n.startswith("tgt")], lr=LR, weight_decay=1e-5)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(idx)).numpy()
        for k in range(0, len(idx) - BATCH + 1, BATCH):
            i = idx[perm[k:k + BATCH]]
            z0 = m.enc(T(nobs(blk.obs(i))))
            z, pred = z0, 0.0
            for j in range(1, K + 1):
                z = m.step(z, T(nact(np.c_[blk.A[i + j - 1], blk.surf[i]])))   # on enchaine
                with torch.no_grad():
                    zt = m.tgt(T(nobs(blk.obs(i + j))))
                pred = pred + F.mse_loss(z, zt) / K
            var, cov = vicreg(z0)
            loss = pred + W_VAR * var + W_COV * cov
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():                      # EMA : la cible suit lentement l'encodeur
                for pt, p in zip(m.tgt.parameters(), m.enc.parameters()):
                    pt.mul_(EMA).add_(p, alpha=1 - EMA)
        if ep in (0, EPOCHS // 2, EPOCHS - 1):
            with torch.no_grad():
                zz = m.enc(T(nobs(blk.obs(idx[:5000]))))
            print(f"    jepa K={K} ep {ep + 1:2d} : pred {float(pred):.4f}  var {var.item():.3f}  cov {cov.item():.3f}  "
                  f"rang effectif {eff_rank(zz):.1f}/{LATENT}", flush=True)
    return m


def train_decoder(m, blk, idx, nobs, nst, seed):
    """Decodeur latent -> etat courant, encodeur GELE.
    Entraine sur les latents de l'encodeur CIBLE : c'est dans cet espace que le
    predicteur apprend a predire, donc c'est cet espace qu'on decode au rollout."""
    torch.manual_seed(seed)
    with torch.no_grad():
        z = m.tgt(torch.tensor(nobs(blk.obs(idx)), dtype=torch.float32))
    y = torch.tensor(nst(blk.S[idx]), dtype=torch.float32)
    dec = mlp(LATENT, HID, 3)
    opt = torch.optim.AdamW(dec.parameters(), lr=LR)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(z))
        for k in range(0, len(z) - BATCH + 1, BATCH):
            b = perm[k:k + BATCH]
            loss = F.mse_loss(dec(z[b]), y[b])
            opt.zero_grad(); loss.backward(); opt.step()
    return dec


# ---------------------------------------------------------------- MLP avec historique

def train_mlp_hist(blk, idx, nobs, nact, ndelta, seed):
    torch.manual_seed(seed)
    x = torch.tensor(np.c_[nobs(blk.obs(idx)), nact(np.c_[blk.A[idx], blk.surf[idx]])], dtype=torch.float32)
    y = torch.tensor(ndelta(blk.S[idx + 1] - blk.S[idx]), dtype=torch.float32)
    net = mlp(x.shape[1], HID, 3)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-5)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(x))
        for k in range(0, len(x) - BATCH + 1, BATCH):
            b = perm[k:k + BATCH]
            loss = F.mse_loss(net(x[b]), y[b])
            opt.zero_grad(); loss.backward(); opt.step()
    return net


# ---------------------------------------------------------------- controle : modeles directs entraines sur rollout

class TNorm:
    """Normalisation en tenseurs torch, avec les memes constantes que Norm."""
    def __init__(self, n):
        self.m, self.s = torch.tensor(n.m, dtype=torch.float32), torch.tensor(n.s, dtype=torch.float32)
    def __call__(self, x):
        return (x - self.m) / self.s
    def inv(self, x):
        return x * self.s + self.m


def direct_step(net, hist, st, ac, s, a, surf, tz):
    """Un pas d'un modele direct. hist=True : entree = fenetre (comme le JEPA) ;
    hist=False : entree = etat courant seul. Sortie : etat suivant."""
    sa = tz["nact"](torch.cat([a, surf[:, None]], 1))
    if hist:
        o = torch.cat([st.flatten(1), ac.flatten(1), surf[:, None]], 1)
        x = torch.cat([tz["nobs"](o), sa], 1)
    else:
        x = torch.cat([tz["nst"](s), sa], 1)
    return s + tz["ndelta"].inv(net(x))


def train_direct(blk, nz, seed, K, hist):
    """Meme principe que le JEPA K pas, mais DANS L'ESPACE D'ETAT : le modele
    enchaine ses propres predictions sur K pas et chaque pas est compare au vrai etat."""
    torch.manual_seed(seed)
    tz = {k: TNorm(v) for k, v in nz.items()}
    idx = rollout_idx(blk, K)
    d_in = (6 * H + 1 if hist else 3) + 4
    net = mlp(d_in, HID, 3)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-5)
    T = lambda x: torch.tensor(x, dtype=torch.float32)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(idx)).numpy()
        for k in range(0, len(idx) - BATCH + 1, BATCH):
            i = idx[perm[k:k + BATCH]]
            st = T(blk.S[i[:, None] + np.arange(-H + 1, 1)])
            ac = T(blk.A[i[:, None] + np.arange(-H, 0)])
            surf = T(blk.surf[i]); s = st[:, -1]; loss = 0.0
            for j in range(K):
                a = T(blk.A[i + j])
                s = direct_step(net, hist, st, ac, s, a, surf, tz)
                loss = loss + F.mse_loss(tz["nst"](s), tz["nst"](T(blk.S[i + j + 1]))) / K
                st = torch.cat([st[:, 1:], s[:, None]], 1)
                ac = torch.cat([ac[:, 1:], a[:, None]], 1)
            opt.zero_grad(); loss.backward(); opt.step()
    return net


@torch.no_grad()
def rollout_direct(net, hist, blk, starts, nz):
    tz = {k: TNorm(v) for k, v in nz.items()}
    T = lambda x: torch.tensor(x, dtype=torch.float32)
    st = T(blk.S[starts[:, None] + np.arange(-H + 1, 1)])
    ac = T(blk.A[starts[:, None] + np.arange(-H, 0)])
    surf = T(blk.surf[starts]); s = st[:, -1]; out = []
    for h in range(HORIZON):
        a = T(blk.A[starts + h])
        s = direct_step(net, hist, st, ac, s, a, surf, tz)
        s[:, 0] = torch.clamp(s[:, 0], min=0.0)
        st = torch.cat([st[:, 1:], s[:, None]], 1)
        ac = torch.cat([ac[:, 1:], a[:, None]], 1)
        out.append(s.numpy().copy())
    return np.stack(out, 1)


# ---------------------------------------------------------------- rollouts

@torch.no_grad()
def rollout_jepa(m, dec, blk, starts, nobs, nact, nst):
    """Renvoie deux lectures du meme rollout latent :
      brut   : on decode directement chaque latent predit ;
      ancre  : s_t + D(z_t+k) - D(z_t). On ne lit que le CHANGEMENT predit, ce qui
               annule l'erreur fixe du decodeur. s_t est connu au depart, comme pour
               toutes les autres methodes : aucune information du futur."""
    z = m.tgt(torch.tensor(nobs(blk.obs(starts)), dtype=torch.float32))   # meme espace que le decodeur
    base = nst.inv(dec(z).numpy())
    s0 = blk.S[starts]
    raw, anchored = [], []
    for h in range(HORIZON):
        i = starts + h
        a = torch.tensor(nact(np.c_[blk.A[i], blk.surf[i]]), dtype=torch.float32)
        z = m.step(z, a)                                   # on avance DANS le latent
        d = nst.inv(dec(z).numpy())                        # et on decode pour comparer
        raw.append(d)
        anchored.append(s0 + d - base)
    return np.stack(raw, 1), np.stack(anchored, 1)


@torch.no_grad()
def rollout_mlp_hist(net, blk, starts, nobs, nact, ndelta):
    N = len(starts)
    st = blk.S[starts[:, None] + np.arange(-H + 1, 1)].copy()     # (N, H, 3)
    ac = blk.A[starts[:, None] + np.arange(-H, 0)].copy()
    surf = blk.surf[starts]
    out = []
    for h in range(HORIZON):
        i = starts + h
        o = np.concatenate([st.reshape(N, -1), ac.reshape(N, -1), surf[:, None]], 1)
        x = np.c_[nobs(o), nact(np.c_[blk.A[i], surf])]
        s_new = st[:, -1] + ndelta.inv(net(torch.tensor(x, dtype=torch.float32)).numpy())
        s_new[:, 0] = np.maximum(s_new[:, 0], 0.0)
        st = np.concatenate([st[:, 1:], s_new[:, None]], 1)       # la fenetre glisse
        ac = np.concatenate([ac[:, 1:], blk.A[i][:, None]], 1)
        out.append(s_new)
    return np.stack(out, 1)


def truth_of(blk, starts):
    return blk.S[starts[:, None] + np.arange(1, HORIZON + 1)]


# ---------------------------------------------------------------- une graine

JEPA_KS = [1]          # longueurs de rollout latent a l'entrainement ; fixe par select_K()


def fit_all(df, train, seed, Ks, direct=None):
    """direct : {"mlp_k": K, "mlp_hist_k": K} pour les modeles de controle (ou None)."""
    """Entraine tous les modeles sur les groupes `train`."""
    tr = Block(df, train)
    idx = tr.transition_idx()
    nz = dict(nobs=Norm(tr.obs(idx)), nact=Norm(np.c_[tr.A[idx], tr.surf[idx]]),
              nst=Norm(tr.S[idx]), ndelta=Norm(tr.S[idx + 1] - tr.S[idx]))
    t0 = time.time()
    jepas = {}
    for K in Ks:
        m = train_jepa(tr, nz["nobs"], nz["nact"], seed, K=K)
        jepas[K] = (m, train_decoder(m, tr, idx, nz["nobs"], nz["nst"], seed))
    mh = train_mlp_hist(tr, idx, nz["nobs"], nz["nact"], nz["ndelta"], seed)
    s, a, sn, sf = transitions_with_surface(df, train)     # baselines sans historique
    models = dict(lin=Learned("lineaire", seed).fit(s, a, sn, sf), mlp=Learned("mlp", seed).fit(s, a, sn, sf),
                  bike=B.fit_per_surface(s, a, sn, sf), mh=mh, jepas=jepas, direct={})
    for name, K in (direct or {}).items():
        hist = name.startswith("mlp_hist")
        models["direct"][name] = (train_direct(tr, nz, seed, K, hist), hist)
    print(f"  [graine {seed}] {len(idx)} transitions, modeles entraines en {time.time() - t0:.0f} s", flush=True)
    return tr, idx, nz, models


def predict_all(df, groups, nz, M):
    blk = Block(df, groups)
    starts = blk.segment_idx()
    truth = truth_of(blk, starts)
    s0, acts, f = blk.S[starts], blk.A[starts[:, None] + np.arange(HORIZON)], blk.surf[starts]

    def bike_step(x, u):
        out = np.empty_like(x)
        for k, p in M["bike"].items():
            out[f == k] = B.step(p, x[f == k], u[f == k])
        return out

    preds = {
        "persistance": np.repeat(s0[:, None], HORIZON, 1),
        "lineaire": rollout_state(lambda x, u: M["lin"].step(x, u, f), s0, acts),
        "mlp": rollout_state(lambda x, u: M["mlp"].step(x, u, f), s0, acts),
        "velo": rollout_state(bike_step, s0, acts),
        "mlp_hist": rollout_mlp_hist(M["mh"], blk, starts, nz["nobs"], nz["nact"], nz["ndelta"]),
    }
    for K, (m, dec) in M["jepas"].items():
        preds[jepa_name(K)] = rollout_jepa(m, dec, blk, starts, nz["nobs"], nz["nact"], nz["nst"])[1]   # lecture ancree
    for name, (net, hist) in M["direct"].items():
        preds[name] = rollout_direct(net, hist, blk, starts, nz)
    rmse = {k: np.sqrt(((p - truth) ** 2).mean(0)) for k, p in preds.items()}
    return rmse, len(starts), blk


def jepa_name(K):
    return "jepa" if K == 1 else f"jepa_k{K}"


def score(rmse, name):
    """Un seul nombre pour choisir K : erreur a 1 s rapportee a la persistance, moyennee sur vx, vy, r."""
    return float(np.mean(rmse[name][-1] / rmse["persistance"][-1]))


def select_K(df, candidates, seed=0):
    """Choix de K SUR UN JEU DE VALIDATION decoupe dans les groupes d'entrainement.
    Les jeux de test (en distribution et glisse) ne sont pas touches."""
    train, _, _ = R.split_groups(df, seed=seed)
    g = np.array(sorted(train)); rng = np.random.default_rng(100 + seed); rng.shuffle(g)
    n_val = int(round(0.2 * len(g)))
    val, inner = set(g[:n_val]), set(g[n_val:])
    print(f"[validation] {len(inner)} groupes d'entrainement interne, {len(val)} groupes de validation", flush=True)
    _, _, nz, M = fit_all(df, inner, seed, candidates)
    rmse, n, _ = predict_all(df, val, nz, M)
    scores = {K: score(rmse, jepa_name(K)) for K in candidates}
    for K in candidates:
        print(f"[validation] K = {K:2d} : erreur a 1 s / persistance = {scores[K]:.3f}  "
              f"(vx {rmse[jepa_name(K)][-1][0]:.2f}, vy {rmse[jepa_name(K)][-1][1]:.3f}, r {rmse[jepa_name(K)][-1][2]:.3f})", flush=True)
    best = min(scores, key=scores.get)
    print(f"[validation] K retenu : {best}", flush=True)
    return best


DIRECT = {}           # {"mlp_k8": 8, ...} rempli par select_direct()


def select_direct(df, candidates, seed=0):
    """Controle : meme protocole de choix de K que le JEPA, meme validation."""
    train, _, _ = R.split_groups(df, seed=seed)
    g = np.array(sorted(train)); rng = np.random.default_rng(100 + seed); rng.shuffle(g)
    n_val = int(round(0.2 * len(g)))
    val, inner = set(g[:n_val]), set(g[n_val:])
    chosen = {}
    for base in ["mlp", "mlp_hist"]:
        cand = {f"{base}_c{K}": K for K in candidates}
        _, _, nz, M = fit_all(df, inner, seed, [], direct=cand)
        rmse, _, _ = predict_all(df, val, nz, M)
        sc = {K: score(rmse, f"{base}_c{K}") for K in candidates}
        for K in candidates:
            print(f"[validation] {base} K = {K:2d} : erreur a 1 s / persistance = {sc[K]:.3f}", flush=True)
        best = min(sc, key=sc.get)
        chosen[f"{base}_k{best}"] = best
        print(f"[validation] {base} : K retenu = {best}", flush=True)
    return chosen


def run_seed(df, seed):
    train, test_in, test_out = R.split_groups(df, seed=seed)
    tr, idx, nz, M = fit_all(df, train, seed, JEPA_KS, DIRECT)
    res, probes = {}, {}
    for name_test, groups in [("en distribution", test_in), ("glisse (hors distribution)", test_out)]:
        rmse, n, blk = predict_all(df, groups, nz, M)
        for k, v in rmse.items():
            res[(name_test, k)] = v
        res[(name_test, "_n")] = n
        pidx = np.where(blk.streak >= H + 1)[0]                 # probe : derive de l'instant courant
        probes[name_test] = (pidx, np.arctan2(blk.S[pidx, 1], blk.S[pidx, 0]), blk)
    tr_beta = np.arctan2(tr.S[idx, 1], tr.S[idx, 0])
    T = lambda x: torch.tensor(x, dtype=torch.float32)
    feats = {"etat brut": (tr.S[idx], lambda b, i: b.S[i]),
             "fenetre brute": (nz["nobs"](tr.obs(idx)), lambda b, i: nz["nobs"](b.obs(i)))}
    for K, (m, _) in M["jepas"].items():
        enc = lambda b, i, m=m: m.enc(T(nz["nobs"](b.obs(i)))).detach().numpy()
        feats[f"latent {jepa_name(K)}"] = (enc(tr, idx), enc)
    rp = {}
    for fname, (Xtr, fn) in feats.items():
        reg = Ridge(alpha=1.0).fit(Xtr, tr_beta)
        for name_test, (pidx, beta, blk) in probes.items():
            rp[(fname, name_test)] = r2_score(beta, reg.predict(fn(blk, pidx)))
    return res, rp


def main():
    global JEPA_KS, DIRECT
    df = add_surface(R.load_all())
    if "--select" in sys.argv:                       # etape 1 : choisir K sur la validation
        best = select_K(df, [1, 4, 8, 16])
        JEPA_KS = sorted({1, best})
    elif "--k" in sys.argv:                          # ou imposer K : --k 8
        JEPA_KS = sorted({1, int(sys.argv[sys.argv.index("--k") + 1])})
    if "--control" in sys.argv:                      # controle : MLP entraines sur rollout, K choisi pareil
        DIRECT = select_direct(df, [1, 4, 8, 16])
    all_res, all_probe = [], []
    for seed in SEEDS:
        r, p = run_seed(df, seed)
        all_res.append(r); all_probe.append(p)

    names = ["persistance", "lineaire", "mlp", "velo", "mlp_hist"] + list(DIRECT) + [jepa_name(K) for K in JEPA_KS]
    rows = []
    for test in ["en distribution", "glisse (hors distribution)"]:
        n = int(np.mean([r[(test, "_n")] for r in all_res]))
        print(f"\n=== Test {test} ({n} departs de rollout) - RMSE moyenne sur {len(SEEDS)} graines (ecart-type)")
        for comp, unit, j in [("vy", "m/s", 1), ("r", "rad/s", 2), ("vx", "m/s", 0)]:
            print(f"  {comp} ({unit})")
            for name in names:
                v = np.array([r[(test, name)][:, j] for r in all_res])
                print(f"    {name:12s} " + "   ".join(f"{h * C.DT:.1f}s: {v[:, h - 1].mean():.3f} ({v[:, h - 1].std():.3f})" for h in REPORT))
                for h in range(HORIZON):
                    rows.append([test, comp, name, (h + 1) * C.DT, v[:, h].mean(), v[:, h].std()])
    print("\n=== Probe lineaire : R2 de l'angle de derive (moyenne sur les graines)")
    for fname in ["etat brut", "fenetre brute"] + [f"latent {jepa_name(K)}" for K in JEPA_KS]:
        print(f"  {fname:16s} " + "   ".join(
            f"{t}: {np.mean([p[(fname, t)] for p in all_probe]):.3f}" for t in ["en distribution", "glisse (hors distribution)"]))
    out = pd.DataFrame(rows, columns=["test", "composante", "modele", "horizon_s", "rmse", "ecart_type"])
    out.to_csv(os.path.join(C.OUT_DIR, "eval_jepa.csv"), index=False)
    print(f"\n[ok] -> {os.path.join(C.OUT_DIR, 'eval_jepa.csv')}")


if __name__ == "__main__":
    main()
