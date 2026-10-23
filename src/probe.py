# src/probe.py
# Probe lineaire : le latent appris encode-t-il une grandeur physique ?
# On gele l'encodeur, on entraine une simple regression lineaire latent -> beta
# (angle de derive). Bonne R2 = le world model a capture la physique sans qu'on
# le lui ait dit. C'est l'evaluation standard du self-supervised (linear probing).

import os
import sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(HERE))
sys.path.append(HERE)

import config as C
from dataset import make_loaders
from model import RallyJEPA


def extract_latents(model, loader):
    Z, B = [], []
    with torch.no_grad():
        for x_state, _action, _y, beta in loader:
            Z.append(model.encoder(x_state).cpu().numpy())
            B.append(beta.cpu().numpy())
    return np.concatenate(Z), np.concatenate(B)


def main(ckpt):
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    model = RallyJEPA()
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
    model.eval()

    loaders, _, _ = make_loaders()
    Ztr, Btr = extract_latents(model, loaders["train"])
    Zte, Bte = extract_latents(model, loaders["test"])

    reg = Ridge(alpha=1.0).fit(Ztr, Btr)
    pred = reg.predict(Zte)
    r2 = r2_score(Bte, pred)
    print(f"[probe] R2 (angle de derive beta, test) = {r2:.3f}")
    print("        R2 eleve => le latent encode bien la physique de derive.")
    return r2


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else os.path.join(C.OUT_DIR, f"jepa_seed{C.SEED}.pt")
    main(ckpt)
