# src/baselines.py
# Baselines pour situer le world model JEPA. Sans elles, un chiffre absolu ne
# veut rien dire. Trois references :
#   1. persistance        : "rien ne change", s_{t+1} = s_t. Le plancher a battre.
#   2. predicteur direct  : MLP qui predit s_{t+1} en ESPACE D'ETAT (pas latent).
#   3. modele analytique  : le modele velo (data/make_dataset.py). Oracle sur le
#                           synthetique, approximation sur vraies donnees BeamNG.

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(HERE))
sys.path.append(HERE)

import config as C
from dataset import make_loaders
from model import mlp


def persistence_error(loader, sd_y):
    """Erreur RMSE de la baseline persistance (en unites physiques, denormalisees)."""
    err = []
    for x_state, _action, y_next, _beta in loader:
        s_t = x_state[:, -1, :]               # dernier etat du contexte (normalise)
        pred = s_t                            # rien ne change
        err.append(((pred - y_next) ** 2).mean(dim=1))
    rmse_norm = torch.sqrt(torch.cat(err).mean()).item()
    return rmse_norm * float(np.mean(sd_y))   # remise a l'echelle approx


class DirectPredictor(nn.Module):
    """Predit directement l'etat suivant (pas de latent). Reference 'non-JEPA'."""
    def __init__(self):
        super().__init__()
        self.net = mlp(C.CONTEXT * C.STATE_DIM + C.ACTION_DIM, C.PRED_HIDDEN, C.STATE_DIM)

    def forward(self, x_state, action):
        b = x_state.shape[0]
        return self.net(torch.cat([x_state.reshape(b, -1), action], dim=-1))


def train_direct(seed=C.SEED):
    loaders, norm, _ = make_loaders(seed=seed)
    model = DirectPredictor()
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    for ep in range(1, C.EPOCHS + 1):
        model.train()
        for x_state, action, y_next, _ in loaders["train"]:
            pred = model(x_state, action)
            loss = F.mse_loss(pred, y_next)
            opt.zero_grad(); loss.backward(); opt.step()
    # Eval test
    model.eval()
    errs = []
    with torch.no_grad():
        for x_state, action, y_next, _ in loaders["test"]:
            pred = model(x_state, action)
            errs.append(((pred - y_next) ** 2).mean(dim=1))
    rmse = torch.sqrt(torch.cat(errs).mean()).item() * float(np.mean(norm["sd_y"]))
    return model, rmse


if __name__ == "__main__":
    loaders, norm, _ = make_loaders()
    pe = persistence_error(loaders["test"], norm["sd_y"])
    print(f"[baseline] persistance RMSE (test) = {pe:.4f}")
    _, de = train_direct()
    print(f"[baseline] predicteur direct RMSE (test) = {de:.4f}")
