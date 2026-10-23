# src/model.py
# World model JEPA pour la dynamique vehicule.
# Structure standard d'un JEPA (cf. V-JEPA / litterature world models) :
#   - encodeur d'etat   E      : etat -> latent z
#   - encodeur d'action        : action -> latent action
#   - predicteur P             : (z_t, a_t) -> z_{t+1} predit
#   - encodeur cible E_target  : copie EMA de E, stop-gradient -> z_{t+1} cible
# Perte = prediction dans le latent (W_PRED) + VICReg (variance + covariance)
# pour empecher le collapse des representations.
#
# Les definitions de couches utilisent nn.Module (idiomatique PyTorch). Tout le
# reste du repo (data, train, eval) est procedural.

import os
import sys
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C


def mlp(d_in, d_hidden, d_out):
    return nn.Sequential(
        nn.Linear(d_in, d_hidden), nn.GELU(),
        nn.Linear(d_hidden, d_hidden), nn.GELU(),
        nn.Linear(d_hidden, d_out),
    )


class StateEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Aplati le contexte (CONTEXT pas) en un vecteur.
        self.net = mlp(C.CONTEXT * C.STATE_DIM, C.ENC_HIDDEN, C.LATENT_DIM)

    def forward(self, x_state):           # x_state: (B, CONTEXT, STATE_DIM)
        b = x_state.shape[0]
        return self.net(x_state.reshape(b, -1))


class ActionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp(C.ACTION_DIM, C.ACTION_LATENT, C.ACTION_LATENT)

    def forward(self, a):                 # a: (B, ACTION_DIM)
        return self.net(a)


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp(C.LATENT_DIM + C.ACTION_LATENT, C.PRED_HIDDEN, C.LATENT_DIM)

    def forward(self, z, za):             # z:(B,LATENT) za:(B,ACTION_LATENT)
        return self.net(torch.cat([z, za], dim=-1))


class RallyJEPA(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = StateEncoder()
        self.action_encoder = ActionEncoder()
        self.predictor = Predictor()
        # Encodeur cible : copie figee, mise a jour par EMA, jamais par gradient.
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def forward(self, x_state, action, y_next_state):
        z = self.encoder(x_state)                      # latent online
        za = self.action_encoder(action)
        z_pred = self.predictor(z, za)                 # prediction du latent suivant

        # Cible : encodeur EMA sur l'etat reel suivant, sans gradient.
        with torch.no_grad():
            y_ctx = y_next_state.unsqueeze(1)          # (B,1,STATE_DIM) -> contexte 1
            if C.CONTEXT > 1:
                # contexte cible = derniers CONTEXT etats reels (approx : repete)
                y_ctx = y_next_state.unsqueeze(1).repeat(1, C.CONTEXT, 1)
            z_tgt = self.target_encoder(y_ctx)
        return z_pred, z_tgt, z


@torch.no_grad()
def ema_update(model, decay=C.EMA_DECAY):
    """Met a jour l'encodeur cible : theta_tgt = decay*theta_tgt + (1-decay)*theta."""
    for pt, p in zip(model.target_encoder.parameters(), model.encoder.parameters()):
        pt.mul_(decay).add_(p.detach(), alpha=1 - decay)


def vicreg_terms(z):
    """Termes variance et covariance de VICReg sur un batch de latents (B, D)."""
    # Variance : on veut std >= 1 par dimension (hinge).
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    var_loss = torch.mean(F.relu(1.0 - std))
    # Covariance : on penalise les correlations hors diagonale.
    zc = z - z.mean(dim=0)
    cov = (zc.T @ zc) / (z.shape[0] - 1)
    d = z.shape[1]
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = (off_diag ** 2).sum() / d
    return var_loss, cov_loss


def jepa_loss(z_pred, z_tgt, z_online):
    pred = F.mse_loss(z_pred, z_tgt)
    var_l, cov_l = vicreg_terms(z_online)
    total = C.W_PRED * pred + C.W_VAR * var_l + C.W_COV * cov_l
    return total, dict(pred=pred.item(), var=var_l.item(), cov=cov_l.item())
