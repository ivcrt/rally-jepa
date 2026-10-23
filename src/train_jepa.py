# src/train_jepa.py
# Entrainement du world model JEPA. Procedural.
# Lancer plusieurs seeds (config.SEED) et moyenner pour un resultat publiable.

import os
import sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(HERE))
sys.path.append(HERE)

import config as C
from dataset import make_loaders
from model import RallyJEPA, ema_update, jepa_loss


def set_seed(seed):
    import numpy as np
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def run_epoch(model, loader, opt=None):
    train = opt is not None
    model.train(train)
    tot, n = 0.0, 0
    logs = {"pred": 0.0, "var": 0.0, "cov": 0.0}
    for x_state, action, y_next, _beta in loader:
        z_pred, z_tgt, z_online = model(x_state, action, y_next)
        loss, parts = jepa_loss(z_pred, z_tgt, z_online)
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
            ema_update(model)
        bs = x_state.shape[0]
        tot += loss.item() * bs; n += bs
        for k in logs:
            logs[k] += parts[k] * bs
    for k in logs:
        logs[k] /= max(n, 1)
    return tot / max(n, 1), logs


def main(seed=C.SEED):
    set_seed(seed)
    loaders, norm_stats, info = make_loaders(seed=seed)
    print(f"[split] runs train/val/test = "
          f"{info['n_train_runs']}/{info['n_val_runs']}/{info['n_test_runs']}")

    model = RallyJEPA()
    opt = torch.optim.AdamW(
        list(model.encoder.parameters())
        + list(model.action_encoder.parameters())
        + list(model.predictor.parameters()),
        lr=C.LR, weight_decay=C.WEIGHT_DECAY,
    )

    best_val = float("inf")
    os.makedirs(C.OUT_DIR, exist_ok=True)
    ckpt = os.path.join(C.OUT_DIR, f"jepa_seed{seed}.pt")

    for ep in range(1, C.EPOCHS + 1):
        tr, _ = run_epoch(model, loaders["train"], opt)
        with torch.no_grad():
            va, vlog = run_epoch(model, loaders["val"])
        # Surveiller var/cov : si la variance des latents s'effondre -> collapse.
        print(f"ep {ep:02d} | train {tr:.4f} | val {va:.4f} "
              f"| pred {vlog['pred']:.4f} var {vlog['var']:.4f} cov {vlog['cov']:.4f}")
        if va < best_val:
            best_val = va
            torch.save({"model": model.state_dict(), "norm": norm_stats}, ckpt)

    print(f"[done] meilleur val {best_val:.4f} -> {ckpt}")
    return ckpt


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else C.SEED
    main(seed)
