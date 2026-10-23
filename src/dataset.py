# src/dataset.py
# Chargement du dataset fenetre + split PAR GROUPE (run) avec assert anti-fuite.
# Regle critique : jamais de split par pas de temps, sinon train et test se
# chevauchent et les scores sont trompeurs (lecon IEM / CL-Drive).

import os
import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C


def load_arrays():
    d = np.load(C.DATASET_NPZ)
    return (d["X_state"], d["X_action"], d["Y_next"], d["groups"], d["beta"])


def grouped_split(groups, seed):
    """Repartit les RUNS (pas les pas de temps) en train/val/test."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    n = len(uniq)
    n_test = max(1, int(round(C.TEST_FRAC * n)))
    n_val = max(1, int(round(C.VAL_FRAC * n)))
    test_g = set(uniq[:n_test])
    val_g = set(uniq[n_test:n_test + n_val])
    train_g = set(uniq[n_test + n_val:])

    # Assert anti-fuite : aucun run partage entre les splits.
    assert train_g.isdisjoint(val_g) and train_g.isdisjoint(test_g) and val_g.isdisjoint(test_g)

    train = np.isin(groups, list(train_g))
    val = np.isin(groups, list(val_g))
    test = np.isin(groups, list(test_g))
    return train, val, test


def compute_norm(x):
    """Stats de normalisation par canal, calculees sur le TRAIN uniquement."""
    mu = x.mean(axis=tuple(range(x.ndim - 1)), keepdims=True)
    sd = x.std(axis=tuple(range(x.ndim - 1)), keepdims=True) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def make_loaders(seed=C.SEED, batch=C.BATCH):
    Xs, Xa, Y, groups, beta = load_arrays()
    train, val, test = grouped_split(groups, seed)

    # Normalisation calee sur le train (pas de fuite stats test).
    mu_s, sd_s = compute_norm(Xs[train])
    mu_a, sd_a = compute_norm(Xa[train])
    mu_y, sd_y = compute_norm(Y[train])

    def norm(x, mu, sd):
        return ((x - mu) / sd).astype(np.float32)

    def build(mask, shuffle):
        ds = TensorDataset(
            torch.from_numpy(norm(Xs[mask], mu_s, sd_s)),
            torch.from_numpy(norm(Xa[mask], mu_a, sd_a)),
            torch.from_numpy(norm(Y[mask], mu_y, sd_y)),
            torch.from_numpy(beta[mask].astype(np.float32)),
        )
        return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=shuffle)

    loaders = dict(
        train=build(train, True),
        val=build(val, False),
        test=build(test, False),
    )
    norm_stats = dict(mu_s=mu_s, sd_s=sd_s, mu_a=mu_a, sd_a=sd_a, mu_y=mu_y, sd_y=sd_y)
    info = dict(n_train_runs=len(np.unique(groups[train])),
                n_val_runs=len(np.unique(groups[val])),
                n_test_runs=len(np.unique(groups[test])))
    return loaders, norm_stats, info
