# config.py
# Hyperparametres et chemins, centralises ici pour tout le projet.
# Style procedural : un simple module de variables, pas de classe.

import os

# --- Chemins ---
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "outputs")
RAW_CSV = os.path.join(DATA_DIR, "raw_runs.csv")      # sortie de collect_beamng.py
DATASET_NPZ = os.path.join(DATA_DIR, "dataset.npz")   # sortie de make_dataset.py

# --- Representation de l'etat et de l'action ---
# Etat (repere voiture) : vx (vitesse longi), vy (vitesse laterale), r (vitesse de lacet).
STATE_COLS = ["vx", "vy", "r"]
# Action : delta (angle volant rad), ax_cmd (acceleration longi commandee).
ACTION_COLS = ["delta", "ax_cmd"]
# Cible physique pour le probe lineaire : beta = angle de derive = atan2(vy, vx).
PROBE_TARGET = "beta"

STATE_DIM = len(STATE_COLS)
ACTION_DIM = len(ACTION_COLS)

# --- Fenetrage ---
CONTEXT = 1          # nb de pas de contexte en entree (1 = modele markovien simple)
DT = 0.02            # pas de temps (s), doit matcher la collecte
MIN_SPEED = 2.0      # m/s : en dessous, la derive n'a pas de sens, fenetre ignoree

# --- Modele JEPA ---
LATENT_DIM = 32
ENC_HIDDEN = 128
PRED_HIDDEN = 128
ACTION_LATENT = 16

# Anti-collapse : EMA de l'encodeur cible + VICReg (ceinture-bretelles).
EMA_DECAY = 0.99
W_PRED = 1.0         # poids de la perte de prediction (dans le latent)
W_VAR = 1.0          # poids du terme variance VICReg
W_COV = 0.04         # poids du terme covariance VICReg

# --- Entrainement ---
EPOCHS = 30
BATCH = 256
LR = 1e-3
WEIGHT_DECAY = 1e-5
SEED = 0             # pour multi-seed, lancer avec plusieurs valeurs et moyenner

# --- Split groupe (anti-fuite) ---
# On split PAR RUN, jamais par pas de temps. Lecon IEM / CL-Drive.
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# --- Synthetique (pour faire tourner le repo sans BeamNG) ---
SYN_N_RUNS = 60
SYN_RUN_LEN = 600    # pas de temps par run
