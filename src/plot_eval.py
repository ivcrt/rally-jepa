# src/plot_eval.py
# Figure : erreur de prediction en fonction de l'horizon, par modele.
# Lit outputs/eval_state.csv (produit par src/eval_state.py).
# Lancer : python src/plot_eval.py

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

STYLE = {   # persistance = reference grise en pointilles ; les 3 modeles en couleur
    "persistance": dict(color="#8a8984", ls=(0, (4, 3)), label="Persistance"),
    "lineaire": dict(color="#2a78d6", ls="-", label="Linéaire"),
    "mlp": dict(color="#eb6834", ls="-", label="MLP direct"),
    "velo": dict(color="#1baf7a", ls="-", label="Modèle vélo"),
}
# Figure JEPA : 4 modeles en couleur (ordre fixe de la palette) + persistance en gris.
STYLE_JEPA = {
    "persistance": dict(color="#8a8984", ls=(0, (4, 3)), label="Persistance"),
    "jepa": dict(color="#2a78d6", ls=(0, (2, 2)), label="JEPA 1 pas"),
    "jepa_k8": dict(color="#2a78d6", ls="-", label="JEPA 8 pas"),
    "mlp_hist_k16": dict(color="#eb6834", ls="-", label="Contrôle (MLP + hist., 16 pas)"),
    "velo": dict(color="#1baf7a", ls="-", label="Modèle vélo"),
    "mlp": dict(color="#eda100", ls="-", label="MLP direct"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TESTS = [("en distribution", "Test en distribution\n(asphalte + terre)"),
         ("glisse (hors distribution)", "Test hors distribution\n(glisse provoquée)")]
COMPS = [("vy", "Vitesse latérale vy\nRMSE (m/s)"), ("r", "Vitesse de lacet r\nRMSE (rad/s)")]
COMPS_JEPA = COMPS + [("vx", "Vitesse longitudinale vx\nRMSE (m/s)")]


def main(csv="eval_state.csv", style=STYLE, png="eval_state.png", comps=COMPS):
    STYLE = style
    df = pd.read_csv(os.path.join(C.OUT_DIR, csv))
    fig, axes = plt.subplots(len(comps), 2, figsize=(11, 3.6 * len(comps) + 0.3), sharex=True, facecolor="#fcfcfb")
    for i, (comp, ylab) in enumerate(comps):
        ymax = df[(df.composante == comp) & df.modele.isin(list(STYLE))].rmse.max() * 1.08
        for j, (test, title) in enumerate(TESTS):
            ax = axes[i, j]
            ax.set_facecolor("#fcfcfb")
            d = df[(df.test == test) & (df.composante == comp)]
            ends = []
            for name, st in STYLE.items():
                m = d[d.modele == name].sort_values("horizon_s")
                ax.fill_between(m.horizon_s, m.rmse - m.ecart_type, m.rmse + m.ecart_type,
                                color=st["color"], alpha=0.12, lw=0)
                ax.plot(m.horizon_s, m.rmse, color=st["color"], ls=st["ls"], lw=2)
                ends.append([m.rmse.iloc[-1], m.rmse.iloc[-1], st["label"], st["color"]])
            # etiquettes directes en bout de courbe, ecartees si elles se chevauchent
            # on ecarte les etiquettes symetriquement autour de leur point (pas seulement vers le haut)
            ends.sort()
            gap = ymax * 0.055
            for _ in range(200):
                moved = False
                for k in range(1, len(ends)):
                    d = ends[k][0] - ends[k - 1][0]
                    if d < gap:
                        push = (gap - d) / 2
                        ends[k][0] += push; ends[k - 1][0] -= push; moved = True
                if not moved:
                    break
            for y_lab, y_end, lab, col in ends:
                ax.plot([1.0], [y_end], marker="o", ms=4, color=col, clip_on=False)
                ax.text(1.03, y_lab, lab, color=INK, fontsize=9, va="center")
            ax.set_ylim(0, ymax)
            ax.set_xlim(0, 1.0)
            ax.grid(True, color=GRID, lw=0.8)
            for s in ["top", "right"]:
                ax.spines[s].set_visible(False)
            for s in ["left", "bottom"]:
                ax.spines[s].set_color(MUTED)
            ax.tick_params(colors=MUTED, labelsize=9)
            if i == 0:
                ax.set_title(title, color=INK, fontsize=11, loc="left")
            if j == 0:
                ax.set_ylabel(ylab, color=INK, fontsize=10)
            if i == len(comps) - 1:
                ax.set_xlabel("Horizon de prédiction (s)", color=INK, fontsize=10)
    handles = [plt.Line2D([], [], color=s["color"], ls=s["ls"], lw=2) for s in STYLE.values()]
    fig.legend(handles, [s["label"] for s in STYLE.values()], loc="lower center", ncol=min(len(STYLE), 3),
               frameon=False, fontsize=10, labelcolor=INK)
    fig.suptitle("Erreur de prédiction multi-pas sur les données BeamNG (moyenne ± écart-type, 3 découpages)",
                 color=INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.06, 0.9, 0.96))
    out = os.path.join(C.OUT_DIR, png)
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print(f"[plot] {out}")


if __name__ == "__main__":
    if "--jepa" in sys.argv:          # python src/plot_eval.py --jepa
        main("eval_jepa.csv", STYLE_JEPA, "eval_jepa.png", COMPS_JEPA)
    else:
        main()
