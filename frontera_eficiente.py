"""
Grafica la frontera eficiente de Markowitz para el universo de acciones,
marcando el punto calibrado para cada perfil de riesgo.

Uso:
    python frontera_eficiente.py
"""
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from sklearn.covariance import LedoitWolf
import pandas as pd

from data.loader import load_universe
from config import RISK_PROFILES, MAX_WEIGHT
from backtesting.runner import _train_window, _CLIP, TRAIN_WINDOW_YEARS
from models.markowitzprelim import calibrar_lambda

GUARDAR = r"C:\Users\aguor\OneDrive - Universidad Católica de Chile\G18 Portafolios\E3final\imagenes\frontera_eficiente.png"

COLORES = {
    "conservador":    "#4CAF50",
    "neutro":         "#FF9800",
    "arriesgado":     "#F44336",
    "muy_arriesgado": "#9C27B0",
}

LABELS = {
    "conservador":    "Conservador (5%)",
    "neutro":         "Neutro (15%)",
    "arriesgado":     "Arriesgado (30%)",
    "muy_arriesgado": "Muy arriesgado (40%)",
}

_VOL_FACTOR = 0.5


def _punto_frontera(mu, Sigma, lam, max_weight, n):
    """Resuelve Markowitz sin QCQP y retorna (vol_anual, ret_anual)."""
    model = gp.Model("frontera")
    model.Params.OutputFlag = 0
    w = model.addVars(n, lb=0, ub=max_weight, name="w")
    model.addConstr(gp.quicksum(w[i] for i in range(n)) == 1)
    port_var = gp.quicksum(
        w[i] * Sigma[i, j] * w[j] for i in range(n) for j in range(n)
    )
    port_ret = gp.quicksum(mu[i] * w[i] for i in range(n))
    model.setObjective(port_ret - lam * port_var, GRB.MAXIMIZE)
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        return None, None
    w_vals = np.array([w[i].X for i in range(n)])
    vol = float(np.sqrt(252 * w_vals @ Sigma @ w_vals))
    ret = float(252 * float(np.dot(mu, w_vals)))
    return vol, ret


def main():
    print("Cargando datos...")
    price_returns, _ = load_universe()

    eval_start = "2019-01-01"
    meses = pd.date_range(eval_start, periods=1, freq="MS")
    train = _train_window(price_returns, meses[0], TRAIN_WINDOW_YEARS)
    train = train.dropna(axis=1, how="any")
    train_r = train.clip(lower=-_CLIP, upper=_CLIP)

    mu    = train_r.mean().values
    lw    = LedoitWolf()
    lw.fit(train_r.values)
    Sigma = lw.covariance_
    n     = len(train_r.columns)

    print(f"  Acciones en ventana de entrenamiento: {n}")

    # Frontera eficiente: barre lambdas en escala log
    lambdas = np.logspace(-2, 4, 80)
    vols, rets = [], []

    print("  Calculando puntos de la frontera eficiente...")
    for lam in lambdas:
        vol, ret = _punto_frontera(mu, Sigma, lam, MAX_WEIGHT, n)
        if vol is not None:
            vols.append(vol * 100)
            rets.append(ret * 100)

    # Calibrar lambda y calcular punto exacto por perfil
    print("  Calibrando lambda por perfil...")
    puntos_perfil = {}
    for perfil, tolerancia in RISK_PROFILES.items():
        if tolerancia == 0.0:
            continue
        target_vol = tolerancia * _VOL_FACTOR
        lam_cal = calibrar_lambda(train_r, target_vol, max_weight=MAX_WEIGHT)
        vol, ret = _punto_frontera(mu, Sigma, lam_cal, MAX_WEIGHT, n)
        puntos_perfil[perfil] = {
            "vol":       vol * 100,
            "ret":       ret * 100,
            "lambda":    lam_cal,
            "target_vol": target_vol * 100,
        }
        print(f"    {perfil:<20} lambda={lam_cal:.3f}  vol={vol*100:.1f}%  ret={ret*100:.1f}%")

    # Grafico
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(vols, rets, color="#1565C0", linewidth=2.5, label="Frontera eficiente", zorder=2)

    for perfil, p in puntos_perfil.items():
        ax.scatter(p["vol"], p["ret"],
                   color=COLORES[perfil], s=120, zorder=5,
                   label=f"{LABELS[perfil]}  (λ={p['lambda']:.2f})")
        ax.annotate(
            f"  {LABELS[perfil]}\n  λ={p['lambda']:.2f}",
            xy=(p["vol"], p["ret"]),
            fontsize=8.5,
            color=COLORES[perfil],
            va="center",
        )
        # linea vertical en el vol objetivo
        ax.axvline(x=p["target_vol"], color=COLORES[perfil],
                   linestyle=":", linewidth=1, alpha=0.4)

    ax.set_xlabel("Volatilidad anual (%)", fontsize=12)
    ax.set_ylabel("Retorno anual esperado (%)", fontsize=12)
    ax.set_title("Frontera eficiente de Markowitz\nUniverso 300 acciones — ventana 2015-2018", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    fig.tight_layout()
    fig.savefig(GUARDAR, dpi=150, bbox_inches="tight")
    print(f"\nImagen guardada en:\n  {GUARDAR}")
    plt.show()


if __name__ == "__main__":
    main()
