import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config import RISK_PROFILES
from portfolio.probabilities import p1_abandono, p2_aceptacion


COLORES = {
    "muy_conservador": "#2196F3",
    "conservador":     "#4CAF50",
    "neutro":          "#FF9800",
    "arriesgado":      "#F44336",
    "muy_arriesgado":  "#9C27B0",
}

LABELS = {
    "muy_conservador": "Muy conservador (0%)",
    "conservador":     "Conservador (5%)",
    "neutro":          "Neutro (15%)",
    "arriesgado":      "Arriesgado (30%)",
    "muy_arriesgado":  "Muy arriesgado (40%)",
}


def _formato_eje_fecha(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.tick_params(axis="x", rotation=0)


def plot_valor_portafolios(
    resultados: dict,
    capital_inicial: float = 1_000_000,
    benchmark: "pd.DataFrame | None" = None,
    guardar: str = None,
):
    """Evolucion normalizada del valor del portafolio por perfil (base 100)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Benchmark
    if benchmark is not None:
        bm_norm = benchmark["valor"] / capital_inicial * 100
        ax.plot(benchmark.index, bm_norm, label="Benchmark (30 acc. azar)",
                color="black", linewidth=1.5, linestyle="--", alpha=0.7)

    for perfil, df in resultados.items():
        if df.empty:
            continue
        valor_norm = df["valor"] / capital_inicial * 100
        ax.plot(df.index, valor_norm, label=LABELS[perfil],
                color=COLORES[perfil], linewidth=2)
        if df["abandona"].any():
            mes = df[df["abandona"]].index[-1]
            ax.scatter([mes], [valor_norm.loc[mes]], marker="x",
                       s=120, color=COLORES[perfil], zorder=5, linewidths=2)

    ax.axhline(y=100, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_title("Evolucion del portafolio 2019-2024 (base 100)", fontsize=14)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Valor (base 100)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    _formato_eje_fecha(ax)
    fig.tight_layout()

    if guardar:
        fig.savefig(guardar, dpi=150, bbox_inches="tight")
    return fig


def plot_drawdown(resultados: dict, benchmark: "pd.DataFrame | None" = None, guardar: str = None):
    """Drawdown mensual por perfil con lineas de tolerancia del perfil."""
    fig, ax = plt.subplots(figsize=(12, 5))

    if benchmark is not None:
        ax.plot(benchmark.index, -benchmark["drawdown"] * 100,
                label="Benchmark (30 acc. azar)", color="black",
                linewidth=1.5, linestyle="--", alpha=0.7)

    for perfil, df in resultados.items():
        if df.empty:
            continue
        ax.plot(df.index, -df["drawdown"] * 100,
                label=LABELS[perfil], color=COLORES[perfil], linewidth=2)
        tolerancia = RISK_PROFILES[perfil]
        if tolerancia > 0:
            ax.axhline(y=-tolerancia * 100, color=COLORES[perfil],
                       linestyle=":", linewidth=1, alpha=0.5)

    ax.set_title("Drawdown mensual por perfil 2019-2024", fontsize=14)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    _formato_eje_fecha(ax)
    fig.tight_layout()

    if guardar:
        fig.savefig(guardar, dpi=150, bbox_inches="tight")
    return fig


def plot_caja_chica(resultados: dict, guardar: str = None):
    """Fraccion en caja chica vs acciones a lo largo del tiempo por perfil."""
    perfiles = list(resultados.keys())
    n = len(perfiles)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), sharey=True)
    axes = axes.flatten()

    for idx, perfil in enumerate(perfiles):
        ax = axes[idx]
        df = resultados[perfil]
        if df.empty:
            ax.set_visible(False)
            continue
        acciones_pct = (1 - df["caja_chica"]) * 100
        ax.fill_between(df.index, 0, acciones_pct,
                        label="Acciones", color=COLORES[perfil], alpha=0.75)
        ax.fill_between(df.index, acciones_pct, 100,
                        label="Caja chica", color="lightgray", alpha=0.8)
        ax.set_title(LABELS[perfil], fontsize=10)
        ax.set_ylim(0, 100)
        ax.set_ylabel("% del portafolio")
        ax.grid(True, alpha=0.3)
        _formato_eje_fecha(ax)
        if idx == 0:
            ax.legend(loc="lower left", fontsize=8)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Composicion: acciones vs caja chica por perfil", fontsize=14, y=1.01)
    fig.tight_layout()

    if guardar:
        fig.savefig(guardar, dpi=150, bbox_inches="tight")
    return fig


def plot_curvas_p1(guardar: str = None):
    """Curvas de P1 en funcion del drawdown para cada perfil."""
    fig, ax = plt.subplots(figsize=(10, 5))
    drawdowns = np.linspace(0, 0.65, 400)

    for perfil, tolerancia in RISK_PROFILES.items():
        if tolerancia == 0.0:
            continue
        p1_vals = [p1_abandono(dd, tolerancia) for dd in drawdowns]
        ax.plot(drawdowns * 100, p1_vals,
                label=LABELS[perfil], color=COLORES[perfil], linewidth=2)
        ax.axvline(x=tolerancia * 100, color=COLORES[perfil],
                   linestyle=":", linewidth=1, alpha=0.4)

    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_title("Probabilidad de abandono P1 por perfil", fontsize=14)
    ax.set_xlabel("Drawdown desde el peak (%)")
    ax.set_ylabel("P1")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if guardar:
        fig.savefig(guardar, dpi=150, bbox_inches="tight")
    return fig


def plot_curva_p2(guardar: str = None):
    """Curva de P2 en funcion del turnover propuesto."""
    fig, ax = plt.subplots(figsize=(8, 4))
    turnovers = np.linspace(0, 0.50, 300)
    p2_vals = [p2_aceptacion(t) for t in turnovers]

    ax.plot(turnovers * 100, p2_vals, color="#333333", linewidth=2)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(x=10, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_title("Probabilidad de aceptacion de rebalanceo P2", fontsize=14)
    ax.set_xlabel("Turnover propuesto (%)")
    ax.set_ylabel("P2")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if guardar:
        fig.savefig(guardar, dpi=150, bbox_inches="tight")
    return fig


def plot_sensibilidad(
    df_sens: "pd.DataFrame",
    param_label: str,
    metrica: str = "retorno_anual",
    metrica_label: str = "Retorno anual (CAGR)",
    guardar: str | None = None,
):
    """
    Gráfico de sensibilidad OAT: barras agrupadas por valor del parámetro,
    coloreadas por perfil.

    Parámetros
    ----------
    df_sens     : DataFrame con MultiIndex (param_value, perfil) de run_sensitivity
    param_label : etiqueta del eje X (nombre del parámetro)
    metrica     : columna de df_sens a graficar (default 'retorno_anual')
    metrica_label : etiqueta del eje Y
    """
    import pandas as pd

    perfiles    = list(df_sens.index.get_level_values("perfil").unique())
    param_vals  = list(df_sens.index.get_level_values("param_value").unique())
    n_vals      = len(param_vals)
    n_perfiles  = len(perfiles)

    fig, ax = plt.subplots(figsize=(max(8, n_vals * n_perfiles * 0.6), 5))

    bar_width = 0.8 / n_perfiles
    x = np.arange(n_vals)

    for i, perfil in enumerate(perfiles):
        vals = [
            df_sens.loc[(pv, perfil), metrica] if (pv, perfil) in df_sens.index else np.nan
            for pv in param_vals
        ]
        offset = (i - n_perfiles / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, vals, width=bar_width,
                      color=COLORES.get(perfil, "gray"),
                      label=LABELS.get(perfil, perfil), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in param_vals])
    ax.set_xlabel(param_label)
    ax.set_ylabel(metrica_label)
    ax.set_title(f"Sensibilidad: {metrica_label} vs {param_label}")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()

    if guardar:
        fig.savefig(guardar, dpi=150, bbox_inches="tight")
    return fig


def generar_todas(
    resultados: dict,
    capital_inicial: float = 1_000_000,
    benchmark: "pd.DataFrame | None" = None,
    carpeta: str = "graficos",
):
    """Genera y guarda los 5 graficos del backtest en `carpeta`."""
    import os
    os.makedirs(carpeta, exist_ok=True)

    plot_valor_portafolios(resultados, capital_inicial, benchmark=benchmark,
                           guardar=f"{carpeta}/valor_portafolios.png")
    plot_drawdown(resultados, benchmark=benchmark,
                  guardar=f"{carpeta}/drawdown.png")
    plot_caja_chica(resultados,
                    guardar=f"{carpeta}/caja_chica.png")
    plot_curvas_p1(guardar=f"{carpeta}/curvas_p1.png")
    plot_curva_p2(guardar=f"{carpeta}/curva_p2.png")

    plt.close("all")
    print(f"Graficos guardados en {carpeta}/")
