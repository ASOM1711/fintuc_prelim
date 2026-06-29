"""
Proyección de retornos bajo tres escenarios de mercado.

Metodología
-----------
Cada escenario aplica un drift mensual constante a los retornos de precio
a partir de eval_start. Esto representa un desplazamiento persistente del
mercado respecto al caso base histórico:

    Desfavorable : -10 % anual  →  mercado sistemáticamente peor
    Neutro       :   0 % anual  →  caso base (retornos históricos reales)
    Favorable    : +10 % anual  →  mercado sistemáticamente mejor

El drift se aplica desde eval_start en adelante, de modo que la ventana de
entrenamiento de Black-Litterman lo incorpora progresivamente — igual que
ocurriría en la práctica si el régimen de mercado cambia.

Los shocks de ±10 % corresponden aproximadamente a ±0.67 σ de la
distribución histórica anual del S&P 500 (σ ≈ 15 %), representando
escenarios moderados, no extremos.

Uso
---
    cd E3definitivo
    python escenarios_main.py

Salidas
-------
    escenarios_resumen.csv              tabla por escenario × perfil
    escenarios_montecarlo_resumen.csv   distribución por escenario × perfil
    output/pdf/escenarios_valor.pdf     valor del portafolio en el tiempo
    output/pdf/escenarios_montecarlo.pdf distribución de retornos
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from data.loader import load_universe, load_stock_info
from backtesting.runner import (
    run_all_profiles,
    run_monte_carlo,
    resumen_backtest,
    resumen_monte_carlo,
)
from config import (
    RISK_PROFILES,
    TRAIN_WINDOW_YEARS,
    PROFILE_MAX_WEIGHTS,
    COMMISSION,
    BL_METHOD,
    BL_CONF_BASE,
    BL_LOOKBACK,
    BL_SKIP,
)

# ── Configuración ────────────────────────────────────────────────────────────

CAPITAL     = 1_000_000
EVAL_START  = "2019-01-01"
EVAL_END    = "2024-12-31"
N_CLIENTES  = 300          # clientes Monte Carlo por escenario/perfil
SEED        = 42

ESCENARIOS: dict[str, float] = {
    "desfavorable": -0.10,   # -10 % anual
    "neutro":        0.00,   #   0 % (base)
    "favorable":    +0.10,   # +10 % anual
}

PERFILES_ORDEN = [
    "muy_conservador", "conservador", "neutro", "arriesgado", "muy_arriesgado"
]

COLORES = {
    "desfavorable": "#d62728",
    "neutro":       "#1f77b4",
    "favorable":    "#2ca02c",
}


# ── Utilidades ───────────────────────────────────────────────────────────────

def apply_market_shock(
    price_returns: pd.DataFrame,
    annual_shock: float,
    desde: str = EVAL_START,
) -> pd.DataFrame:
    """
    Suma un drift mensual constante a todos los retornos desde `desde`.

    El drift diario se obtiene como: (1 + annual_shock)^(1/252) - 1
    para preservar la capitalización compuesta.
    """
    if annual_shock == 0.0:
        return price_returns
    daily_shock = (1.0 + annual_shock) ** (1.0 / 252) - 1
    pr = price_returns.copy()
    pr.loc[pr.index >= desde] += daily_shock
    return pr


def _etiqueta(perfil: str) -> str:
    return perfil.replace("_", " ").title()


# ── Resultados por escenario ─────────────────────────────────────────────────

def correr_escenarios(
    price_returns: pd.DataFrame,
    div_yields: pd.DataFrame,
    market_caps,
    verbose: bool = True,
) -> tuple[dict, dict, dict]:
    """
    Corre run_all_profiles y run_monte_carlo para los 3 escenarios.

    Retorna
    -------
    resultados_bt  : {escenario -> {perfil -> df_backtest}}
    resultados_mc  : {escenario -> {perfil -> df_montecarlo}}
    mu_caches      : {escenario -> mu_bl_cache}  (para IC u otros análisis)
    """
    resultados_bt: dict  = {}
    resultados_mc: dict  = {}
    mu_caches:     dict  = {}

    for escenario, shock in ESCENARIOS.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"  ESCENARIO: {escenario.upper()}  (shock={shock:+.0%}/año)")
            print(f"{'='*60}")

        pr_esc = apply_market_shock(price_returns, shock, desde=EVAL_START)

        if verbose:
            print("  Backtest walk-forward...")

        res_bt, mu_cache, tickers = run_all_profiles(
            pr_esc, div_yields,
            capital_inicial=CAPITAL,
            eval_start=EVAL_START,
            eval_end=EVAL_END,
            train_years=TRAIN_WINDOW_YEARS,
            market_caps=market_caps,
            commission=COMMISSION,
            bl_method=BL_METHOD,
            conf_base=BL_CONF_BASE,
            lookback=BL_LOOKBACK,
            skip=BL_SKIP,
            seed=SEED,
            return_cache=True,
            verbose=verbose,
        )

        if verbose:
            print(f"  Monte Carlo ({N_CLIENTES} clientes)...")

        res_mc = run_monte_carlo(
            pr_esc, div_yields,
            n_clientes=N_CLIENTES,
            mu_bl_cache=mu_cache,
            tickers=tickers,
            capital_inicial=CAPITAL,
            eval_start=EVAL_START,
            eval_end=EVAL_END,
            seed_base=SEED,
            verbose=verbose,
        )

        resultados_bt[escenario] = res_bt
        resultados_mc[escenario] = res_mc
        mu_caches[escenario]     = mu_cache

    return resultados_bt, resultados_mc, mu_caches


# ── Tablas resumen ───────────────────────────────────────────────────────────

def tabla_backtest(resultados_bt: dict) -> pd.DataFrame:
    """Tabla plana: (escenario, perfil) × métricas de backtest."""
    filas = []
    for escenario, res in resultados_bt.items():
        for perfil in PERFILES_ORDEN:
            df = res.get(perfil)
            if df is None or df.empty:
                continue
            r = resumen_backtest(df)
            filas.append({
                "escenario":      escenario,
                "perfil":         perfil,
                "retorno_anual":  r["retorno_anual"],
                "volatilidad":    r["volatilidad_anual"],
                "sharpe":         r["sharpe"],
                "max_drawdown":   r["max_drawdown"],
                "meses_activos":  r["meses_activos"],
                "abandono":       r["abandono"],
                "comision_total": r["comision_total"],
            })
    return pd.DataFrame(filas).set_index(["escenario", "perfil"])


def tabla_montecarlo(resultados_mc: dict) -> pd.DataFrame:
    """Tabla plana: (escenario, perfil) × métricas Monte Carlo."""
    filas = []
    for escenario, mc in resultados_mc.items():
        resumen = resumen_monte_carlo(mc)
        for perfil in PERFILES_ORDEN:
            if perfil not in resumen.index:
                continue
            r = resumen.loc[perfil]
            filas.append({
                "escenario":           escenario,
                "perfil":              perfil,
                "retorno_prom":        r["retorno_prom"],
                "retorno_std":         r["retorno_std"],
                "retorno_p10":         mc[perfil]["retorno_anual"].quantile(0.10),
                "retorno_p50":         mc[perfil]["retorno_anual"].quantile(0.50),
                "retorno_p90":         mc[perfil]["retorno_anual"].quantile(0.90),
                "tasa_abandono":       r["tasa_abandono"],
                "meses_activos_prom":  r["meses_activos_prom"],
                "comision_prom":       r["comision_prom"],
            })
    return pd.DataFrame(filas).set_index(["escenario", "perfil"])


def imprimir_resumen(df_bt: pd.DataFrame, df_mc: pd.DataFrame) -> None:
    sep = "=" * 85
    for escenario in ESCENARIOS:
        shock = ESCENARIOS[escenario]
        print(f"\n{sep}")
        print(f"  ESCENARIO {escenario.upper()}  (drift mercado: {shock:+.0%}/año)")
        print(sep)
        print(f"  {'Perfil':<22} {'Retorno':>9} {'Sharpe':>7} {'MaxDD':>7} "
              f"{'Meses':>7} {'T.Abnd':>7} {'p10':>7} {'p50':>7} {'p90':>7}")
        print(f"  {'-'*22} {'-'*9} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

        for perfil in PERFILES_ORDEN:
            try:
                bt = df_bt.loc[(escenario, perfil)]
                mc = df_mc.loc[(escenario, perfil)]
            except KeyError:
                continue

            abandono_str = "Sí" if bt["abandono"] else "No"
            print(
                f"  {_etiqueta(perfil):<22}"
                f" {bt['retorno_anual']:>8.1%}"
                f" {bt['sharpe']:>7.2f}"
                f" {bt['max_drawdown']:>7.1%}"
                f" {int(bt['meses_activos']):>7}"
                f" {abandono_str:>7}"
                f" {mc['retorno_p10']:>7.1%}"
                f" {mc['retorno_p50']:>7.1%}"
                f" {mc['retorno_p90']:>7.1%}"
            )
    print(f"\n{sep}")


# ── Gráficos ─────────────────────────────────────────────────────────────────

def plot_valor_portafolio(resultados_bt: dict, out_path: Path) -> None:
    """Evolución del valor del portafolio por perfil y escenario."""
    perfiles_plot = ["conservador", "neutro", "arriesgado", "muy_arriesgado"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, perfil in zip(axes, perfiles_plot):
        for escenario, res in resultados_bt.items():
            df = res.get(perfil)
            if df is None or df.empty:
                continue
            ax.plot(
                df.index, df["valor"] / CAPITAL,
                label=escenario.capitalize(),
                color=COLORES[escenario],
                linewidth=1.8,
                linestyle="--" if escenario == "desfavorable" else
                          ":"  if escenario == "favorable" else "-",
            )
        ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.set_title(_etiqueta(perfil), fontsize=11)
        ax.set_ylabel("Valor relativo (inicial = 1)")
        ax.legend(fontsize=9)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Evolución del portafolio por escenario de mercado  "
        f"({EVAL_START[:4]}–{EVAL_END[:4]})",
        fontsize=13,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico guardado en: {out_path}")


def plot_distribucion_montecarlo(resultados_mc: dict, out_path: Path) -> None:
    """Distribución de retornos anuales por perfil y escenario (boxplot)."""
    perfiles_plot = ["conservador", "neutro", "arriesgado", "muy_arriesgado"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    escenario_labels = list(ESCENARIOS.keys())
    x = np.arange(len(escenario_labels))
    width = 0.6

    for ax, perfil in zip(axes, perfiles_plot):
        data_boxes = []
        colors_box = []
        for escenario in escenario_labels:
            mc = resultados_mc.get(escenario, {}).get(perfil)
            if mc is not None and not mc.empty:
                data_boxes.append(mc["retorno_anual"].values)
            else:
                data_boxes.append(np.array([]))
            colors_box.append(COLORES[escenario])

        bp = ax.boxplot(
            data_boxes,
            positions=x,
            widths=width * 0.8,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
        )
        for patch, color in zip(bp["boxes"], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.axhline(0.0, color="grey", linestyle=":", linewidth=0.8)
        ax.set_title(_etiqueta(perfil), fontsize=11)
        ax.set_ylabel("Retorno anual")
        ax.set_xticks(x)
        ax.set_xticklabels([e.capitalize() for e in escenario_labels])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"Distribución de retornos por escenario (Monte Carlo, n={N_CLIENTES})  "
        f"({EVAL_START[:4]}–{EVAL_END[:4]})",
        fontsize=13,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico guardado en: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Cargando datos...")
    price_returns, div_yields = load_universe()
    if price_returns.empty:
        raise FileNotFoundError("No se encontraron datos de precio.")

    try:
        info        = load_stock_info()
        market_caps = info["marketCap"] if "marketCap" in info.columns else None
    except FileNotFoundError:
        market_caps = None

    print(f"  Periodo: {price_returns.index[0].date()} -> {price_returns.index[-1].date()}")
    print(f"  Escenarios: {list(ESCENARIOS.keys())}  (shocks: {list(ESCENARIOS.values())})")
    print(f"  Monte Carlo: {N_CLIENTES} clientes por escenario × perfil")

    # ── Correr los 3 escenarios ──────────────────────────────────────────────
    resultados_bt, resultados_mc, _ = correr_escenarios(
        price_returns, div_yields, market_caps, verbose=True
    )

    # ── Tablas ───────────────────────────────────────────────────────────────
    df_bt = tabla_backtest(resultados_bt)
    df_mc = tabla_montecarlo(resultados_mc)

    imprimir_resumen(df_bt, df_mc)

    root = Path(__file__).parent
    df_bt.to_csv(str(root / "escenarios_resumen.csv"))
    df_mc.to_csv(str(root / "escenarios_montecarlo_resumen.csv"))
    print("\nTablas guardadas en escenarios_resumen.csv y escenarios_montecarlo_resumen.csv")

    # ── Gráficos ─────────────────────────────────────────────────────────────
    plot_valor_portafolio(
        resultados_bt,
        root / "output" / "pdf" / "escenarios_valor.pdf",
    )
    plot_distribucion_montecarlo(
        resultados_mc,
        root / "output" / "pdf" / "escenarios_montecarlo.pdf",
    )


if __name__ == "__main__":
    main()
