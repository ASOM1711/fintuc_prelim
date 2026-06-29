"""
Comparacion: Markowitz + BL  vs  CVaR + BL

Ambos modelos usan la misma senal Black-Litterman (mu_bl_cache compartido).
La diferencia es el criterio de riesgo:
  - Markowitz: restriccion de VARIANZA anualizada <= (Tol * VOL_FACTOR)^2
  - CVaR:      restriccion CVaR_0.95 diario <= Tol * VOL_FACTOR * 2.063 / sqrt(252)
               (equivalente bajo normalidad al constraint de Markowitz)

Uso:
    cd E3definitivo
    python comparacion_cvar.py

Salidas:
    comparacion_cvar_resumen.csv        tabla comparativa por perfil
    output/pdf/comparacion_cvar.pdf     grafico de valor del portafolio en el tiempo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from data.loader import load_universe, load_stock_info
from backtesting.runner import run_all_profiles, resumen_backtest
from portfolio.engine import PortfolioState, paso_mensual
from models.cvar import optimizar_cvar, cvar_limit_para_perfil
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

CAPITAL    = 1_000_000
EVAL_START = "2019-01-01"
EVAL_END   = "2024-12-31"
ALPHA      = 0.95


def _train_window(returns: pd.DataFrame, mes: pd.Timestamp, train_years: int) -> pd.DataFrame:
    end   = mes - pd.Timedelta(days=1)
    start = mes - pd.DateOffset(years=train_years)
    return returns.loc[start:end]


def _run_backtest_cvar(
    pr: pd.DataFrame,
    dv: pd.DataFrame,
    perfil: str,
    mu_bl_cache: dict,
    tickers: list,
    eval_start: str = EVAL_START,
    eval_end: str = EVAL_END,
    train_years: int = TRAIN_WINDOW_YEARS,
    capital_inicial: float = CAPITAL,
    commission: float = COMMISSION,
    alpha: float = ALPHA,
    full_invest: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """Backtest walk-forward con CVaR como restriccion de riesgo."""
    tolerancia = RISK_PROFILES[perfil]
    cvar_lim   = cvar_limit_para_perfil(tolerancia)
    max_weight = PROFILE_MAX_WEIGHTS.get(perfil, 0.025)
    rng        = np.random.default_rng(seed)
    meses      = pd.date_range(eval_start, eval_end, freq="MS")

    pr_f = pr[tickers]
    dv_f = dv.reindex(columns=tickers).fillna(0)

    def _pesos(mes: pd.Timestamp) -> np.ndarray:
        train  = _train_window(pr_f, mes, train_years)
        mu_bl  = mu_bl_cache.get(mes)
        return optimizar_cvar(
            train, cvar_lim, perfil,
            mu_personalizado=mu_bl,
            max_weight=max_weight,
            alpha=alpha,
            full_invest=full_invest,
        )

    pesos_0 = _pesos(meses[0])
    state   = PortfolioState.crear(list(tickers), capital_inicial, pesos_0)

    registros = []
    for mes in meses:
        mes_end = mes + pd.offsets.MonthEnd(0)
        pr_mes  = pr_f.loc[mes:mes_end]
        dv_mes  = dv_f.loc[mes:mes_end]
        if pr_mes.empty:
            continue

        pesos_opt = _pesos(mes)
        mu_signal = mu_bl_cache.get(mes)

        metricas = paso_mensual(
            state, pr_mes, dv_mes, pesos_opt, tolerancia,
            mu_bl=mu_signal,
            commission_rate=commission,
            rng=rng,
        )
        metricas["fecha"]  = mes
        metricas["perfil"] = perfil
        registros.append(metricas)

        if metricas["abandona"]:
            break

    return pd.DataFrame(registros).set_index("fecha")


def _tabla_comparacion(
    resultados_mk: dict[str, pd.DataFrame],
    resultados_cvar: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    filas = []
    for perfil in RISK_PROFILES:
        mk = resumen_backtest(resultados_mk[perfil])
        cv = resumen_backtest(resultados_cvar[perfil])
        filas.append({
            "perfil":        perfil,
            "MK_retorno":    mk["retorno_anual"],
            "CVaR_retorno":  cv["retorno_anual"],
            "MK_sharpe":     mk["sharpe"],
            "CVaR_sharpe":   cv["sharpe"],
            "MK_vol":        mk["volatilidad_anual"],
            "CVaR_vol":      cv["volatilidad_anual"],
            "MK_maxDD":      mk["max_drawdown"],
            "CVaR_maxDD":    cv["max_drawdown"],
            "MK_abandono":   mk["abandono"],
            "CVaR_abandono": cv["abandono"],
            "MK_meses":      mk["meses_activos"],
            "CVaR_meses":    cv["meses_activos"],
            "MK_comision":   mk["comision_total"],
            "CVaR_comision": cv["comision_total"],
        })
    return pd.DataFrame(filas).set_index("perfil")


def _imprimir_tabla(df: pd.DataFrame) -> None:
    sep = "=" * 108
    print("\n" + sep)
    print(f"COMPARACION: Markowitz + BL  vs  CVaR (alpha={ALPHA})  --  Backtest {EVAL_START[:4]}-{EVAL_END[:4]}")
    print(sep)
    print(f"{'Perfil':<22} {'Retorno':>14}  {'Sharpe':>12}  {'Vol':>12}  {'MaxDD':>12}  {'Abandona':>10}  {'Meses':>8}")
    print(f"{'':22} {'MK':>7} {'CVaR':>6}  {'MK':>6} {'CVaR':>5}  {'MK':>6} {'CVaR':>5}  {'MK':>6} {'CVaR':>5}  {'MK':>5} {'CVaR':>4}  {'MK':>4} {'CVaR':>3}")
    print("-" * 108)
    for perfil, row in df.iterrows():
        print(
            f"{perfil:<22}"
            f" {row['MK_retorno']:>6.1%} {row['CVaR_retorno']:>6.1%}"
            f"  {row['MK_sharpe']:>6.2f} {row['CVaR_sharpe']:>5.2f}"
            f"  {row['MK_vol']:>6.1%} {row['CVaR_vol']:>5.1%}"
            f"  {row['MK_maxDD']:>6.1%} {row['CVaR_maxDD']:>5.1%}"
            f"  {str(row['MK_abandono']):>5} {str(row['CVaR_abandono']):>4}"
            f"  {int(row['MK_meses']):>4} {int(row['CVaR_meses']):>3}"
        )
    print(sep)


def _plot_valores(
    resultados_mk: dict[str, pd.DataFrame],
    resultados_cvar: dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    perfiles_plot = ["conservador", "neutro", "arriesgado", "muy_arriesgado"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, perfil in zip(axes, perfiles_plot):
        df_mk   = resultados_mk[perfil]
        df_cvar = resultados_cvar[perfil]

        ax.plot(df_mk.index,   df_mk["valor"]   / CAPITAL, label="Markowitz + BL", linewidth=1.8)
        ax.plot(df_cvar.index, df_cvar["valor"] / CAPITAL, label="CVaR + BL",
                linewidth=1.8, linestyle="--", color="darkorange")
        ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8)
        ax.set_title(perfil.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel("Valor relativo (inicial = 1)")
        ax.legend(fontsize=9)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Markowitz + BL  vs  CVaR (alpha={ALPHA}) + BL   "
        f"(backtest {EVAL_START[:4]}-{EVAL_END[:4]})",
        fontsize=13,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Grafico guardado en: {out_path}")


def main() -> None:
    print("Cargando datos...")
    pr, dv = load_universe()
    if pr.empty:
        raise FileNotFoundError("No se encontraron datos de precio.")

    try:
        info        = load_stock_info()
        market_caps = info["marketCap"] if "marketCap" in info.columns else None
    except FileNotFoundError:
        market_caps = None

    # 1. Markowitz + BL (caso base del proyecto)
    print("\n[1/2] Corriendo Markowitz + BL...")
    resultados_mk, mu_bl_cache, tickers = run_all_profiles(
        pr, dv,
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
        return_cache=True,
        verbose=True,
    )

    # 2. CVaR + BL (mismo mu_bl, distinta restriccion de riesgo)
    print("\n[2/2] Corriendo CVaR (alpha=0.95) + BL...")
    print("      Limites CVaR diario por perfil:")
    for perfil, tol in RISK_PROFILES.items():
        lim = cvar_limit_para_perfil(tol)
        mw  = PROFILE_MAX_WEIGHTS.get(perfil, 0.025)
        print(f"        {perfil:<22} tol={tol:.0%}  max_w={mw:.1%}  CVaR_lim={lim:.4%}/dia")

    pr_f = pr[tickers]
    dv_f = dv.reindex(columns=tickers).fillna(0)

    resultados_cvar = {}
    for perfil in RISK_PROFILES:
        print(f"  {perfil}...", end="", flush=True)
        resultados_cvar[perfil] = _run_backtest_cvar(
            pr_f, dv_f, perfil, mu_bl_cache, tickers,
            eval_start=EVAL_START,
            eval_end=EVAL_END,
            train_years=TRAIN_WINDOW_YEARS,
            capital_inicial=CAPITAL,
            commission=COMMISSION,
            alpha=ALPHA,
            full_invest=False,
            seed=42,
        )
        n = len(resultados_cvar[perfil])
        print(f" {n} meses")

    # 3. Tabla comparativa
    df_comp = _tabla_comparacion(resultados_mk, resultados_cvar)
    _imprimir_tabla(df_comp)

    root = Path(__file__).parent
    csv_path = root / "comparacion_cvar_resumen.csv"
    df_comp.to_csv(str(csv_path))
    print(f"\nTabla guardada en: {csv_path}")

    # 4. Grafico
    pdf_path = root / "output" / "pdf" / "comparacion_cvar.pdf"
    _plot_valores(resultados_mk, resultados_cvar, pdf_path)


if __name__ == "__main__":
    main()
