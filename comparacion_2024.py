"""
Comparacion de los tres modelos para el año 2024:
  1. Benchmark 1/30 (30 acciones al azar, pesos iguales)
  2. Caso base: Markowitz puro (media historica, sin BL)
  3. Modelo final: BL + Markowitz

KPIs: retorno anual, Sharpe, max drawdown, tasa abandono, comision total
"""
import numpy as np
import pandas as pd

from data.loader import load_universe, load_stock_info
from backtesting.runner import (
    run_all_profiles, run_benchmark, run_benchmark_markowitz,
    run_monte_carlo, resumen_backtest, resumen_monte_carlo,
)

EVAL_START = "2024-01-01"
EVAL_END   = "2024-12-31"
CAPITAL    = 1_000_000
SEED       = 42
N_CLIENTES = 500

PERFILES_ORDEN = [
    "muy_conservador", "conservador", "neutro", "arriesgado", "muy_arriesgado"
]

LABELS = {
    "muy_conservador": "Muy conservador",
    "conservador":     "Conservador",
    "neutro":          "Neutro",
    "arriesgado":      "Arriesgado",
    "muy_arriesgado":  "Muy arriesgado",
}


def kpis(res_backtest: dict, res_mc: dict | None = None) -> pd.DataFrame:
    rows = []
    for perfil in PERFILES_ORDEN:
        df = res_backtest.get(perfil)
        if df is None or df.empty:
            continue
        r = resumen_backtest(df)
        row = {
            "Perfil":         LABELS[perfil],
            "Retorno anual":  f"{r['retorno_anual']:+.1%}",
            "Sharpe":         f"{r['sharpe']:.2f}",
            "Max drawdown":   f"{r['max_drawdown']:.1%}",
            "Comision total": f"${r['comision_total']:,.0f}",
        }
        if res_mc is not None and perfil in res_mc:
            mc_df  = res_mc[perfil]
            tasa   = mc_df["abandona"].mean()
            row["Tasa abandono"] = f"{tasa:.1%}"
        else:
            row["Tasa abandono"] = "—"
        rows.append(row)
    return pd.DataFrame(rows).set_index("Perfil")


def main():
    print("Cargando datos...")
    price_returns, div_yields = load_universe()
    info        = load_stock_info()
    market_caps = info["marketCap"] if "marketCap" in info.columns else None
    print(f"  Periodo disponible: {price_returns.index[0].date()} -> {price_returns.index[-1].date()}")

    # ── Modelo final: BL + Markowitz ────────────────────────────────────────
    print("\n[1/3] Modelo final: BL + Markowitz...")
    res_bl, mu_bl_cache, tickers = run_all_profiles(
        price_returns, div_yields,
        capital_inicial=CAPITAL,
        eval_start=EVAL_START,
        eval_end=EVAL_END,
        market_caps=market_caps,
        lam=None,           # calibra lambda por perfil
        full_invest=True,
        seed=SEED,
        return_cache=True,
        verbose=True,
    )

    print("  Monte Carlo BL+Markowitz...")
    mc_bl = run_monte_carlo(
        price_returns, div_yields,
        n_clientes=N_CLIENTES,
        mu_bl_cache=mu_bl_cache,
        tickers=tickers,
        capital_inicial=CAPITAL,
        eval_start=EVAL_START,
        eval_end=EVAL_END,
        full_invest=True,
        seed_base=0,
        verbose=False,
    )

    # ── Caso base: Markowitz puro ────────────────────────────────────────────
    print("\n[2/3] Caso base: Markowitz puro (sin BL)...")
    res_mk = run_benchmark_markowitz(
        price_returns, div_yields,
        capital_inicial=CAPITAL,
        eval_start=EVAL_START,
        eval_end=EVAL_END,
        full_invest=True,
        seed=SEED,
        verbose=True,
    )

    # ── Benchmark: 1/30 acciones al azar ────────────────────────────────────
    print("\n[3/3] Benchmark: 1/30 acciones al azar...")
    res_bm = run_benchmark(
        price_returns, div_yields,
        n_acciones=30,
        capital_inicial=CAPITAL,
        eval_start=EVAL_START,
        eval_end=EVAL_END,
        seed=SEED,
    )
    n_meses_bm  = len(res_bm)
    ret_bm      = (1 + res_bm["valor"].pct_change().dropna()).prod() ** (12 / n_meses_bm) - 1
    vol_bm      = res_bm["valor"].pct_change().dropna().std() * np.sqrt(12)
    sharpe_bm   = ret_bm / vol_bm if vol_bm > 1e-6 else 0.0
    maxdd_bm    = res_bm["drawdown"].max()

    # ── Imprimir tablas ──────────────────────────────────────────────────────
    sep = "=" * 75

    print(f"\n{sep}")
    print("RESULTADOS 2024 — MODELO FINAL (BL + Markowitz)")
    print(sep)
    print(kpis(res_bl, mc_bl).to_string())

    print(f"\n{sep}")
    print("RESULTADOS 2024 — CASO BASE (Markowitz puro, sin BL)")
    print(sep)
    print(kpis(res_mk).to_string())

    print(f"\n{sep}")
    print("RESULTADOS 2024 — BENCHMARK (1/30 acciones al azar)")
    print(sep)
    print(f"  Retorno anual : {ret_bm:+.1%}")
    print(f"  Sharpe        : {sharpe_bm:.2f}")
    print(f"  Max drawdown  : {maxdd_bm:.1%}")
    print(f"  Comision      : $0  (sin comision)")
    print(f"  Tasa abandono : —  (sin modelo de cliente)")

    print(f"\n{sep}")
    print("RESUMEN MONTE CARLO 2024 — BL + Markowitz")
    print(sep)
    print(resumen_monte_carlo(mc_bl).to_string())


if __name__ == "__main__":
    main()
