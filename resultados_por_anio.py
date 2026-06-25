"""
Resultados por año (2022, 2023, 2024) para los tres modelos:
  - BL + Markowitz (modelo final)
  - Markowitz puro (caso base metodologico)
  - Benchmark 1/30 (caso base simple)

KPIs: retorno anual, Sharpe, max drawdown, tasa aceptacion, tasa abandono, comision
"""
import numpy as np
import pandas as pd

from data.loader import load_universe, load_stock_info
from backtesting.runner import (
    run_all_profiles, run_benchmark, run_benchmark_markowitz,
    run_monte_carlo, resumen_backtest, resumen_monte_carlo,
)

ANIOS       = [2022, 2023, 2024]
CAPITAL     = 1_000_000
SEED        = 42
N_CLIENTES  = 500

PERFILES = [
    "muy_conservador", "conservador", "neutro", "arriesgado", "muy_arriesgado"
]
LABELS = {
    "muy_conservador": "Muy conserv.",
    "conservador":     "Conservador",
    "neutro":          "Neutro",
    "arriesgado":      "Arriesgado",
    "muy_arriesgado":  "Muy arriesg.",
}


def tabla_año(año, res_bl, mc_bl, res_mk, res_bm):
    """Imprime tabla comparativa para un año dado."""
    n_meses_bm = len(res_bm)
    ret_bm = (1 + res_bm["valor"].pct_change().dropna()).prod() ** (12 / n_meses_bm) - 1
    mdd_bm = res_bm["drawdown"].max()

    sep = "=" * 118
    print(f"\n{sep}")
    print(f"AÑO {año}")
    print(sep)
    print(f"{'Perfil':<16}  "
          f"{'--- BL + Markowitz ---':^50}  "
          f"{'--- Markowitz puro ---':^50}")
    print(f"{'':16}  "
          f"{'Retorno':>8} {'Acept':>7} {'Aband':>7} {'Comis%':>7} {'MaxDD':>7}  "
          f"{'Retorno':>8} {'Acept':>7} {'Aband':>7} {'Comis%':>7} {'MaxDD':>7}")
    print("-" * 118)

    for perfil in PERFILES:
        df_bl = res_bl.get(perfil, pd.DataFrame())
        df_mk = res_mk.get(perfil, pd.DataFrame())
        mc_p  = mc_bl.get(perfil, pd.DataFrame())

        r_bl = resumen_backtest(df_bl) if not df_bl.empty else {}
        r_mk = resumen_backtest(df_mk) if not df_mk.empty else {}

        ret_bl   = f"{r_bl['retorno_anual']:+.1%}"          if r_bl else "—"
        acept_bl = f"{df_bl['rebalanceo'].mean():.1%}"       if not df_bl.empty else "—"
        aband_bl = f"{mc_p['abandona'].mean():.1%}"          if not mc_p.empty else "—"
        com_bl   = f"{r_bl['comision_total']/CAPITAL:.2%}"   if r_bl else "—"
        mdd_bl   = f"{r_bl['max_drawdown']:.1%}"             if r_bl else "—"

        ret_mk   = f"{r_mk['retorno_anual']:+.1%}"           if r_mk else "—"
        acept_mk = f"{df_mk['rebalanceo'].mean():.1%}"       if not df_mk.empty else "—"
        aband_mk = f"{int(df_mk['abandona'].any()):.0%}"     if not df_mk.empty else "—"
        com_mk   = f"{r_mk['comision_total']/CAPITAL:.2%}"   if r_mk else "—"
        mdd_mk   = f"{r_mk['max_drawdown']:.1%}"             if r_mk else "—"

        print(f"{LABELS[perfil]:<16}  "
              f"{ret_bl:>8} {acept_bl:>7} {aband_bl:>7} {com_bl:>7} {mdd_bl:>7}  "
              f"{ret_mk:>8} {acept_mk:>7} {aband_mk:>7} {com_mk:>7} {mdd_mk:>7}")

    print("-" * 118)
    print(f"{'Benchmark 1/30':<16}  "
          f"{ret_bm:>+8.1%} {'N/A':>7} {'N/A':>7} {'0.00%':>7} {mdd_bm:>7.1%}  "
          f"{'N/A':>8} {'N/A':>7} {'N/A':>7} {'N/A':>7} {'N/A':>7}")


def main():
    print("Cargando datos...")
    price_returns, div_yields = load_universe()
    info        = load_stock_info()
    market_caps = info["marketCap"] if "marketCap" in info.columns else None

    for año in ANIOS:
        start = f"{año}-01-01"
        end   = f"{año}-12-31"
        print(f"\n{'='*50}")
        print(f"Procesando {año}...")
        print(f"{'='*50}")

        # BL + Markowitz
        print(f"  [BL+Markowitz]...")
        res_bl, mu_bl_cache, tickers = run_all_profiles(
            price_returns, div_yields,
            capital_inicial=CAPITAL,
            eval_start=start, eval_end=end,
            market_caps=market_caps,
            lam=None, full_invest=True,
            seed=SEED, return_cache=True, verbose=False,
        )
        mc_bl = run_monte_carlo(
            price_returns, div_yields,
            n_clientes=N_CLIENTES,
            mu_bl_cache=mu_bl_cache, tickers=tickers,
            capital_inicial=CAPITAL,
            eval_start=start, eval_end=end,
            full_invest=True, seed_base=0, verbose=False,
        )

        # Markowitz puro
        print(f"  [Markowitz puro]...")
        res_mk = run_benchmark_markowitz(
            price_returns, div_yields,
            capital_inicial=CAPITAL,
            eval_start=start, eval_end=end,
            full_invest=True, seed=SEED, verbose=False,
        )

        # Benchmark 1/30
        print(f"  [Benchmark 1/30]...")
        res_bm = run_benchmark(
            price_returns, div_yields,
            n_acciones=30, capital_inicial=CAPITAL,
            eval_start=start, eval_end=end, seed=SEED,
        )

        tabla_año(año, res_bl, mc_bl, res_mk, res_bm)


if __name__ == "__main__":
    main()
