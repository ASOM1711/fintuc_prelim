import pandas as pd

from data.loader import load_universe, load_stock_info
from backtesting.runner import run_all_profiles, run_benchmark, run_benchmark_markowitz, resumen_backtest, calcular_ic
from visualization.plots import generar_todas


def main():
    print("Cargando datos...")
    price_returns, div_yields = load_universe()
    info        = load_stock_info()
    market_caps = info["marketCap"] if "marketCap" in info.columns else None

    print(f"  Tickers cargados : {price_returns.shape[1]}")
    print(f"  Periodo de datos : {price_returns.index[0].date()} -> {price_returns.index[-1].date()}")

    print("\nCorriendo backtest walk-forward 2019–2024...")
    resultados, mu_bl_cache, tickers = run_all_profiles(
        price_returns,
        div_yields,
        capital_inicial=1_000_000,
        eval_start="2019-01-01",
        eval_end="2024-12-31",
        market_caps=market_caps,
        seed=42,
        return_cache=True,
    )

    print("\n-- Resumen por perfil ------------------------------------------")
    resumenes = {}
    for perfil, df in resultados.items():
        r = resumen_backtest(df)
        resumenes[perfil] = r
        abandono_str = f"abandono mes {r['meses_activos']}" if r["abandono"] else "completo"
        print(
            f"  {perfil:<20} "
            f"ret={r['retorno_anual']:+.1%}  "
            f"sharpe={r['sharpe']:.2f}  "
            f"maxDD={r['max_drawdown']:.1%}  "
            f"({abandono_str})"
        )

    print("\n-- KPIs de negocio --------------------------------------------")
    print(f"  {'Perfil':<20} {'Comision total':>15} {'Rebalanceo':>12} {'Caja chica':>12}")
    print(f"  {'-'*20} {'-'*15} {'-'*12} {'-'*12}")
    for perfil, r in resumenes.items():
        print(
            f"  {perfil:<20} "
            f"${r['comision_total']:>14,.0f} "
            f"{r['frecuencia_rebalanceo']:>11.1%} "
            f"{r['caja_chica_promedio']:>11.1%}"
        )

    # Guardar resultados mensuales y resumen en CSV
    for perfil, df in resultados.items():
        df.to_csv(f"resultados_{perfil}.csv")

    pd.DataFrame(resumenes).T.to_csv("resumen_backtest.csv")
    print("\nResultados guardados en resultados_<perfil>.csv y resumen_backtest.csv")

    print("\n-- Information Coefficient (IC) del modelo BL ------------------")
    ic = calcular_ic(price_returns, mu_bl_cache, tickers)
    print(f"  IC medio  : {ic['IC_medio']:+.4f}  (> 0 = prediccion correcta del ranking)")
    print(f"  IC std    : {ic['IC_std']:.4f}")
    print(f"  ICIR      : {ic['ICIR']:+.4f}  (> 0.5 = senal consistente)")
    print(f"  Meses IC>0: {ic['IC_pos']:.1%}  ({ic['n_meses']} meses evaluados)")

    print("\nCorriendo benchmark (30 acciones al azar)...")
    benchmark = run_benchmark(price_returns, div_yields, seed=42)
    bm_ret = (benchmark["valor"].iloc[-1] / 1_000_000) ** (1 / 6) - 1
    print(f"  Benchmark retorno anual: {bm_ret:+.1%}")

    print("\nCorriendo caso base Markowitz puro (sin Black-Litterman)...")
    res_markowitz = run_benchmark_markowitz(price_returns, div_yields, seed=42)
    print("\n-- Caso base Markowitz vs BL+Markowitz -------------------------")
    print(f"  {'Perfil':<20} {'Markowitz':>12} {'BL+Markowitz':>14} {'Diferencia':>12}")
    print(f"  {'-'*20} {'-'*12} {'-'*14} {'-'*12}")
    for perfil in res_markowitz:
        r_mk = resumen_backtest(res_markowitz[perfil])
        r_bl = resumenes[perfil]
        diff = r_bl["retorno_anual"] - r_mk["retorno_anual"]
        print(
            f"  {perfil:<20} "
            f"{r_mk['retorno_anual']:>+11.1%} "
            f"{r_bl['retorno_anual']:>+13.1%} "
            f"{diff:>+11.1%}"
        )

    print("\nGenerando graficos...")
    generar_todas(resultados, capital_inicial=1_000_000,
                  benchmark=benchmark, carpeta="graficos")


if __name__ == "__main__":
    main()
