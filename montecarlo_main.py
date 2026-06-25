"""
Simulacion Monte Carlo: N clientes independientes por perfil.
Los pesos optimos se calculan una sola vez; lo aleatorio es P1 y P2 por cliente.

Uso:
    python montecarlo_main.py
"""
import pandas as pd
from data.loader import load_universe, load_stock_info
from backtesting.runner import (
    run_all_profiles, run_monte_carlo, resumen_monte_carlo, calcular_ic
)

N_CLIENTES = 500


def main():
    print("Cargando datos...")
    price_returns, div_yields = load_universe()
    info        = load_stock_info()
    market_caps = info["marketCap"] if "marketCap" in info.columns else None

    print("\nPrecomputando Black-Litterman y pesos optimos...")
    _, mu_bl_cache, tickers = run_all_profiles(
        price_returns, div_yields,
        market_caps=market_caps,
        seed=42,
        return_cache=True,
        verbose=True,
    )

    print(f"\nSimulando {N_CLIENTES} clientes por perfil...")
    mc = run_monte_carlo(
        price_returns, div_yields,
        n_clientes=N_CLIENTES,
        mu_bl_cache=mu_bl_cache,
        tickers=tickers,
        seed_base=0,
        verbose=True,
    )

    resumen = resumen_monte_carlo(mc)

    print("\n" + "=" * 70)
    print(f"RESUMEN MONTE CARLO  ({N_CLIENTES} clientes por perfil, 2019-2024)")
    print("=" * 70)

    print("\n  Retorno anual promedio y tasa de abandono:")
    print(f"  {'Perfil':<20} {'Retorno prom':>14} {'Retorno (no abnd)':>18} {'Retorno std':>13} {'Tasa abandono':>14}")
    print(f"  {'-'*20} {'-'*14} {'-'*18} {'-'*13} {'-'*14}")
    for perfil, r in resumen.iterrows():
        print(
            f"  {perfil:<20} "
            f"{r['retorno_prom']:>13.1%} "
            f"{r['retorno_si_completa']:>17.1%} "
            f"{r['retorno_std']:>12.1%} "
            f"{r['tasa_abandono']:>13.1%}"
        )

    print("\n  Comportamiento del cliente:")
    print(f"  {'Perfil':<20} {'Meses activos':>14} {'Acepta rebalanceo':>18} {'Comision prom':>14}")
    print(f"  {'-'*20} {'-'*14} {'-'*18} {'-'*14}")
    for perfil, r in resumen.iterrows():
        print(
            f"  {perfil:<20} "
            f"{r['meses_activos_prom']:>13.1f} "
            f"{r['frec_rebalanceo_prom']:>17.1%} "
            f"${r['comision_prom']:>13,.0f}"
        )

    print("\n  Information Coefficient (acierto de predicciones BL):")
    ic = calcular_ic(price_returns, mu_bl_cache, tickers)
    print(f"  IC medio : {ic['IC_medio']:+.4f}")
    print(f"  ICIR     : {ic['ICIR']:+.4f}")
    print(f"  Meses IC>0: {ic['IC_pos']:.1%} de {ic['n_meses']} meses")

    # Guardar resultados
    resumen.to_csv("resumen_montecarlo.csv")
    for perfil, df in mc.items():
        df.to_csv(f"montecarlo_{perfil}.csv", index=False)
    print("\nResultados guardados en resumen_montecarlo.csv y montecarlo_<perfil>.csv")


if __name__ == "__main__":
    main()
