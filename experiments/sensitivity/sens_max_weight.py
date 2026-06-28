"""
Sensibilidad OAT para max_weight (peso maximo por activo).
Corre para 2024 con valores [10%, 15%, 20%].
"""
import pandas as pd
from data.loader import load_universe, load_stock_info
from backtesting.runner import run_all_profiles, run_monte_carlo, resumen_backtest

CAPITAL    = 1_000_000
SEED       = 42
N_CLIENTES = 500
START      = "2024-01-01"
END        = "2024-12-31"
VALORES    = [0.10, 0.15, 0.20]

PERFILES = ["muy_conservador","conservador","neutro","arriesgado","muy_arriesgado"]
LABELS   = {
    "muy_conservador": "Muy conserv.",
    "conservador":     "Conservador",
    "neutro":          "Neutro",
    "arriesgado":      "Arriesgado",
    "muy_arriesgado":  "Muy arriesg.",
}

def main():
    print("Cargando datos...")
    price_returns, div_yields = load_universe()
    info        = load_stock_info()
    market_caps = info["marketCap"] if "marketCap" in info.columns else None

    resultados = {}
    for mw in VALORES:
        print(f"\n  max_weight = {mw:.0%} ...")
        res, mu_cache, tickers = run_all_profiles(
            price_returns, div_yields,
            capital_inicial=CAPITAL,
            eval_start=START, eval_end=END,
            market_caps=market_caps,
            lam=None, full_invest=True,
            max_weight=mw,
            seed=SEED, return_cache=True, verbose=False,
        )
        mc = run_monte_carlo(
            price_returns, div_yields,
            n_clientes=N_CLIENTES,
            mu_bl_cache=mu_cache, tickers=tickers,
            capital_inicial=CAPITAL,
            eval_start=START, eval_end=END,
            max_weight=mw, full_invest=True,
            seed_base=0, verbose=False,
        )
        resultados[mw] = (res, mc)

    # Imprimir tabla por KPI
    sep = "=" * 100
    for kpi, titulo in [
        ("retorno_anual", "RETORNO ANUAL"),
        ("max_drawdown",  "MAX DRAWDOWN"),
        ("aband",         "TASA ABANDONO"),
        ("acept",         "TASA ACEPTACION"),
        ("comision",      "COMISION (% capital)"),
    ]:
        print(f"\n{sep}")
        print(f"{titulo}  —  Sensibilidad max_weight  (año 2024)")
        print(sep)
        print(f"{'Perfil':<16} {'max_w=10%':>12} {'max_w=15%':>12} {'max_w=20%':>12}  {'Mejor':>10}")
        print("-" * 65)

        for perfil in PERFILES:
            vals = {}
            for mw in VALORES:
                res, mc = resultados[mw]
                df  = res.get(perfil, pd.DataFrame())
                mcp = mc.get(perfil, pd.DataFrame())
                r   = resumen_backtest(df) if not df.empty else {}

                if kpi == "retorno_anual":
                    v = r.get("retorno_anual", 0)
                    vals[mw] = (v, f"{v:+.1%}")
                elif kpi == "max_drawdown":
                    v = r.get("max_drawdown", 0)
                    vals[mw] = (v, f"{v:.1%}")
                elif kpi == "aband":
                    v = mcp["abandona"].mean() if not mcp.empty else 0
                    vals[mw] = (v, f"{v:.1%}")
                elif kpi == "acept":
                    v = df["rebalanceo"].mean() if not df.empty else 0
                    vals[mw] = (v, f"{v:.1%}")
                elif kpi == "comision":
                    v = r.get("comision_total", 0) / CAPITAL
                    vals[mw] = (v, f"{v:.2%}")

            # mejor: max retorno, min drawdown/abandono/comision, max acept
            if kpi in ("retorno_anual", "acept"):
                mejor_mw = max(vals, key=lambda k: vals[k][0])
            else:
                mejor_mw = min(vals, key=lambda k: vals[k][0])

            print(f"{LABELS[perfil]:<16} "
                  f"{vals[0.10][1]:>12} {vals[0.15][1]:>12} {vals[0.20][1]:>12}  "
                  f"{'max_w='+f'{mejor_mw:.0%}':>10}")

if __name__ == "__main__":
    main()
