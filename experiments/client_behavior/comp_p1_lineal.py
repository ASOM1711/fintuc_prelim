"""
Compara P1 sigmoid (actual) vs P1 lineal (nueva) para 2022, 2023, 2024.
"""
import pandas as pd
from data.loader import load_universe, load_stock_info
from backtesting.runner import run_all_profiles, run_monte_carlo, resumen_backtest

CAPITAL    = 1_000_000
SEED       = 42
N_CLIENTES = 500
ANIOS      = [2022, 2023, 2024]

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

    for año in ANIOS:
        start, end = f"{año}-01-01", f"{año}-12-31"
        print(f"\nProcesando {año}...")

        datos = {}
        for modo, lineal in [("Sigmoid", False), ("Lineal", True)]:
            res, mu_cache, tickers = run_all_profiles(
                price_returns, div_yields,
                capital_inicial=CAPITAL,
                eval_start=start, eval_end=end,
                market_caps=market_caps,
                lam=None, full_invest=True,
                p1_lineal=lineal,
                seed=SEED, return_cache=True, verbose=False,
            )
            mc = run_monte_carlo(
                price_returns, div_yields,
                n_clientes=N_CLIENTES,
                mu_bl_cache=mu_cache, tickers=tickers,
                capital_inicial=CAPITAL,
                eval_start=start, eval_end=end,
                full_invest=True, p1_lineal=lineal,
                seed_base=0, verbose=False,
            )
            datos[modo] = (res, mc)

        sep = "=" * 95
        print(f"\n{sep}")
        print(f"AÑO {año}  —  Comparacion P1 Sigmoid vs P1 Lineal")
        print(sep)
        print(f"{'Perfil':<16}  {'——— Sigmoid ———':^35}  {'——— Lineal ———':^35}")
        print(f"{'':16}  {'Retorno':>8} {'Aband':>8} {'Acept':>8} {'Comis%':>8}  "
              f"{'Retorno':>8} {'Aband':>8} {'Acept':>8} {'Comis%':>8}")
        print("-" * 95)

        for perfil in PERFILES:
            fila = f"{LABELS[perfil]:<16}  "
            for modo in ["Sigmoid", "Lineal"]:
                res, mc = datos[modo]
                df  = res.get(perfil, pd.DataFrame())
                mcp = mc.get(perfil, pd.DataFrame())
                r   = resumen_backtest(df) if not df.empty else {}

                ret   = f"{r['retorno_anual']:+.1%}"         if r else "—"
                aband = f"{mcp['abandona'].mean():.1%}"      if not mcp.empty else "—"
                acept = f"{df['rebalanceo'].mean():.1%}"     if not df.empty else "—"
                com   = f"{r['comision_total']/CAPITAL:.2%}" if r else "—"
                fila += f"{ret:>8} {aband:>8} {acept:>8} {com:>8}  "
            print(fila)

if __name__ == "__main__":
    main()
