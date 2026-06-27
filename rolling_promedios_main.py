"""
Diagnostico rolling para revisar promedios por perfil.

Compara:
  1. Modelo actual con lambda calibrado por perfil (lam=None).
  2. Modelo con lambda fijo en 1.0.

Para cada caso imprime:
  - Resultado de una trayectoria rolling.
  - Retornos por año calendario.
  - Promedios Monte Carlo opcionales, usando los mismos pesos/mu_BL cache.
"""
from pathlib import Path

import pandas as pd

import data.loader as loader
from backtesting.runner import (
    run_all_profiles,
    run_monte_carlo,
    resumen_backtest,
    resumen_monte_carlo,
)


CAPITAL = 1_000_000
EVAL_START = "2019-01-01"
EVAL_END = "2024-12-31"
SEED = 42
N_CLIENTES = 200
RUN_MONTE_CARLO = False


def _load_universe_with_fallback():
    price_returns, div_yields = loader.load_universe()
    if not price_returns.empty:
        return price_returns, div_yields

    root = Path.cwd()
    candidates = [
        root.parent.parent / "Capstone" / "universo_300_acciones" / "universo_300_acciones",
    ]
    for candidate in candidates:
        if list(candidate.glob("stock_return_*.csv")):
            print(f"  DATA_DIR no encontro datos; usando fallback: {candidate}")
            loader.DATA_DIR = candidate
            return loader.load_universe()

    return price_returns, div_yields


def _load_market_caps():
    try:
        info = loader.load_stock_info()
    except FileNotFoundError:
        print("  stocks_info.txt no encontrado; usando pesos iguales en Black-Litterman.")
        return None

    return info["marketCap"] if "marketCap" in info.columns else None


def _tabla_backtest(resultados: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for perfil, df in resultados.items():
        r = resumen_backtest(df)
        mensual = df["valor"].pct_change().dropna()
        rows.append({
            "perfil": perfil,
            "retorno_anual": r["retorno_anual"],
            "retorno_mensual_prom": mensual.mean() if len(mensual) else 0.0,
            "vol_mensual": mensual.std() if len(mensual) else 0.0,
            "max_drawdown": r["max_drawdown"],
            "meses_activos": r["meses_activos"],
            "abandono": r["abandono"],
            "aceptacion_rebalanceo": df["rebalanceo"].mean(),
            "caja_chica_promedio": r["caja_chica_promedio"],
        })
    return pd.DataFrame(rows).set_index("perfil")


def _tabla_retornos_por_anio(resultados: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    anios = range(pd.Timestamp(EVAL_START).year, pd.Timestamp(EVAL_END).year + 1)

    for perfil, df in resultados.items():
        valor_inicio = CAPITAL
        abandono_previo = False

        for anio in anios:
            df_anio = df[df.index.year == anio]

            if df_anio.empty or abandono_previo:
                rows.append({
                    "perfil": perfil,
                    "anio": anio,
                    "retorno": pd.NA,
                    "meses": 0,
                    "abandono_en_anio": False,
                    "anio_completo": False,
                })
                continue

            valor_fin = float(df_anio["valor"].iloc[-1])
            abandono_en_anio = bool(df_anio["abandona"].any())
            meses = int(len(df_anio))

            rows.append({
                "perfil": perfil,
                "anio": anio,
                "retorno": valor_fin / valor_inicio - 1,
                "meses": meses,
                "abandono_en_anio": abandono_en_anio,
                "anio_completo": meses == 12 and not abandono_en_anio,
            })

            valor_inicio = valor_fin
            abandono_previo = abandono_en_anio

    return pd.DataFrame(rows)


def _format_pct_table(df: pd.DataFrame) -> str:
    pct_cols = [
        "retorno_anual",
        "retorno_mensual_prom",
        "vol_mensual",
        "max_drawdown",
        "aceptacion_rebalanceo",
        "caja_chica_promedio",
        "tasa_abandono",
        "retorno_prom",
        "retorno_std",
        "retorno_si_completa",
        "frec_rebalanceo_prom",
    ]
    formatters = {
        col: (lambda x: "" if pd.isna(x) else f"{x:.1%}")
        for col in pct_cols
        if col in df.columns
    }
    return df.to_string(formatters=formatters)


def _format_yearly_returns(tabla_anual: pd.DataFrame) -> str:
    pivot = tabla_anual.pivot(index="perfil", columns="anio", values="retorno")
    return pivot.to_string(formatters={
        col: (lambda x: "" if pd.isna(x) else f"{x:.1%}")
        for col in pivot.columns
    })


def correr_caso(nombre: str, price_returns, div_yields, market_caps, lam):
    print(f"\n{'=' * 78}")
    print(nombre)
    print(f"{'=' * 78}")

    resultados, mu_bl_cache, tickers = run_all_profiles(
        price_returns,
        div_yields,
        capital_inicial=CAPITAL,
        eval_start=EVAL_START,
        eval_end=EVAL_END,
        market_caps=market_caps,
        lam=lam,
        full_invest=True,
        seed=SEED,
        return_cache=True,
        verbose=True,
    )

    tabla_bt = _tabla_backtest(resultados)
    tabla_anual = _tabla_retornos_por_anio(resultados)

    print("\n-- Trayectoria rolling unica -----------------------------------")
    print(_format_pct_table(tabla_bt))

    print("\n-- Retornos por año calendario ---------------------------------")
    print(_format_yearly_returns(tabla_anual))

    tabla_mc = pd.DataFrame()
    if RUN_MONTE_CARLO:
        mc = run_monte_carlo(
            price_returns,
            div_yields,
            n_clientes=N_CLIENTES,
            mu_bl_cache=mu_bl_cache,
            tickers=tickers,
            capital_inicial=CAPITAL,
            eval_start=EVAL_START,
            eval_end=EVAL_END,
            lam=1.0 if lam is None else lam,
            full_invest=True,
            seed_base=0,
            verbose=False,
        )
        tabla_mc = resumen_monte_carlo(mc)
        print(f"\n-- Promedios Monte Carlo ({N_CLIENTES} clientes por perfil) -----")
        print(_format_pct_table(tabla_mc))

    return tabla_bt, tabla_anual, tabla_mc


def main():
    print("Cargando datos...")
    price_returns, div_yields = _load_universe_with_fallback()
    market_caps = _load_market_caps()

    if price_returns.empty:
        raise FileNotFoundError(
            "No se encontraron retornos de acciones. Revisa DATA_DIR en config.py."
        )

    print(f"  Tickers cargados : {price_returns.shape[1]}")
    print(f"  Periodo de datos : {price_returns.index[0].date()} -> {price_returns.index[-1].date()}")
    print(f"  Periodo test     : {EVAL_START} -> {EVAL_END}")

    out = {}
    out["lambda_calibrado"] = correr_caso(
        "MODELO ACTUAL: lambda calibrado por perfil (lam=None)",
        price_returns,
        div_yields,
        market_caps,
        lam=None,
    )
    out["lambda_1"] = correr_caso(
        "CONTROL: lambda fijo = 1.0",
        price_returns,
        div_yields,
        market_caps,
        lam=1.0,
    )

    Path("resultados_diagnostico").mkdir(exist_ok=True)
    for nombre, (tabla_bt, tabla_anual, tabla_mc) in out.items():
        tabla_bt.to_csv(f"resultados_diagnostico/{nombre}_rolling_unico.csv")
        tabla_anual.to_csv(f"resultados_diagnostico/{nombre}_retornos_por_anio.csv", index=False)
        if not tabla_mc.empty:
            tabla_mc.to_csv(f"resultados_diagnostico/{nombre}_montecarlo_promedios.csv")

    print("\nResultados guardados en resultados_diagnostico/")


if __name__ == "__main__":
    main()
