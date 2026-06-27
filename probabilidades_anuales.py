"""
Probabilidades anuales de aceptacion y abandono por perfil.

Cada año se corre como backtest independiente. Para cada perfil se reporta:
  - prob_aceptacion_prom: promedio mensual de P2.
  - aceptacion_observada: fraccion de meses en que efectivamente acepto.
  - prob_abandono_prom: promedio mensual de P1.
"""
from pathlib import Path

import pandas as pd

import data.loader as loader
from backtesting.runner import run_all_profiles


CAPITAL = 1_000_000
START_YEAR = 2019
SEED = 42
FULL_INVEST = True

CASOS = {
    "lambda_0": 0.0,
    "lambda_1": 1.0,
}


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


def _year_end_for_data(year: int, last_date: pd.Timestamp) -> str:
    if year == last_date.year:
        return last_date.strftime("%Y-%m-%d")
    return f"{year}-12-31"


def _format_prob_table(tabla: pd.DataFrame, value_col: str) -> str:
    pivot = tabla.pivot(index="perfil", columns="anio", values=value_col)
    return pivot.to_string(formatters={
        col: (lambda x: "" if pd.isna(x) else f"{x:.1%}")
        for col in pivot.columns
    })


def correr_caso(nombre: str, lam, price_returns, div_yields, market_caps, end_year: int):
    print(f"\n{'=' * 78}")
    print(nombre)
    print(f"{'=' * 78}")

    registros = []
    for year in range(START_YEAR, end_year + 1):
        eval_start = f"{year}-01-01"
        eval_end = _year_end_for_data(year, price_returns.index[-1])

        print(f"  Año {year}: {eval_start} -> {eval_end}")
        resultados = run_all_profiles(
            price_returns,
            div_yields,
            capital_inicial=CAPITAL,
            eval_start=eval_start,
            eval_end=eval_end,
            market_caps=market_caps,
            lam=lam,
            full_invest=FULL_INVEST,
            seed=SEED,
            verbose=False,
        )

        for perfil, df in resultados.items():
            prob_aceptacion = float(df["prob_p2"].mean()) if len(df) else pd.NA
            registros.append({
                "caso": nombre,
                "anio": year,
                "perfil": perfil,
                "prob_aceptacion_prom": prob_aceptacion,
                "aceptacion_observada": float(df["rebalanceo"].mean()) if len(df) else pd.NA,
                "prob_abandono_prom": float(df["prob_p1"].mean()) if len(df) else pd.NA,
                "abandono_observado": bool(df["abandona"].any()) if len(df) else False,
                "meses_activos": int(len(df)),
            })

    tabla = pd.DataFrame(registros)
    print("\n-- Probabilidad promedio de aceptacion -------------------------")
    print(_format_prob_table(tabla, "prob_aceptacion_prom"))
    print("\n-- Probabilidad promedio de abandono ---------------------------")
    print(_format_prob_table(tabla, "prob_abandono_prom"))
    return tabla


def main():
    print("Cargando datos...")
    price_returns, div_yields = _load_universe_with_fallback()
    market_caps = _load_market_caps()

    if price_returns.empty:
        raise FileNotFoundError(
            "No se encontraron retornos de acciones. Revisa DATA_DIR en config.py."
        )

    last_date = price_returns.index[-1]
    end_year = last_date.year

    print(f"  Tickers cargados : {price_returns.shape[1]}")
    print(f"  Periodo de datos : {price_returns.index[0].date()} -> {last_date.date()}")
    print(f"  Años evaluados   : {START_YEAR} -> {end_year}")
    if last_date.month < 12:
        print(f"  Nota: {end_year} es parcial; datos hasta {last_date.date()}.")

    out_dir = Path("resultados_diagnostico")
    out_dir.mkdir(exist_ok=True)

    tablas = []
    for nombre, lam in CASOS.items():
        tabla = correr_caso(nombre, lam, price_returns, div_yields, market_caps, end_year)
        tabla.to_csv(out_dir / f"{nombre}_probabilidades_anuales.csv", index=False)
        tablas.append(tabla)

    combinado = pd.concat(tablas, ignore_index=True)
    combinado.to_csv(out_dir / "probabilidades_anuales.csv", index=False)
    print(f"\nResultados guardados en {out_dir}/")


if __name__ == "__main__":
    main()
