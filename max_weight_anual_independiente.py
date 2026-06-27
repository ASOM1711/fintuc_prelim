"""
Sensibilidad anual independiente para el peso maximo por accion.

Compara max_weight = 10%, 15%, 20% reiniciando el portafolio cada año.
Esto ayuda a justificar si 15% es un punto medio razonable entre
diversificacion y flexibilidad.
"""
from pathlib import Path

import pandas as pd

import data.loader as loader
from backtesting.runner import run_all_profiles, resumen_backtest


CAPITAL = 1_000_000
START_YEAR = 2019
SEED = 42
FULL_INVEST = True
LAMBDA = 0.0
MAX_WEIGHTS = [0.10, 0.15, 0.20]


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


def _format_pivot(tabla: pd.DataFrame, value_col: str, percent: bool = True) -> str:
    pivot = tabla.pivot_table(
        index=["anio", "perfil"],
        columns="max_weight",
        values=value_col,
        aggfunc="first",
    )
    if percent:
        return pivot.to_string(formatters={
            col: (lambda x: "" if pd.isna(x) else f"{x:.1%}")
            for col in pivot.columns
        })
    return pivot.to_string()


def _format_resumen(tabla: pd.DataFrame, value_col: str) -> str:
    pivot = tabla.pivot(index="perfil", columns="max_weight", values=value_col)
    return pivot.to_string(formatters={
        col: (lambda x: "" if pd.isna(x) else f"{x:.1%}")
        for col in pivot.columns
    })


def correr(price_returns, div_yields, market_caps, end_year: int) -> pd.DataFrame:
    registros = []

    for max_weight in MAX_WEIGHTS:
        print(f"\n{'=' * 78}")
        print(f"max_weight = {max_weight:.0%}")
        print(f"{'=' * 78}")

        for year in range(START_YEAR, end_year + 1):
            eval_start = f"{year}-01-01"
            eval_end = _year_end_for_data(year, price_returns.index[-1])
            print(f"  Año {year}: {eval_start} -> {eval_end}", flush=True)

            resultados = run_all_profiles(
                price_returns,
                div_yields,
                capital_inicial=CAPITAL,
                eval_start=eval_start,
                eval_end=eval_end,
                market_caps=market_caps,
                lam=LAMBDA,
                max_weight=max_weight,
                full_invest=FULL_INVEST,
                seed=SEED,
                verbose=False,
            )

            for perfil, df in resultados.items():
                r = resumen_backtest(df)
                registros.append({
                    "max_weight": max_weight,
                    "anio": year,
                    "periodo_inicio": eval_start,
                    "periodo_fin": eval_end,
                    "perfil": perfil,
                    "prob_aceptacion_prom": float(df["prob_p2"].mean()) if len(df) else pd.NA,
                    "prob_abandono_prom": float(df["prob_p1"].mean()) if len(df) else pd.NA,
                    "aceptacion_observada": float(df["rebalanceo"].mean()) if len(df) else pd.NA,
                    **r,
                })

            out_dir = Path("resultados_diagnostico")
            out_dir.mkdir(exist_ok=True)
            pd.DataFrame(registros).to_csv(
                out_dir / "max_weight_anual_independiente_parcial.csv",
                index=False,
            )

    return pd.DataFrame(registros)


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
    print(f"  Lambda usado     : {LAMBDA:g}")
    if last_date.month < 12:
        print(f"  Nota: {end_year} es parcial; datos hasta {last_date.date()}.")

    tabla = correr(price_returns, div_yields, market_caps, end_year)

    out_dir = Path("resultados_diagnostico")
    out_dir.mkdir(exist_ok=True)
    tabla.to_csv(out_dir / "max_weight_anual_independiente.csv", index=False)

    resumen = tabla.groupby(["max_weight", "perfil"]).agg(
        retorno_promedio=("retorno_anual", "mean"),
        retorno_mediano=("retorno_anual", "median"),
        drawdown_promedio=("max_drawdown", "mean"),
        abandono_promedio=("abandono", "mean"),
        aceptacion_promedio=("prob_aceptacion_prom", "mean"),
        caja_promedio=("caja_chica_promedio", "mean"),
        meses_activos_promedio=("meses_activos", "mean"),
    ).reset_index()
    resumen.to_csv(out_dir / "max_weight_resumen_por_perfil.csv", index=False)

    print("\n-- Retorno anual por año/perfil -------------------------------")
    print(_format_pivot(tabla, "retorno_anual"))
    print("\n-- Max drawdown por año/perfil --------------------------------")
    print(_format_pivot(tabla, "max_drawdown"))
    print("\n-- Resumen promedio por perfil --------------------------------")
    print("\nRetorno promedio:")
    print(_format_resumen(resumen, "retorno_promedio"))
    print("\nDrawdown promedio:")
    print(_format_resumen(resumen, "drawdown_promedio"))
    print("\nAbandono promedio:")
    print(_format_resumen(resumen, "abandono_promedio"))

    print(f"\nResultados guardados en {out_dir}/")


if __name__ == "__main__":
    main()
