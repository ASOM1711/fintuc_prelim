"""
Compara la configuracion final con y sin Black-Litterman.

Configuracion:
- train_years = 5
- max_weight = 2.5%
- lambda = 0
- full_invest = True
- backtesting anual independiente 2019-2025, excluyendo anos parciales
"""
from pathlib import Path

import pandas as pd

import data.loader as loader
from backtesting.prediction_error import (
    run_profile_prediction_error,
    resumen_prediction_error_por_perfil,
)
from backtesting.runner import run_all_profiles, resumen_backtest


CAPITAL = 1_000_000
START_YEAR = 2019
ERROR_END = "2024-12-31"
SEED = 42
LAM = 0.0
FULL_INVEST = True
TRAIN_YEARS = 5
MAX_WEIGHT = 0.025

OUT_DIR = Path("resultados_diagnostico")
BL_ANNUAL = OUT_DIR / "train5_maxweight_2_5_anual_independiente.csv"
BL_ERROR = OUT_DIR / "max_weight_decision_train5_error_predictivo.csv"
NO_BL_ANNUAL = OUT_DIR / "sin_bl_train5_maxweight_2_5_anual_independiente.csv"
NO_BL_ERROR = OUT_DIR / "sin_bl_train5_maxweight_2_5_error_predictivo.csv"
COMPARISON = OUT_DIR / "comparacion_bl_vs_sin_bl_train5_maxweight_2_5.csv"


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


def _run_no_bl_annual(price_returns, div_yields, end_year: int) -> pd.DataFrame:
    registros = []
    for year in range(START_YEAR, end_year + 1):
        eval_start = f"{year}-01-01"
        eval_end = _year_end_for_data(year, price_returns.index[-1])
        print(f"  Sin BL ano {year}: {eval_start} -> {eval_end}", flush=True)

        resultados = run_all_profiles(
            price_returns,
            div_yields,
            capital_inicial=CAPITAL,
            eval_start=eval_start,
            eval_end=eval_end,
            train_years=TRAIN_YEARS,
            market_caps=None,
            lam=LAM,
            max_weight=MAX_WEIGHT,
            full_invest=FULL_INVEST,
            seed=SEED,
            verbose=False,
            use_bl=False,
        )

        for perfil, df in resultados.items():
            r = resumen_backtest(df)
            registros.append({
                "modelo": "sin_bl",
                "train_years": TRAIN_YEARS,
                "max_weight": MAX_WEIGHT,
                "anio": year,
                "perfil": perfil,
                "prob_aceptacion_prom": float(df["prob_p2"].mean()) if len(df) else pd.NA,
                "prob_abandono_prom": float(df["prob_p1"].mean()) if len(df) else pd.NA,
                **r,
            })

        pd.DataFrame(registros).to_csv(NO_BL_ANNUAL, index=False)

    return pd.DataFrame(registros)


def _run_no_bl_error(price_returns) -> pd.DataFrame:
    print("  Calculando error predictivo sin BL...", flush=True)
    df_error = run_profile_prediction_error(
        price_returns,
        market_caps=None,
        eval_start=f"{START_YEAR}-01-01",
        eval_end=ERROR_END,
        train_years=TRAIN_YEARS,
        lam=LAM,
        max_weight=MAX_WEIGHT,
        full_invest=FULL_INVEST,
        use_bl=False,
    )
    resumen = resumen_prediction_error_por_perfil(df_error).reset_index()
    resumen["modelo"] = "sin_bl"
    resumen["train_years"] = TRAIN_YEARS
    resumen["max_weight"] = MAX_WEIGHT
    resumen.to_csv(NO_BL_ERROR, index=False)
    return resumen


def _load_bl_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    annual = pd.read_csv(BL_ANNUAL)
    annual = annual[(annual["train_years"] == TRAIN_YEARS) & (annual["max_weight"].round(6) == MAX_WEIGHT)]
    annual = annual.copy()
    annual["modelo"] = "con_bl"

    error = pd.read_csv(BL_ERROR)
    error = error[(error["train_years"] == TRAIN_YEARS) & (error["max_weight"].round(6) == MAX_WEIGHT)]
    error = error.copy()
    error["modelo"] = "con_bl"
    return annual, error


def _build_comparison(bl_annual, bl_error, no_bl_annual, no_bl_error) -> pd.DataFrame:
    annual = pd.concat([bl_annual, no_bl_annual], ignore_index=True)
    error = pd.concat([bl_error, no_bl_error], ignore_index=True)

    annual_summary = annual.groupby(["modelo", "perfil"]).agg(
        retorno_promedio=("retorno_anual", "mean"),
        drawdown_promedio=("max_drawdown", "mean"),
        tasa_aceptacion=("frecuencia_rebalanceo", "mean"),
        tasa_abandono=("abandono", "mean"),
        prob_aceptacion_promedio=("prob_aceptacion_prom", "mean"),
        prob_abandono_promedio=("prob_abandono_prom", "mean"),
        meses_activos_promedio=("meses_activos", "mean"),
    ).reset_index()

    error_cols = [
        "modelo",
        "perfil",
        "mae",
        "rmse",
        "bias",
        "hit_rate_signo",
        "n_activos_promedio",
    ]
    comparison = annual_summary.merge(
        error[error_cols],
        on=["modelo", "perfil"],
        how="left",
    )
    comparison.to_csv(COMPARISON, index=False)
    return comparison


def _print_table(df: pd.DataFrame, value_col: str, title: str) -> None:
    pivot = df.pivot(index="perfil", columns="modelo", values=value_col)
    order = ["muy_conservador", "conservador", "neutro", "arriesgado", "muy_arriesgado"]
    pivot = pivot.reindex(order)
    print(f"\n-- {title} ----------------------------------------------")
    print(pivot.to_string(formatters={
        col: (lambda x: "" if pd.isna(x) else f"{x:.1%}")
        for col in pivot.columns
    }))


def main():
    OUT_DIR.mkdir(exist_ok=True)

    print("Cargando resultados con BL ya calculados...")
    bl_annual, bl_error = _load_bl_results()

    print("Cargando datos...")
    price_returns, div_yields = _load_universe_with_fallback()
    _load_market_caps()

    if price_returns.empty:
        raise FileNotFoundError("No se encontraron retornos de acciones.")

    last_date = price_returns.index[-1]
    end_year = last_date.year if last_date.month == 12 else last_date.year - 1
    print(f"  Tickers cargados : {price_returns.shape[1]}")
    print(f"  Periodo de datos : {price_returns.index[0].date()} -> {last_date.date()}")
    print(f"  Anos metricas    : {START_YEAR} -> {end_year}")
    print(f"  Error predictivo : {START_YEAR} -> 2024")

    no_bl_annual = _run_no_bl_annual(price_returns, div_yields, end_year)
    no_bl_error = _run_no_bl_error(price_returns)
    comparison = _build_comparison(bl_annual, bl_error, no_bl_annual, no_bl_error)

    _print_table(comparison, "retorno_promedio", "Retorno promedio")
    _print_table(comparison, "tasa_aceptacion", "Tasa de aceptacion observada")
    _print_table(comparison, "tasa_abandono", "Tasa de abandono observada")
    _print_table(comparison, "mae", "MAE error predictivo")
    _print_table(comparison, "rmse", "RMSE error predictivo")

    print(f"\nResultados guardados:")
    print(f"  {NO_BL_ANNUAL}")
    print(f"  {NO_BL_ERROR}")
    print(f"  {COMPARISON}")


if __name__ == "__main__":
    main()
