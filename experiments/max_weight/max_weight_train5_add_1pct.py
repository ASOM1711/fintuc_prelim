"""Agrega max_weight=1% al test train_years=5 existente."""
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
SEED = 42
LAM = 0.0
FULL_INVEST = True
TRAIN_YEARS = 5
MAX_WEIGHT = 0.01
OUT_DIR = Path("resultados_diagnostico")


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


def main():
    OUT_DIR.mkdir(exist_ok=True)

    print("Cargando datos...")
    price_returns, div_yields = _load_universe_with_fallback()
    market_caps = _load_market_caps()

    last_date = price_returns.index[-1]
    end_year = last_date.year if last_date.month == 12 else last_date.year - 1

    registros = []
    print(f"\nMetricas anuales max_weight={MAX_WEIGHT:.0%}, train_years={TRAIN_YEARS}")
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
            train_years=TRAIN_YEARS,
            market_caps=market_caps,
            lam=LAM,
            max_weight=MAX_WEIGHT,
            full_invest=FULL_INVEST,
            seed=SEED,
            verbose=False,
        )

        for perfil, df in resultados.items():
            r = resumen_backtest(df)
            registros.append({
                "train_years": TRAIN_YEARS,
                "max_weight": MAX_WEIGHT,
                "anio": year,
                "perfil": perfil,
                "prob_aceptacion_prom": float(df["prob_p2"].mean()) if len(df) else pd.NA,
                "prob_abandono_prom": float(df["prob_p1"].mean()) if len(df) else pd.NA,
                **r,
            })

    metricas_1 = pd.DataFrame(registros)
    metricas_1.to_csv(OUT_DIR / "max_weight_decision_train5_metricas_1pct.csv", index=False)

    print(f"\nError predictivo max_weight={MAX_WEIGHT:.0%}, train_years={TRAIN_YEARS}")
    df_error = run_profile_prediction_error(
        price_returns,
        market_caps=market_caps,
        eval_start=f"{START_YEAR}-01-01",
        eval_end="2024-12-31",
        train_years=TRAIN_YEARS,
        lam=LAM,
        max_weight=MAX_WEIGHT,
        full_invest=FULL_INVEST,
    )
    error_1 = resumen_prediction_error_por_perfil(df_error).reset_index()
    error_1["train_years"] = TRAIN_YEARS
    error_1["max_weight"] = MAX_WEIGHT
    error_1.to_csv(OUT_DIR / "max_weight_decision_train5_error_1pct.csv", index=False)

    metricas_path = OUT_DIR / "max_weight_decision_train5_metricas_anuales.csv"
    error_path = OUT_DIR / "max_weight_decision_train5_error_predictivo.csv"

    metricas = pd.read_csv(metricas_path)
    metricas = metricas[metricas["max_weight"] != MAX_WEIGHT]
    metricas = pd.concat([metricas, metricas_1], ignore_index=True)
    metricas.to_csv(metricas_path, index=False)

    error = pd.read_csv(error_path)
    error = error[error["max_weight"] != MAX_WEIGHT]
    error = pd.concat([error, error_1], ignore_index=True)
    error.to_csv(error_path, index=False)

    resumen_metricas = metricas.groupby(["train_years", "max_weight", "perfil"]).agg(
        retorno_promedio=("retorno_anual", "mean"),
        retorno_mediano=("retorno_anual", "median"),
        drawdown_promedio=("max_drawdown", "mean"),
        abandono_promedio=("abandono", "mean"),
        prob_abandono_promedio=("prob_abandono_prom", "mean"),
        prob_aceptacion_promedio=("prob_aceptacion_prom", "mean"),
        meses_activos_promedio=("meses_activos", "mean"),
    ).reset_index()

    resumen = resumen_metricas.merge(
        error[[
            "train_years", "max_weight", "perfil", "mae", "rmse", "bias",
            "hit_rate_signo", "n_activos_promedio",
        ]],
        on=["train_years", "max_weight", "perfil"],
        how="left",
    )
    resumen["score"] = (
        resumen["retorno_promedio"]
        - 0.50 * resumen["drawdown_promedio"]
        - 0.50 * resumen["mae"]
        - 0.30 * resumen["abandono_promedio"]
    )
    resumen = resumen.sort_values(["perfil", "score"], ascending=[True, False])
    resumen.to_csv(OUT_DIR / "max_weight_decision_train5_resumen.csv", index=False)

    print("\n-- Resumen actualizado -----------------------------------------")
    for col in ["retorno_promedio", "drawdown_promedio", "mae", "score"]:
        print(f"\n{col}")
        pivot = resumen.pivot(index="perfil", columns="max_weight", values=col)
        print(pivot.to_string(formatters={
            c: (lambda x: "" if pd.isna(x) else f"{x:.1%}")
            for c in pivot.columns
        }))


if __name__ == "__main__":
    main()
