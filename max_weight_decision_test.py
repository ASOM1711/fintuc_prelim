"""
Test para elegir max_weight combinando retorno, riesgo y error predictivo.

Evalua max_weight = 5%, 10%, 15%, 20%, 25%.
Usa lambda=0 y full_invest=True, consistente con la idea de quitar la
penalizacion redundante por varianza.
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
SEED = 42
LAM = 0.0
FULL_INVEST = True
MAX_WEIGHTS = [0.05, 0.10, 0.15, 0.20, 0.25]


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


def _run_metricas_anuales(price_returns, div_yields, market_caps, end_year: int) -> pd.DataFrame:
    registros = []

    for max_weight in MAX_WEIGHTS:
        print(f"\nMetricas anuales max_weight={max_weight:.0%}")
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
                lam=LAM,
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
                    "perfil": perfil,
                    "prob_aceptacion_prom": float(df["prob_p2"].mean()) if len(df) else pd.NA,
                    "prob_abandono_prom": float(df["prob_p1"].mean()) if len(df) else pd.NA,
                    **r,
                })

    return pd.DataFrame(registros)


def _run_error_predictivo(price_returns, market_caps) -> pd.DataFrame:
    registros = []
    for max_weight in MAX_WEIGHTS:
        print(f"\nError predictivo max_weight={max_weight:.0%}", flush=True)
        df_error = run_profile_prediction_error(
            price_returns,
            market_caps=market_caps,
            eval_start=f"{START_YEAR}-01-01",
            eval_end="2024-12-31",
            lam=LAM,
            max_weight=max_weight,
            full_invest=FULL_INVEST,
        )
        resumen = resumen_prediction_error_por_perfil(df_error).reset_index()
        resumen["max_weight"] = max_weight
        registros.append(resumen)

    return pd.concat(registros, ignore_index=True)


def _score_decision(resumen: pd.DataFrame) -> pd.DataFrame:
    df = resumen.copy()
    # Score simple: recompensa retorno; penaliza drawdown, error y abandono.
    df["score"] = (
        df["retorno_promedio"]
        - 0.50 * df["drawdown_promedio"]
        - 0.50 * df["mae"]
        - 0.30 * df["abandono_promedio"]
    )
    return df.sort_values(["perfil", "score"], ascending=[True, False])


def _format_resumen(tabla: pd.DataFrame, value_col: str) -> str:
    pivot = tabla.pivot(index="perfil", columns="max_weight", values=value_col)
    return pivot.to_string(formatters={
        col: (lambda x: "" if pd.isna(x) else f"{x:.1%}")
        for col in pivot.columns
    })


def main():
    print("Cargando datos...")
    price_returns, div_yields = _load_universe_with_fallback()
    market_caps = _load_market_caps()

    if price_returns.empty:
        raise FileNotFoundError(
            "No se encontraron retornos de acciones. Revisa DATA_DIR en config.py."
        )

    last_date = price_returns.index[-1]
    end_year = last_date.year if last_date.month == 12 else last_date.year - 1
    print(f"  Tickers cargados : {price_returns.shape[1]}")
    print(f"  Periodo de datos : {price_returns.index[0].date()} -> {last_date.date()}")
    print(f"  Años metricas    : {START_YEAR} -> {end_year}")
    print("  Error predictivo : 2019 -> 2024")

    out_dir = Path("resultados_diagnostico")
    out_dir.mkdir(exist_ok=True)

    metricas = _run_metricas_anuales(price_returns, div_yields, market_caps, end_year)
    metricas.to_csv(out_dir / "max_weight_decision_metricas_anuales.csv", index=False)

    error = _run_error_predictivo(price_returns, market_caps)
    error.to_csv(out_dir / "max_weight_decision_error_predictivo.csv", index=False)

    resumen_metricas = metricas.groupby(["max_weight", "perfil"]).agg(
        retorno_promedio=("retorno_anual", "mean"),
        retorno_mediano=("retorno_anual", "median"),
        drawdown_promedio=("max_drawdown", "mean"),
        abandono_promedio=("abandono", "mean"),
        prob_abandono_promedio=("prob_abandono_prom", "mean"),
        prob_aceptacion_promedio=("prob_aceptacion_prom", "mean"),
        meses_activos_promedio=("meses_activos", "mean"),
    ).reset_index()

    resumen = resumen_metricas.merge(
        error[["max_weight", "perfil", "mae", "rmse", "bias", "hit_rate_signo", "n_activos_promedio"]],
        on=["max_weight", "perfil"],
        how="left",
    )
    resumen = _score_decision(resumen)
    resumen.to_csv(out_dir / "max_weight_decision_resumen.csv", index=False)

    print("\n-- Retorno promedio --------------------------------------------")
    print(_format_resumen(resumen, "retorno_promedio"))
    print("\n-- Drawdown promedio -------------------------------------------")
    print(_format_resumen(resumen, "drawdown_promedio"))
    print("\n-- MAE error predictivo ----------------------------------------")
    print(_format_resumen(resumen, "mae"))
    print("\n-- Abandono promedio -------------------------------------------")
    print(_format_resumen(resumen, "abandono_promedio"))
    print("\n-- Score decision ----------------------------------------------")
    print(_format_resumen(resumen, "score"))

    mejores = resumen.sort_values("score", ascending=False).groupby("perfil").head(1)
    print("\n-- Mejor max_weight por score ----------------------------------")
    for _, row in mejores.sort_values("perfil").iterrows():
        print(f"  {row['perfil']:<20} max_weight={row['max_weight']:.0%}  score={row['score']:.2%}")

    print(f"\nResultados guardados en {out_dir}/")


if __name__ == "__main__":
    main()
