"""
Test de decision para elegir cuantos años usar en la ventana de training.

Compara ventanas rolling de 2, 3, 4, 5 y 6 años.
Mantiene fijos: lambda=0, max_weight=15%, full_invest=True.
Evalua 2019-2025 para evitar años parciales.
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
EVAL_START = "2019-01-01"
EVAL_END = "2025-12-31"
SEED = 42
LAM = 0.0
MAX_WEIGHT = 0.15
FULL_INVEST = True
TRAIN_WINDOWS = [2, 3, 4, 5, 6]


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


def _run_backtest_metrics(price_returns, div_yields, market_caps):
    rows = []
    for train_years in TRAIN_WINDOWS:
        print(f"\nBacktest rolling train_years={train_years}", flush=True)
        resultados = run_all_profiles(
            price_returns,
            div_yields,
            capital_inicial=CAPITAL,
            eval_start=EVAL_START,
            eval_end=EVAL_END,
            train_years=train_years,
            market_caps=market_caps,
            lam=LAM,
            max_weight=MAX_WEIGHT,
            full_invest=FULL_INVEST,
            seed=SEED,
            verbose=True,
        )
        for perfil, df in resultados.items():
            r = resumen_backtest(df)
            rows.append({
                "train_years": train_years,
                "perfil": perfil,
                "prob_aceptacion_prom": float(df["prob_p2"].mean()) if len(df) else pd.NA,
                "prob_abandono_prom": float(df["prob_p1"].mean()) if len(df) else pd.NA,
                **r,
            })
    return pd.DataFrame(rows)


def _run_prediction_error(price_returns, market_caps):
    rows = []
    for train_years in TRAIN_WINDOWS:
        print(f"\nError predictivo train_years={train_years}", flush=True)
        df_error = run_profile_prediction_error(
            price_returns,
            market_caps=market_caps,
            eval_start=EVAL_START,
            eval_end=EVAL_END,
            train_years=train_years,
            lam=LAM,
            max_weight=MAX_WEIGHT,
            full_invest=FULL_INVEST,
        )
        resumen = resumen_prediction_error_por_perfil(df_error).reset_index()
        resumen["train_years"] = train_years
        rows.append(resumen)
    return pd.concat(rows, ignore_index=True)


def _score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["score"] = (
        out["retorno_anual"]
        - 0.50 * out["max_drawdown"]
        - 0.50 * out["mae"]
        - 0.30 * out["abandono"].astype(float)
    )
    return out.sort_values(["perfil", "score"], ascending=[True, False])


def _format_pivot(df: pd.DataFrame, value_col: str) -> str:
    pivot = df.pivot(index="perfil", columns="train_years", values=value_col)
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

    print(f"  Tickers cargados : {price_returns.shape[1]}")
    print(f"  Periodo de datos : {price_returns.index[0].date()} -> {price_returns.index[-1].date()}")
    print(f"  Periodo test     : {EVAL_START} -> {EVAL_END}")
    print(f"  Lambda           : {LAM:g}")
    print(f"  Max weight       : {MAX_WEIGHT:.0%}")

    out_dir = Path("resultados_diagnostico")
    out_dir.mkdir(exist_ok=True)

    metrics = _run_backtest_metrics(price_returns, div_yields, market_caps)
    metrics.to_csv(out_dir / "train_window_metricas_rolling.csv", index=False)

    error = _run_prediction_error(price_returns, market_caps)
    error.to_csv(out_dir / "train_window_error_predictivo.csv", index=False)

    resumen = metrics.merge(
        error[["train_years", "perfil", "mae", "rmse", "bias", "hit_rate_signo", "n_activos_promedio"]],
        on=["train_years", "perfil"],
        how="left",
    )
    resumen = _score(resumen)
    resumen.to_csv(out_dir / "train_window_decision_resumen.csv", index=False)

    print("\n-- Retorno anual -----------------------------------------------")
    print(_format_pivot(resumen, "retorno_anual"))
    print("\n-- Max drawdown -------------------------------------------------")
    print(_format_pivot(resumen, "max_drawdown"))
    print("\n-- MAE error predictivo ----------------------------------------")
    print(_format_pivot(resumen, "mae"))
    print("\n-- Score decision ----------------------------------------------")
    print(_format_pivot(resumen, "score"))

    mejores = resumen.sort_values("score", ascending=False).groupby("perfil").head(1)
    print("\n-- Mejor ventana por score -------------------------------------")
    for _, row in mejores.sort_values("perfil").iterrows():
        print(f"  {row['perfil']:<20} train_years={int(row['train_years'])}  score={row['score']:.2%}")

    print(f"\nResultados guardados en {out_dir}/")


if __name__ == "__main__":
    main()
