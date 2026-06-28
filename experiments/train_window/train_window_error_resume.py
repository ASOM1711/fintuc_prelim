"""Calcula/reanuda el error predictivo para ventanas de training."""
from pathlib import Path

import pandas as pd

import data.loader as loader
from backtesting.prediction_error import (
    run_profile_prediction_error,
    resumen_prediction_error_por_perfil,
)


EVAL_START = "2019-01-01"
EVAL_END = "2025-12-31"
LAM = 0.0
MAX_WEIGHT = 0.15
FULL_INVEST = True
TRAIN_WINDOWS = [2, 3, 4, 5, 6]
OUT_DIR = Path("resultados_diagnostico")
ERROR_PATH = OUT_DIR / "train_window_error_predictivo.csv"


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


def main():
    OUT_DIR.mkdir(exist_ok=True)
    price_returns, _ = _load_universe_with_fallback()
    market_caps = _load_market_caps()

    existing = pd.DataFrame()
    if ERROR_PATH.exists():
        existing = pd.read_csv(ERROR_PATH)
    done = set(existing["train_years"].unique()) if not existing.empty else set()

    rows = [existing] if not existing.empty else []
    for train_years in TRAIN_WINDOWS:
        if train_years in done:
            print(f"train_years={train_years} ya calculado.")
            continue

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
        pd.concat(rows, ignore_index=True).to_csv(ERROR_PATH, index=False)
        print(f"train_years={train_years} guardado en {ERROR_PATH}", flush=True)

    print(ERROR_PATH)


if __name__ == "__main__":
    main()
