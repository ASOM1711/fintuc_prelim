"""
Testing del error de prediccion de Black-Litterman.

Uso:
    python prediction_error_main.py
"""
import os
from pathlib import Path

import pandas as pd

import data.loader as loader
from backtesting.prediction_error import (
    run_profile_prediction_error,
    run_prediction_error,
    resumen_prediction_error_por_perfil,
    resumen_prediction_error,
    resumen_prediction_error_mensual,
)


CARPETA = "resultados_prediccion"


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
    print("Cargando datos...")
    price_returns, _ = _load_universe_with_fallback()
    market_caps = _load_market_caps()

    if price_returns.empty:
        raise FileNotFoundError(
            "No se encontraron retornos de acciones. Revisa DATA_DIR en config.py."
        )

    print(f"  Tickers cargados : {price_returns.shape[1]}")
    print(f"  Periodo de datos : {price_returns.index[0].date()} -> {price_returns.index[-1].date()}")

    print("\nEvaluando error predictivo mensual 2019-2024...")
    df_error = run_prediction_error(
        price_returns,
        market_caps=market_caps,
        eval_start="2019-01-01",
        eval_end="2024-12-31",
    )

    resumen = resumen_prediction_error(df_error)
    mensual = resumen_prediction_error_mensual(df_error)

    print("\nEvaluando error predictivo por perfil 2019-2024...")
    df_perfiles = run_profile_prediction_error(
        price_returns,
        market_caps=market_caps,
        eval_start="2019-01-01",
        eval_end="2024-12-31",
    )
    resumen_perfiles = resumen_prediction_error_por_perfil(df_perfiles)

    print("\n-- Error promedio de prediccion -------------------------------")
    print(f"  Observaciones activo-mes : {resumen['observaciones']:,}")
    print(f"  MAE promedio             : {resumen['mae']:.2%}")
    print(f"  RMSE promedio            : {resumen['rmse']:.2%}")
    print(f"  Sesgo promedio           : {resumen['bias']:+.2%}")
    print(f"  Acierto de signo         : {resumen['hit_rate_signo']:.1%}")
    print(f"  Corr predicho-realizado  : {resumen['corr_pred_real']:.2f}")

    print("\n-- Error promedio por perfil ----------------------------------")
    print(resumen_perfiles.to_string(formatters={
        "mae": "{:.2%}".format,
        "rmse": "{:.2%}".format,
        "bias": "{:+.2%}".format,
        "hit_rate_signo": "{:.1%}".format,
        "peso_acciones_promedio": "{:.1%}".format,
        "n_activos_promedio": "{:.1f}".format,
        "corr_pred_real": "{:.2f}".format,
    }))

    os.makedirs(CARPETA, exist_ok=True)
    df_error.to_csv(f"{CARPETA}/errores_prediccion_activo_mes.csv", index=False)
    mensual.to_csv(f"{CARPETA}/resumen_error_prediccion_mensual.csv")
    pd.DataFrame([resumen]).to_csv(f"{CARPETA}/resumen_error_prediccion.csv", index=False)
    df_perfiles.to_csv(f"{CARPETA}/errores_prediccion_perfil_mes.csv", index=False)
    resumen_perfiles.to_csv(f"{CARPETA}/resumen_error_prediccion_por_perfil.csv")

    print(f"\nResultados guardados en {CARPETA}/")


if __name__ == "__main__":
    main()
