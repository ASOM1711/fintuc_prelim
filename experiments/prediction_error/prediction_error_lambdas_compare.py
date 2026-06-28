"""Compara error predictivo por perfil para lambda=0 y lambda=1 con el codigo actual."""
from pathlib import Path

import pandas as pd

import data.loader as loader
from backtesting.prediction_error import (
    run_profile_prediction_error,
    resumen_prediction_error_por_perfil,
)


CARPETA = Path("resultados_prediccion")
LAMBDAS = [0.0, 1.0]


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

    CARPETA.mkdir(exist_ok=True)
    resumenes = []

    for lam in LAMBDAS:
        print(f"\nEvaluando error predictivo por perfil con lambda={lam}...")
        df_error = run_profile_prediction_error(
            price_returns,
            market_caps=market_caps,
            eval_start="2019-01-01",
            eval_end="2024-12-31",
            lam=lam,
        )
        resumen = resumen_prediction_error_por_perfil(df_error)
        resumen["lambda"] = lam
        resumenes.append(resumen.reset_index())
        df_error.to_csv(CARPETA / f"errores_prediccion_perfil_mes_lambda{lam:g}.csv", index=False)
        resumen.to_csv(CARPETA / f"resumen_error_prediccion_por_perfil_lambda{lam:g}.csv")

    combinado = pd.concat(resumenes, ignore_index=True)
    combinado.to_csv(CARPETA / "comparacion_error_lambdas_actual.csv", index=False)

    print("\n-- MAE por perfil ----------------------------------------------")
    mae = combinado.pivot(index="perfil", columns="lambda", values="mae")
    print(mae.to_string(formatters={col: "{:.2%}".format for col in mae.columns}))

    print("\n-- RMSE por perfil ---------------------------------------------")
    rmse = combinado.pivot(index="perfil", columns="lambda", values="rmse")
    print(rmse.to_string(formatters={col: "{:.2%}".format for col in rmse.columns}))

    print(f"\nResultados guardados en {CARPETA}/")


if __name__ == "__main__":
    main()
