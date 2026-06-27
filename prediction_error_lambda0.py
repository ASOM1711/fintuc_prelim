"""Corre el error predictivo por perfil usando lambda = 0."""
from pathlib import Path

import pandas as pd

import data.loader as loader
from backtesting.prediction_error import (
    run_profile_prediction_error,
    resumen_prediction_error_por_perfil,
)


CARPETA = Path("resultados_prediccion")


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

    print("\nEvaluando error predictivo por perfil con lambda=0...")
    df_error = run_profile_prediction_error(
        price_returns,
        market_caps=market_caps,
        eval_start="2019-01-01",
        eval_end="2024-12-31",
        lam=0.0,
    )
    resumen = resumen_prediction_error_por_perfil(df_error)

    print("\n-- Error promedio por perfil: lambda=0 -------------------------")
    print(resumen.to_string(formatters={
        "mae": "{:.2%}".format,
        "rmse": "{:.2%}".format,
        "bias": "{:+.2%}".format,
        "hit_rate_signo": "{:.1%}".format,
        "peso_acciones_promedio": "{:.1%}".format,
        "n_activos_promedio": "{:.1f}".format,
        "corr_pred_real": "{:.2f}".format,
    }))

    CARPETA.mkdir(exist_ok=True)
    df_error.to_csv(CARPETA / "errores_prediccion_perfil_mes_lambda0.csv", index=False)
    resumen.to_csv(CARPETA / "resumen_error_prediccion_por_perfil_lambda0.csv")

    lambda1_path = CARPETA / "resumen_error_prediccion_por_perfil.csv"
    if lambda1_path.exists():
        lambda1 = pd.read_csv(lambda1_path).set_index("perfil")
        comparacion = resumen.add_suffix("_lambda0").join(
            lambda1.add_suffix("_lambda1"),
            how="outer",
        )
        comparacion["delta_mae"] = comparacion["mae_lambda0"] - comparacion["mae_lambda1"]
        comparacion["delta_rmse"] = comparacion["rmse_lambda0"] - comparacion["rmse_lambda1"]
        comparacion.to_csv(CARPETA / "comparacion_error_lambda0_vs_lambda1.csv")

        print("\n-- Diferencia lambda=0 menos lambda=1 --------------------------")
        print(comparacion[["delta_mae", "delta_rmse"]].to_string(formatters={
            "delta_mae": "{:+.2%}".format,
            "delta_rmse": "{:+.2%}".format,
        }))

    print(f"\nResultados guardados en {CARPETA}/")


if __name__ == "__main__":
    main()
