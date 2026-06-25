"""
Analisis de sensibilidad exclusivo para lambda.

Uso:
    python lambda_sensitivity_main.py
"""
import os
from pathlib import Path

import pandas as pd

import data.loader as loader
from backtesting.sensitivity import run_sensitivity, tabla_sensibilidad


CARPETA = "graficos/sensibilidad"
LAMBDAS = [0.0, 0.5, 1.0, 2.0]


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
    price_returns, div_yields = _load_universe_with_fallback()
    market_caps = _load_market_caps()

    if price_returns.empty:
        raise FileNotFoundError(
            "No se encontraron retornos de acciones. Revisa DATA_DIR en config.py."
        )

    print(f"  Tickers cargados : {price_returns.shape[1]}")
    print(f"  Periodo de datos : {price_returns.index[0].date()} -> {price_returns.index[-1].date()}")

    print("\n=== Sensibilidad: Aversion al riesgo (lambda) ===")
    df_sens = run_sensitivity(
        price_returns,
        div_yields,
        param_name="lam",
        param_values=LAMBDAS,
        market_caps=market_caps,
        seed=42,
    )

    os.makedirs(CARPETA, exist_ok=True)
    out_path = f"{CARPETA}/sens_lam.csv"
    df_sens.to_csv(out_path)

    print("\n-- Tablas resumen lambda --------------------------------------")
    tablas = tabla_sensibilidad(
        df_sens,
        metricas=[
            "retorno_anual",
            "sharpe",
            "max_drawdown",
            "comision_total",
            "frecuencia_rebalanceo",
            "caja_chica_promedio",
        ],
    )

    for metrica, pivot in tablas.items():
        print(f"\n  {metrica}:")
        if metrica in ("retorno_anual", "max_drawdown", "frecuencia_rebalanceo", "caja_chica_promedio"):
            print(pivot.map(lambda x: f"{x:.1%}" if pd.notna(x) else "").to_string())
        elif metrica == "comision_total":
            print(pivot.map(lambda x: f"${x:,.0f}" if pd.notna(x) else "").to_string())
        else:
            print(pivot.map(lambda x: f"{x:.2f}" if pd.notna(x) else "").to_string())

    print(f"\nResultados guardados en {out_path}")


if __name__ == "__main__":
    main()
