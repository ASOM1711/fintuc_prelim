"""
Análisis de sensibilidad one-at-a-time (OAT) para FinPUC.
Varía cada parámetro por separado y guarda tablas + gráficos.

Uso:
    python sensitivity_main.py
"""
import os
import pandas as pd

from data.loader import load_universe, load_stock_info
from backtesting.sensitivity import (
    run_sensitivity, tabla_sensibilidad, PARAM_GRIDS, PARAM_LABELS,
)
from visualization.plots import plot_sensibilidad


CARPETA = "graficos/sensibilidad"
METRICAS = {
    "retorno_anual":      "Retorno anual (CAGR)",
    "sharpe":             "Ratio de Sharpe",
    "max_drawdown":       "Drawdown maximo",
    "caja_chica_promedio":"Caja chica promedio",
}

# Parametros a analizar (puedes comentar los que no quieras correr)
PARAMS_A_ANALIZAR = [
    "conf_base",
    "lookback",
    "lam",
    "max_weight",
    "commission",
    "exceso_critico",
]


def main():
    print("Cargando datos...")
    price_returns, div_yields = load_universe()
    info        = load_stock_info()
    market_caps = info["marketCap"] if "marketCap" in info.columns else None

    os.makedirs(CARPETA, exist_ok=True)
    resultados_sens = {}

    for param in PARAMS_A_ANALIZAR:
        label = PARAM_LABELS[param]
        print(f"\n=== Sensibilidad: {label} ===")

        df_sens = run_sensitivity(
            price_returns, div_yields,
            param_name=param,
            market_caps=market_caps,
            seed=42,
        )
        resultados_sens[param] = df_sens
        df_sens.to_csv(f"{CARPETA}/sens_{param}.csv")

        # Grafico de retorno anual
        plot_sensibilidad(
            df_sens,
            param_label=label,
            metrica="retorno_anual",
            metrica_label="Retorno anual (CAGR)",
            guardar=f"{CARPETA}/sens_{param}_retorno.png",
        )

        # Grafico de drawdown maximo
        plot_sensibilidad(
            df_sens,
            param_label=label,
            metrica="max_drawdown",
            metrica_label="Drawdown maximo",
            guardar=f"{CARPETA}/sens_{param}_drawdown.png",
        )

    # Imprimir tabla resumen para cada parametro
    print("\n\n" + "="*70)
    print("TABLAS RESUMEN DE SENSIBILIDAD")
    print("="*70)

    for param, df_sens in resultados_sens.items():
        label = PARAM_LABELS[param]
        print(f"\n--- {label} ---")
        tablas = tabla_sensibilidad(df_sens, metricas=["retorno_anual", "sharpe", "max_drawdown"])

        for metrica, pivot in tablas.items():
            print(f"\n  {metrica}:")
            # Formatear segun tipo de metrica
            if metrica in ("retorno_anual", "max_drawdown", "caja_chica_promedio"):
                print(pivot.map(lambda x: f"{x:.1%}" if pd.notna(x) else "").to_string())
            else:
                print(pivot.map(lambda x: f"{x:.2f}" if pd.notna(x) else "").to_string())

    print(f"\nResultados guardados en {CARPETA}/")


if __name__ == "__main__":
    main()
