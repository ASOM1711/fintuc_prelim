from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "resultados_diagnostico"
METRICAS = OUT_DIR / "max_weight_decision_metricas_anuales.csv"
ERROR = OUT_DIR / "max_weight_decision_error_predictivo.csv"
RESUMEN = OUT_DIR / "max_weight_decision_resumen.csv"
RESUMEN_SIN_PARCIAL = OUT_DIR / "max_weight_decision_resumen_2019_2025.csv"
METRICAS_SIN_PARCIAL = OUT_DIR / "max_weight_decision_metricas_anuales_2019_2025.csv"


def main():
    metricas = pd.read_csv(METRICAS)
    error = pd.read_csv(ERROR)

    max_full_year = 2025
    metricas = metricas[metricas["anio"] <= max_full_year].copy()
    metricas.to_csv(METRICAS_SIN_PARCIAL, index=False)

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
    resumen["score"] = (
        resumen["retorno_promedio"]
        - 0.50 * resumen["drawdown_promedio"]
        - 0.50 * resumen["mae"]
        - 0.30 * resumen["abandono_promedio"]
    )
    resumen = resumen.sort_values(["perfil", "score"], ascending=[True, False])

    resumen.to_csv(RESUMEN_SIN_PARCIAL, index=False)
    resumen.to_csv(RESUMEN, index=False)
    print(RESUMEN)
    print(RESUMEN_SIN_PARCIAL)


if __name__ == "__main__":
    main()
