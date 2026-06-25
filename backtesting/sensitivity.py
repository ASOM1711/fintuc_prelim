"""
Análisis de sensibilidad one-at-a-time (OAT) para el modelo FinPUC.

Parámetros analizables
----------------------
conf_base       : confianza en los views de momentum de Black-Litterman (default 0.05)
lookback        : días de historia para el momentum BL (default 252 = 1 año)
lam             : aversión al riesgo en el objetivo de Markowitz (default 1.0)
max_weight      : peso máximo por activo (default 0.15)
commission      : tasa de comisión anual sobre AUM (default 0.01 = 1%)
exceso_critico  : exceso de drawdown sobre tolerancia que lleva P1 al 50% (default 0.15)
"""
import numpy as np
import pandas as pd

from backtesting.runner import run_all_profiles, resumen_backtest
from config import COMMISSION, MAX_WEIGHT

# Valores base de todos los parámetros
_BASE = {
}

# Grillas de valores a explorar por parámetro
PARAM_GRIDS: dict[str, list] = {
    "conf_base":      [0.01, 0.05, 0.20],
    "lookback":       [126, 252],
    "lam":            [0.0, 0.5, 1.0, 2.0],
    "max_weight":     [0.10, 0.15, 0.20],
    "commission":     [0.005, 0.01, 0.015],
    "exceso_critico": [0.10, 0.15, 0.20],
}

# Etiquetas legibles para tablas y gráficos
PARAM_LABELS: dict[str, str] = {
}


def run_sensitivity(
    price_returns: pd.DataFrame,
    div_yields: pd.DataFrame,
    param_name: str,
    param_values: list | None = None,
    market_caps: pd.Series | None = None,
    capital_inicial: float = 1_000_000,
    eval_start: str = "2019-01-01",
    eval_end: str = "2024-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Corre run_all_profiles variando un único parámetro (OAT).

    Parámetros
    ----------
    price_returns : retornos de precio diarios
    div_yields    : dividend yields diarios
    param_name    : nombre del parámetro a variar (ver PARAM_GRIDS)
    param_values  : valores a probar; si None usa PARAM_GRIDS[param_name]
    market_caps   : capitalización de mercado para Black-Litterman
    capital_inicial, eval_start, eval_end, seed : igual que run_all_profiles

    Retorna
    -------
    DataFrame con MultiIndex (param_value, perfil) y columnas de resumen_backtest.
    """
    if param_name not in _BASE:
        raise ValueError(f"param_name debe ser uno de: {list(_BASE)}")

    valores = param_values if param_values is not None else PARAM_GRIDS[param_name]

    registros = []
    for val in valores:
        kwargs = dict(_BASE)
        kwargs[param_name] = val

        label = f"{val:.0%}" if param_name == "commission" else (
                f"{val:.0%}" if param_name in ("conf_base", "max_weight") else str(val))

        print(f"  [{PARAM_LABELS[param_name]}] {label} ...", end="", flush=True)

        resultados = run_all_profiles(
            price_returns, div_yields,
            capital_inicial=capital_inicial,
            eval_start=eval_start,
            eval_end=eval_end,
            market_caps=market_caps,
            lam=kwargs["lam"],
            max_weight=kwargs["max_weight"],
            commission=kwargs["commission"],
            conf_base=kwargs["conf_base"],
            lookback=kwargs["lookback"],
            seed=seed,
            verbose=False,
        )

        for perfil, df in resultados.items():
            r = resumen_backtest(df)
            r["param_value"] = val
            r["perfil"]      = perfil
            registros.append(r)

        print(" listo.")

    df_out = pd.DataFrame(registros).set_index(["param_value", "perfil"])
    return df_out


def tabla_sensibilidad(df_sens: pd.DataFrame, metricas: list | None = None) -> pd.DataFrame:
    """
    Formatea el DataFrame de sensibilidad para presentación en el informe.

    Retorna tabla con param_value como filas, perfiles como columnas,
    y la métrica seleccionada como valores.
    """
    if metricas is None:
        metricas = ["retorno_anual", "sharpe", "max_drawdown"]

    perfiles = list(df_sens.index.get_level_values("perfil").unique())
    param_vals = list(df_sens.index.get_level_values("param_value").unique())

    partes = {}
    for m in metricas:
        pivot = df_sens[m].unstack("perfil").reindex(columns=perfiles)
        partes[m] = pivot

    return partes
