"""
Evaluacion out-of-sample del error de prediccion de Black-Litterman.

Compara los retornos esperados estimados al inicio de cada mes contra los
retornos efectivamente realizados durante ese mes.
"""
import numpy as np
import pandas as pd

from config import TRAIN_WINDOW_YEARS
from models.black_littermanprelim import black_litterman


def _train_window(
    returns: pd.DataFrame,
    mes: pd.Timestamp,
    train_years: int,
) -> pd.DataFrame:
    end = mes - pd.Timedelta(days=1)
    start = mes - pd.DateOffset(years=train_years)
    return returns.loc[start:end]


def _retorno_realizado_periodo(retornos_mes: pd.DataFrame) -> pd.Series:
    """Retorno compuesto observado durante el periodo para cada activo."""
    return (1 + retornos_mes).prod() - 1


def _retorno_predicho_periodo(mu_diario: np.ndarray, n_dias: int) -> np.ndarray:
    """Convierte retorno esperado diario a retorno compuesto del periodo."""
    return (1 + mu_diario) ** n_dias - 1


def run_prediction_error(
    price_returns: pd.DataFrame,
    market_caps: pd.Series | None = None,
    eval_start: str = "2019-01-01",
    eval_end: str = "2024-12-31",
    train_years: int = TRAIN_WINDOW_YEARS,
    conf_base: float = 0.05,
    lookback: int = 252,
    skip: int = 21,
) -> pd.DataFrame:
    """
    Corre un test walk-forward mensual del error predictivo.

    Para cada mes:
    1. Entrena Black-Litterman con la ventana historica previa.
    2. Proyecta el retorno esperado de cada activo para el mes.
    3. Lo compara contra el retorno compuesto observado en ese mes.

    Retorna un DataFrame a nivel activo-mes con prediccion, realizado y error.
    """
    meses = pd.date_range(eval_start, eval_end, freq="MS")

    train_init = _train_window(price_returns, meses[0], train_years)
    train_init = train_init.dropna(axis=1, how="any")
    tickers = list(train_init.columns)
    pr = price_returns[tickers]

    registros: list[dict] = []
    for mes in meses:
        mes_end = mes + pd.offsets.MonthEnd(0)
        train = _train_window(pr, mes, train_years).dropna(axis=1, how="any")
        retornos_mes = pr.loc[mes:mes_end, train.columns].dropna(axis=0, how="any")

        if train.empty or retornos_mes.empty:
            continue

        train_r = train.clip(lower=-0.50, upper=0.50)
        mu_bl = black_litterman(
            train_r,
            market_caps=market_caps,
            conf_base=conf_base,
            lookback=lookback,
            skip=skip,
        )

        predicho = _retorno_predicho_periodo(mu_bl, len(retornos_mes))
        realizado = _retorno_realizado_periodo(retornos_mes).values
        error = predicho - realizado

        for ticker, pred, real, err in zip(train.columns, predicho, realizado, error):
            registros.append({
                "fecha": mes,
                "ticker": ticker,
                "retorno_predicho": pred,
                "retorno_realizado": real,
                "error": err,
                "abs_error": abs(err),
                "sq_error": err ** 2,
                "signo_correcto": np.sign(pred) == np.sign(real),
            })

    return pd.DataFrame(registros)


def resumen_prediction_error(df: pd.DataFrame) -> dict:
    """Resume el error predictivo promedio a traves de todos los activo-mes."""
    if df.empty:
        return {
            "observaciones": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
            "hit_rate_signo": np.nan,
            "corr_pred_real": np.nan,
        }

    corr = df["retorno_predicho"].corr(df["retorno_realizado"])
    return {
        "observaciones": int(len(df)),
        "mae": round(float(df["abs_error"].mean()), 6),
        "rmse": round(float(np.sqrt(df["sq_error"].mean())), 6),
        "bias": round(float(df["error"].mean()), 6),
        "hit_rate_signo": round(float(df["signo_correcto"].mean()), 4),
        "corr_pred_real": round(float(corr), 4) if pd.notna(corr) else np.nan,
    }


def resumen_prediction_error_mensual(df: pd.DataFrame) -> pd.DataFrame:
    """Entrega metricas agregadas por mes."""
    if df.empty:
        return pd.DataFrame()

    out = df.groupby("fecha").agg(
        mae=("abs_error", "mean"),
        rmse=("sq_error", lambda x: float(np.sqrt(x.mean()))),
        bias=("error", "mean"),
        hit_rate_signo=("signo_correcto", "mean"),
        n_activos=("ticker", "count"),
    )
    return out.round({
        "mae": 6,
        "rmse": 6,
        "bias": 6,
        "hit_rate_signo": 4,
    })


def run_profile_prediction_error(
    price_returns: pd.DataFrame,
    market_caps: pd.Series | None = None,
    eval_start: str = "2019-01-01",
    eval_end: str = "2024-12-31",
    train_years: int = TRAIN_WINDOW_YEARS,
    conf_base: float = 0.05,
    lookback: int = 252,
    skip: int = 21,
    lam: float = 1.0,
    max_weight: float | None = None,
) -> pd.DataFrame:
    """
    Evalua el error predictivo a nivel portafolio para cada perfil.

    Para cada mes rolling:
    1. Calcula mu_BL con informacion historica previa.
    2. Optimiza pesos para cada perfil usando esa prediccion.
    3. Compara retorno mensual predicho del portafolio contra el realizado.
    """
    from backtesting.runner import _optimizar
    from config import MAX_WEIGHT, RISK_PROFILES

    meses = pd.date_range(eval_start, eval_end, freq="MS")
    max_weight = MAX_WEIGHT if max_weight is None else max_weight

    train_init = _train_window(price_returns, meses[0], train_years)
    train_init = train_init.dropna(axis=1, how="any")
    tickers = list(train_init.columns)
    pr = price_returns[tickers]

    registros: list[dict] = []
    for mes in meses:
        mes_end = mes + pd.offsets.MonthEnd(0)
        train = _train_window(pr, mes, train_years).dropna(axis=1, how="any")
        retornos_mes = pr.loc[mes:mes_end, train.columns].dropna(axis=0, how="any")

        if train.empty or retornos_mes.empty:
            continue

        train_r = train.clip(lower=-0.50, upper=0.50)
        mu_bl = black_litterman(
            train_r,
            market_caps=market_caps,
            conf_base=conf_base,
            lookback=lookback,
            skip=skip,
        )

        predicho_activos = _retorno_predicho_periodo(mu_bl, len(retornos_mes))
        realizado_activos = _retorno_realizado_periodo(retornos_mes).values

        for perfil, tolerancia in RISK_PROFILES.items():
            pesos = _optimizar(
                train_r,
                tolerancia,
                perfil,
                mu_bl=mu_bl,
                lam=lam,
                max_weight=max_weight,
            )
            predicho = float(np.dot(pesos, predicho_activos))
            realizado = float(np.dot(pesos, realizado_activos))
            error = predicho - realizado

            registros.append({
                "fecha": mes,
                "perfil": perfil,
                "retorno_predicho": predicho,
                "retorno_realizado": realizado,
                "error": error,
                "abs_error": abs(error),
                "sq_error": error ** 2,
                "signo_correcto": np.sign(predicho) == np.sign(realizado),
                "peso_acciones": float(pesos.sum()),
                "n_activos": int((pesos > 1e-8).sum()),
            })

    return pd.DataFrame(registros)


def resumen_prediction_error_por_perfil(df: pd.DataFrame) -> pd.DataFrame:
    """Resume el error predictivo agregado por perfil."""
    if df.empty:
        return pd.DataFrame()

    out = df.groupby("perfil").agg(
        observaciones=("fecha", "count"),
        mae=("abs_error", "mean"),
        rmse=("sq_error", lambda x: float(np.sqrt(x.mean()))),
        bias=("error", "mean"),
        hit_rate_signo=("signo_correcto", "mean"),
        peso_acciones_promedio=("peso_acciones", "mean"),
        n_activos_promedio=("n_activos", "mean"),
    )

    corr = df.groupby("perfil").apply(
        lambda x: x["retorno_predicho"].corr(x["retorno_realizado"]),
        include_groups=False,
    )
    out["corr_pred_real"] = corr

    return out.round({
        "mae": 6,
        "rmse": 6,
        "bias": 6,
        "hit_rate_signo": 4,
        "peso_acciones_promedio": 4,
        "n_activos_promedio": 2,
        "corr_pred_real": 4,
    })
