import numpy as np
import pandas as pd
from gurobipy import GRB

from config import RISK_PROFILES, TRAIN_WINDOW_YEARS, MAX_WEIGHT, COMMISSION
from models.black_littermanprelim import black_litterman
from models.markowitzprelim import optimizar_markowitz
from portfolio.engine import PortfolioState, paso_mensual


_LAM = 1.0   # tradeoff riesgo-retorno; la restricción QCQP maneja el riesgo
_CLIP = 0.50  # winsorización de retornos extremos (ej. NCPL 2665x el 2020-11-11)


def _train_window(
    returns: pd.DataFrame,
    mes: pd.Timestamp,
    train_years: int,
) -> pd.DataFrame:
    end   = mes - pd.Timedelta(days=1)
    start = mes - pd.DateOffset(years=train_years)
    return returns.loc[start:end]


def _optimizar(
    train: pd.DataFrame,
    tolerancia: float,
    perfil: str,
    mu_bl: np.ndarray | None = None,
    lam: float = _LAM,
    max_weight: float = MAX_WEIGHT,
) -> np.ndarray:
    """
    Resuelve Markowitz dado mu_BL precalculado (o lo calcula si es None).
    Devuelve pesos óptimos; zeros si tolerancia==0 (muy_conservador).
    """
    n = len(train.columns)
    if tolerancia == 0.0:
        return np.zeros(n)

    train_r = train.clip(lower=-_CLIP, upper=_CLIP)

    if mu_bl is None:
        mu_bl = black_litterman(train_r)

    model, w_vars = optimizar_markowitz(
        train_r,
        lam=lam,
        perdida_max_anual=tolerancia,
        perfil=perfil,
        mu_personalizado=mu_bl,
        max_weight=max_weight,
    )
    if model.Status == GRB.OPTIMAL:
        return np.array([w_vars[i].X for i in range(n)])
    return np.ones(n) / n


def run_backtest(
    price_returns: pd.DataFrame,
    div_yields: pd.DataFrame,
    perfil: str,
    capital_inicial: float = 1_000_000,
    eval_start: str = "2019-01-01",
    eval_end: str = "2024-12-31",
    train_years: int = TRAIN_WINDOW_YEARS,
    mu_bl_cache: dict | None = None,
    lam: float = _LAM,
    max_weight: float = MAX_WEIGHT,
    commission: float = COMMISSION,
    exceso_critico_p1: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Walk-forward backtest mensual para un perfil de riesgo.

    Parámetros parametrizables para análisis de sensibilidad
    ---------------------------------------------------------
    lam               : aversión al riesgo en la función objetivo de Markowitz
    max_weight        : peso máximo por activo en el portafolio
    commission        : tasa de comisión anual sobre AUM (fracción, e.g. 0.01 = 1%)
    exceso_critico_p1 : exceso de drawdown sobre tolerancia que dispara P1 al 50%
    """
    tolerancia = RISK_PROFILES[perfil]
    rng        = np.random.default_rng(seed)
    meses      = pd.date_range(eval_start, eval_end, freq="MS")

    pr = price_returns
    dv = div_yields

    train_init = _train_window(pr, meses[0], train_years)
    mu_bl_init = mu_bl_cache.get(meses[0]) if mu_bl_cache else None
    pesos_0    = _optimizar(train_init, tolerancia, perfil,
                            mu_bl=mu_bl_init, lam=lam, max_weight=max_weight)
    state      = PortfolioState.crear(list(pr.columns), capital_inicial, pesos_0)

    registros = []
    for mes in meses:
        mes_end = mes + pd.offsets.MonthEnd(0)
        pr_mes  = pr.loc[mes:mes_end]
        dv_mes  = dv.loc[mes:mes_end]

        if pr_mes.empty:
            continue

        train     = _train_window(pr, mes, train_years)
        mu_bl     = mu_bl_cache.get(mes) if mu_bl_cache else None
        pesos_opt = _optimizar(train, tolerancia, perfil,
                               mu_bl=mu_bl, lam=lam, max_weight=max_weight)

        metricas = paso_mensual(
            state, pr_mes, dv_mes, pesos_opt, tolerancia,
            exceso_critico_p1=exceso_critico_p1,
            commission_rate=commission,
            rng=rng,
        )
        metricas["fecha"]  = mes
        metricas["perfil"] = perfil
        registros.append(metricas)

        if metricas["abandona"]:
            break

    return pd.DataFrame(registros).set_index("fecha")


def run_all_profiles(
    price_returns: pd.DataFrame,
    div_yields: pd.DataFrame,
    capital_inicial: float = 1_000_000,
    eval_start: str = "2019-01-01",
    eval_end: str = "2024-12-31",
    market_caps: pd.Series | None = None,
    lam: float = _LAM,
    max_weight: float = MAX_WEIGHT,
    commission: float = COMMISSION,
    conf_base: float = 0.05,
    lookback: int = 252,
    exceso_critico_p1: float = 0.15,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Corre el backtest para los 5 perfiles de riesgo.

    Calcula Black-Litterman UNA vez por mes (compartido entre perfiles).
    Todos los parámetros del modelo son inyectables para análisis de sensibilidad.
    """
    meses = pd.date_range(eval_start, eval_end, freq="MS")

    train_init = _train_window(price_returns, meses[0], TRAIN_WINDOW_YEARS)
    train_init = train_init.dropna(axis=1, how="any")
    tickers    = list(train_init.columns)

    pr = price_returns[tickers]
    dv = div_yields.reindex(columns=tickers).fillna(0)

    if verbose:
        print("  Calculando Black-Litterman por mes...", end="", flush=True)

    mu_bl_cache: dict = {}
    for mes in meses:
        train   = _train_window(pr, mes, TRAIN_WINDOW_YEARS)
        train_r = train.clip(lower=-_CLIP, upper=_CLIP)
        mu_bl_cache[mes] = black_litterman(
            train_r, market_caps=market_caps,
            conf_base=conf_base, lookback=lookback,
        )

    if verbose:
        print(f" {len(mu_bl_cache)} meses listos.")

    return {
        perfil: run_backtest(
            pr, dv,
            perfil=perfil,
            capital_inicial=capital_inicial,
            eval_start=eval_start,
            eval_end=eval_end,
            mu_bl_cache=mu_bl_cache,
            lam=lam,
            max_weight=max_weight,
            commission=commission,
            exceso_critico_p1=exceso_critico_p1,
            seed=seed,
        )
        for perfil in RISK_PROFILES
    }


def run_benchmark(
    price_returns: pd.DataFrame,
    div_yields: pd.DataFrame,
    n_acciones: int = 30,
    capital_inicial: float = 1_000_000,
    eval_start: str = "2019-01-01",
    eval_end: str = "2024-12-31",
    train_years: int = TRAIN_WINDOW_YEARS,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Benchmark pasivo: n_acciones elegidas al azar con pesos iguales (1/n).
    Buy-and-hold sin rebalanceo ni comisión.
    """
    meses = pd.date_range(eval_start, eval_end, freq="MS")

    train_init = _train_window(price_returns, meses[0], train_years)
    train_init = train_init.dropna(axis=1, how="any")
    tickers_universo = list(train_init.columns)

    rng        = np.random.default_rng(seed)
    tickers_bm = list(rng.choice(tickers_universo, n_acciones, replace=False))

    pr = price_returns[tickers_bm]
    dv = div_yields.reindex(columns=tickers_bm).fillna(0)

    pesos_bm = np.ones(n_acciones) / n_acciones
    state    = PortfolioState.crear(tickers_bm, capital_inicial, pesos_bm)

    registros = []
    for mes in meses:
        mes_end = mes + pd.offsets.MonthEnd(0)
        pr_mes  = pr.loc[mes:mes_end].values
        dv_mes  = dv.loc[mes:mes_end].values

        if len(pr_mes) == 0:
            continue

        for t in range(len(pr_mes)):
            state.actualizar_dia(pr_mes[t], dv_mes[t])

        registros.append({
            "fecha":    mes,
            "valor":    state.valor,
            "drawdown": state.drawdown(),
        })

    return pd.DataFrame(registros).set_index("fecha")


def resumen_backtest(df: pd.DataFrame) -> dict:
    retornos  = df["valor"].pct_change().dropna()
    n_meses   = len(df)

    ret_anual = (1 + retornos).prod() ** (12 / n_meses) - 1 if n_meses > 0 else 0.0
    vol_anual = retornos.std() * np.sqrt(12)
    sharpe    = ret_anual / vol_anual if vol_anual > 1e-6 else 0.0

    return {
        "retorno_anual":         round(ret_anual, 4),
        "volatilidad_anual":     round(vol_anual, 4),
        "sharpe":                round(sharpe, 4),
        "max_drawdown":          round(df["drawdown"].max(), 4),
        "meses_activos":         n_meses,
        "abandono":              bool(df["abandona"].any()),
        "comision_total":        round(df["comision"].sum(), 2),
        "frecuencia_rebalanceo": round(df["rebalanceo"].mean(), 4),
        "caja_chica_promedio":   round(df["caja_chica"].mean(), 4),
    }
