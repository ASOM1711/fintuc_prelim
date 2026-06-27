import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from gurobipy import GRB

from config import RISK_PROFILES, TRAIN_WINDOW_YEARS, MAX_WEIGHT, COMMISSION
from models.black_littermanprelim import black_litterman, robust_factor_black_litterman
from models.markowitzprelim import optimizar_markowitz, calibrar_lambda
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


_VOL_FACTOR = 0.5  # fracción de la tolerancia usada como límite de volatilidad QCQP


def _optimizar(
    train: pd.DataFrame,
    tolerancia: float,
    perfil: str,
    mu_bl: np.ndarray | None = None,
    lam: float = _LAM,
    max_weight: float = MAX_WEIGHT,
    vol_factor: float = _VOL_FACTOR,
    use_bl: bool = True,
    full_invest: bool = False,
) -> np.ndarray:
    """
    Resuelve Markowitz dado mu_BL precalculado (o lo calcula si es None).
    Devuelve pesos óptimos; zeros si tolerancia==0 (muy_conservador).

    use_bl     : si False, usa media histórica en lugar de Black-Litterman.
    vol_factor : fracción de la tolerancia usada en el constraint QCQP.
    """
    n = len(train.columns)
    if tolerancia == 0.0:
        return np.zeros(n)

    train_r = train.clip(lower=-_CLIP, upper=_CLIP)

    if use_bl:
        if mu_bl is None:
            mu_bl = black_litterman(train_r)
        mu_final = mu_bl
    else:
        mu_final = None  # optimizar_markowitz usará la media histórica

    model, w_vars = optimizar_markowitz(
        train_r,
        lam=lam,
        perdida_max_anual=tolerancia * vol_factor,
        perfil=perfil,
        mu_personalizado=mu_final,
        max_weight=max_weight,
        full_invest=full_invest,
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
    vol_factor: float = _VOL_FACTOR,
    full_invest: bool = False,
    p1_lineal: bool = False,
    use_bl: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Walk-forward backtest mensual para un perfil de riesgo.

    Parámetros parametrizables para análisis de sensibilidad
    ---------------------------------------------------------
    lam         : aversión al riesgo en la función objetivo de Markowitz
    max_weight  : peso máximo por activo en el portafolio
    commission  : tasa de comisión anual sobre AUM (fracción, e.g. 0.01 = 1%)
    vol_factor  : fracción de la tolerancia usada como límite de volatilidad QCQP
    full_invest : si True, Σw=1 (cliente invierte el 100%; caja chica solo dividendos)
    """
    tolerancia = RISK_PROFILES[perfil]
    rng        = np.random.default_rng(seed)
    meses      = pd.date_range(eval_start, eval_end, freq="MS")

    pr = price_returns
    dv = div_yields

    train_init = _train_window(pr, meses[0], train_years)
    mu_bl_init = mu_bl_cache.get(meses[0]) if mu_bl_cache else None
    pesos_0    = _optimizar(train_init, tolerancia, perfil,
                            mu_bl=mu_bl_init, lam=lam, max_weight=max_weight,
                            full_invest=full_invest, use_bl=use_bl)
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
        mu_signal = mu_bl
        if not use_bl:
            train_r = train.clip(lower=-_CLIP, upper=_CLIP)
            mu_signal = train_r.mean().values
        pesos_opt = _optimizar(train, tolerancia, perfil,
                               mu_bl=mu_bl, lam=lam, max_weight=max_weight,
                               vol_factor=vol_factor, full_invest=full_invest,
                               use_bl=use_bl)

        metricas = paso_mensual(
            state, pr_mes, dv_mes, pesos_opt, tolerancia,
            mu_bl=mu_signal,
            commission_rate=commission,
            rng=rng,
            p1_lineal=p1_lineal,
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
    train_years: int = TRAIN_WINDOW_YEARS,
    market_caps: pd.Series | None = None,
    lam: float | None = None,
    max_weight: float = MAX_WEIGHT,
    commission: float = COMMISSION,
    vol_factor: float = _VOL_FACTOR,
    full_invest: bool = False,
    p1_lineal: bool = False,
    conf_base: float = 0.05,
    lookback: int = 252,
    skip: int = 21,
    bl_method: str = "asset_momentum",
    seed: int = 42,
    use_bl: bool = True,
    verbose: bool = True,
    return_cache: bool = False,
) -> "dict[str, pd.DataFrame] | tuple[dict, dict]":
    """
    Corre el backtest para los 5 perfiles de riesgo.

    Calcula Black-Litterman UNA vez por mes (compartido entre perfiles).
    Todos los parámetros del modelo son inyectables para análisis de sensibilidad.
    """
    meses = pd.date_range(eval_start, eval_end, freq="MS")

    train_init = _train_window(price_returns, meses[0], train_years)
    train_init = train_init.dropna(axis=1, how="any")
    tickers    = list(train_init.columns)

    pr = price_returns[tickers]
    dv = div_yields.reindex(columns=tickers).fillna(0)

    # Calibrar lambda por perfil usando la ventana inicial (una sola vez)
    train_init_r = train_init.clip(lower=-_CLIP, upper=_CLIP)
    if lam is None:
        if verbose:
            print("  Calibrando lambda por perfil (frontera eficiente)...")
        lambda_por_perfil: dict = {}
        _LAM_MAX = 5000.0
        for perfil, tolerancia in RISK_PROFILES.items():
            if tolerancia == 0.0:
                lambda_por_perfil[perfil] = _LAM_MAX
            else:
                target_vol = tolerancia * vol_factor
                lam_cal = calibrar_lambda(train_init_r, target_vol, max_weight=max_weight)
                # si toca el techo la vol objetivo es menor que la min alcanzable:
                # usar lambda maximo (minima varianza posible)
                lambda_por_perfil[perfil] = lam_cal
                if verbose:
                    nota = " (min varianza)" if lam_cal >= _LAM_MAX * 0.99 else ""
                    print(f"    {perfil:<20} lambda={lam_cal:.3f}  (target vol={target_vol:.1%}){nota}")
    else:
        lambda_por_perfil = {perfil: lam for perfil in RISK_PROFILES}

    mu_bl_cache: dict = {}
    if use_bl:
        if verbose:
            print("  Calculando Black-Litterman por mes...", end="", flush=True)

        for mes in meses:
            train   = _train_window(pr, mes, train_years)
            train_r = train.clip(lower=-_CLIP, upper=_CLIP)
            if bl_method == "robust_factor":
                mu_bl_cache[mes] = robust_factor_black_litterman(
                    train_r, market_caps=market_caps,
                    conf_base=conf_base, lookback=lookback, skip=skip,
                )
            elif bl_method == "asset_momentum":
                mu_bl_cache[mes] = black_litterman(
                    train_r, market_caps=market_caps,
                    conf_base=conf_base, lookback=lookback, skip=skip,
                )
            else:
                raise ValueError(f"bl_method no soportado: {bl_method}")

        if verbose:
            print(f" {len(mu_bl_cache)} meses listos.")
    elif verbose:
        print("  Sin Black-Litterman: usando media historica mensualizada en cada ventana.")

    resultados = {
        perfil: run_backtest(
            pr, dv,
            perfil=perfil,
            capital_inicial=capital_inicial,
            eval_start=eval_start,
            eval_end=eval_end,
            train_years=train_years,
            mu_bl_cache=mu_bl_cache,
            lam=lambda_por_perfil[perfil],
            max_weight=max_weight,
            commission=commission,
            vol_factor=vol_factor,
            full_invest=full_invest,
            p1_lineal=p1_lineal,
            use_bl=use_bl,
            seed=seed,
        )
        for perfil in RISK_PROFILES
    }
    if return_cache:
        return resultados, mu_bl_cache, list(tickers)
    return resultados


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


def run_benchmark_markowitz(
    price_returns: pd.DataFrame,
    div_yields: pd.DataFrame,
    capital_inicial: float = 1_000_000,
    eval_start: str = "2019-01-01",
    eval_end: str = "2024-12-31",
    lam: float = _LAM,
    max_weight: float = MAX_WEIGHT,
    commission: float = COMMISSION,
    vol_factor: float = _VOL_FACTOR,
    full_invest: bool = False,
    seed: int = 42,
    verbose: bool = True,
) -> "dict[str, pd.DataFrame]":
    """
    Caso base metodológico: Markowitz puro con media histórica (sin Black-Litterman).
    Permite comparar directamente contra el modelo completo BL+Markowitz.
    Corre los mismos 5 perfiles con los mismos parámetros excepto mu.
    """
    meses = pd.date_range(eval_start, eval_end, freq="MS")

    train_init = _train_window(price_returns, meses[0], TRAIN_WINDOW_YEARS)
    train_init = train_init.dropna(axis=1, how="any")
    tickers    = list(train_init.columns)

    pr = price_returns[tickers]
    dv = div_yields.reindex(columns=tickers).fillna(0)

    if verbose:
        print("  Caso base Markowitz (sin BL)...")

    resultados = {}
    for perfil in RISK_PROFILES:
        tolerancia = RISK_PROFILES[perfil]
        rng        = np.random.default_rng(seed)
        state      = PortfolioState.crear(
            tickers, capital_inicial,
            _optimizar(_train_window(pr, meses[0], TRAIN_WINDOW_YEARS),
                       tolerancia, perfil, use_bl=False,
                       lam=lam, max_weight=max_weight, vol_factor=vol_factor,
                       full_invest=full_invest)
        )

        registros = []
        for mes in meses:
            mes_end = mes + pd.offsets.MonthEnd(0)
            pr_mes  = pr.loc[mes:mes_end]
            dv_mes  = dv.loc[mes:mes_end]
            if pr_mes.empty:
                continue

            train     = _train_window(pr, mes, TRAIN_WINDOW_YEARS)
            train_r   = train.clip(lower=-_CLIP, upper=_CLIP)
            mu_sample = train_r.mean().values   # media histórica para P2
            pesos_opt = _optimizar(train, tolerancia, perfil, use_bl=False,
                                   lam=lam, max_weight=max_weight, vol_factor=vol_factor,
                                   full_invest=full_invest)

            metricas = paso_mensual(
                state, pr_mes, dv_mes, pesos_opt, tolerancia,
                mu_bl=mu_sample,
                commission_rate=commission,
                rng=rng,
            )
            metricas["fecha"]  = mes
            metricas["perfil"] = perfil
            registros.append(metricas)

            if metricas["abandona"]:
                break

        resultados[perfil] = pd.DataFrame(registros).set_index("fecha")
        if verbose:
            print(f"    {perfil}: {len(registros)} meses")

    return resultados


def run_monte_carlo(
    price_returns: pd.DataFrame,
    div_yields: pd.DataFrame,
    n_clientes: int = 500,
    mu_bl_cache: dict | None = None,
    tickers: list | None = None,
    capital_inicial: float = 1_000_000,
    eval_start: str = "2019-01-01",
    eval_end: str = "2024-12-31",
    lam: float = _LAM,
    max_weight: float = MAX_WEIGHT,
    commission: float = COMMISSION,
    vol_factor: float = _VOL_FACTOR,
    full_invest: bool = False,
    p1_lineal: bool = False,
    seed_base: int = 0,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Simula N clientes independientes por perfil.

    Los pesos óptimos se precomputan una sola vez (un Gurobi por mes/perfil).
    Lo único aleatorio entre clientes son las decisiones de P1 y P2.
    Retorna dict perfil -> DataFrame con una fila por cliente.
    """
    pr = price_returns[tickers] if tickers else price_returns
    dv = div_yields.reindex(columns=pr.columns).fillna(0)
    meses = pd.date_range(eval_start, eval_end, freq="MS")
    perfiles = list(RISK_PROFILES.keys())

    # Precomputar pesos óptimos para cada mes y perfil (único bloque Gurobi)
    if verbose:
        print("  Precomputando pesos optimos por mes/perfil...", end="", flush=True)
    pesos_cache: dict = {}
    for mes in meses:
        mu_bl = mu_bl_cache.get(mes) if mu_bl_cache else None
        train = _train_window(pr, mes, TRAIN_WINDOW_YEARS)
        pesos_cache[mes] = {
            perfil: _optimizar(train, RISK_PROFILES[perfil], perfil,
                               mu_bl=mu_bl, lam=lam, max_weight=max_weight,
                               vol_factor=vol_factor, full_invest=full_invest)
                for perfil in perfiles
        }
    if verbose:
        print(f" listo.")

    # Simular N clientes por perfil
    resultados: dict[str, list] = {p: [] for p in perfiles}

    for i in range(n_clientes):
        for j, perfil in enumerate(perfiles):
            tolerancia = RISK_PROFILES[perfil]
            rng = np.random.default_rng(seed_base + i * 17 + j * 3)

            state = PortfolioState.crear(
                list(pr.columns), capital_inicial, pesos_cache[meses[0]][perfil]
            )

            meses_activos = 0
            abandona = False
            rebalanceos = []
            comision_acum = 0.0

            for mes in meses:
                mes_end = mes + pd.offsets.MonthEnd(0)
                pr_mes = pr.loc[mes:mes_end]
                dv_mes = dv.loc[mes:mes_end]
                if pr_mes.empty:
                    continue

                m = paso_mensual(
                    state, pr_mes, dv_mes,
                    pesos_cache[mes][perfil], tolerancia,
                    mu_bl=mu_bl_cache.get(mes) if mu_bl_cache else None,
                    commission_rate=commission,
                    rng=rng,
                    p1_lineal=p1_lineal,
                )
                meses_activos += 1
                rebalanceos.append(m["rebalanceo"])
                comision_acum += m["comision"]

                if m["abandona"]:
                    abandona = True
                    break

            n = max(meses_activos, 1)
            retorno = (state.valor / capital_inicial) ** (12 / n) - 1

            resultados[perfil].append({
                "cliente":              i,
                "abandona":             abandona,
                "meses_activos":        meses_activos,
                "valor_final":          round(state.valor, 2),
                "retorno_anual":        round(retorno, 4),
                "frecuencia_rebalanceo": round(float(np.mean(rebalanceos)), 4),
                "comision_total":       round(comision_acum, 2),
            })

        if verbose and (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_clientes} clientes simulados")

    return {perfil: pd.DataFrame(rows) for perfil, rows in resultados.items()}


def resumen_monte_carlo(mc: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Agrega los resultados de run_monte_carlo en una tabla resumen por perfil.
    """
    rows = []
    for perfil, df in mc.items():
        completan = df[~df["abandona"]]
        rows.append({
            "perfil":                perfil,
            "tasa_abandono":         round(df["abandona"].mean(), 4),
            "meses_activos_prom":    round(df["meses_activos"].mean(), 1),
            "retorno_prom":          round(df["retorno_anual"].mean(), 4),
            "retorno_std":           round(df["retorno_anual"].std(), 4),
            "retorno_si_completa":   round(completan["retorno_anual"].mean(), 4) if len(completan) else float("nan"),
            "frec_rebalanceo_prom":  round(df["frecuencia_rebalanceo"].mean(), 4),
            "comision_prom":         round(df["comision_total"].mean(), 2),
            "n_clientes":            len(df),
        })
    return pd.DataFrame(rows).set_index("perfil")


def calcular_ic(
    price_returns: pd.DataFrame,
    mu_bl_cache: dict,
    tickers: list,
    eval_start: str = "2019-01-01",
    eval_end: str = "2024-12-31",
) -> dict:
    """
    Information Coefficient (IC): correlación de Spearman entre retornos
    predichos por BL al inicio de cada mes y los retornos reales de ese mes.

    IC > 0  → el modelo predice correctamente el ranking entre acciones
    IC ~ 0  → sin poder predictivo
    ICIR    → IC medio / desv. estándar del IC (consistencia de la señal)
    """
    pr    = price_returns[tickers]
    meses = pd.date_range(eval_start, eval_end, freq="MS")

    ics = []
    for mes in meses:
        mu_pred = mu_bl_cache.get(mes)
        if mu_pred is None:
            continue
        mes_end     = mes + pd.offsets.MonthEnd(0)
        pr_mes      = pr.loc[mes:mes_end]
        if pr_mes.empty:
            continue
        ret_real    = (1 + pr_mes).prod() - 1
        corr, _     = spearmanr(mu_pred, ret_real.values)
        if not np.isnan(corr):
            ics.append(corr)

    ics  = np.array(ics)
    mean = float(np.mean(ics))
    std  = float(np.std(ics))
    return {
        "IC_medio": round(mean, 4),
        "IC_std":   round(std, 4),
        "ICIR":     round(mean / std, 4) if std > 1e-8 else 0.0,
        "IC_pos":   round(float(np.mean(ics > 0)), 4),  # fraccion de meses con IC > 0
        "n_meses":  len(ics),
    }


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
