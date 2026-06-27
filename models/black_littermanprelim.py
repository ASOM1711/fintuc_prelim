import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def _momentum_signal(train_returns: pd.DataFrame, lookback: int, skip: int) -> np.ndarray:
    """
    Retorno diario equivalente sobre la ventana [-(lookback+skip) : -skip].
    Excluir los últimos `skip` días evita capturar la reversión de corto plazo.
    Se convierte a escala diaria para que sea comparable con Π.
    """
    n_obs = len(train_returns)
    start = max(0, n_obs - lookback - skip)
    end   = max(0, n_obs - skip)
    if end <= start:
        return np.zeros(len(train_returns.columns))
    window   = train_returns.iloc[start:end]
    n_window = end - start
    Q_total  = (1 + window).prod().values - 1          # retorno total acumulado
    return (1 + Q_total) ** (1.0 / n_window) - 1       # convertir a escala diaria


def _safe_zscore(values: np.ndarray, clip: float = 3.0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    std = float(values.std())
    if std < 1e-12:
        return np.zeros_like(values)
    z = (values - values.mean()) / std
    return np.clip(z, -clip, clip)


def _normalize_view_weights(signal: np.ndarray, long_short: bool = True) -> np.ndarray:
    signal = np.nan_to_num(np.asarray(signal, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if long_short:
        signal = signal - signal.mean()
    denom = float(np.abs(signal).sum())
    if denom < 1e-12:
        return np.zeros_like(signal)
    return signal / denom


def _window(train_returns: pd.DataFrame, lookback: int, skip: int = 0) -> pd.DataFrame:
    n_obs = len(train_returns)
    start = max(0, n_obs - lookback - skip)
    end = n_obs - skip if skip > 0 else n_obs
    end = max(0, end)
    if end <= start:
        return train_returns.iloc[0:0]
    return train_returns.iloc[start:end]


def _robust_prior_weights(
    train_returns: pd.DataFrame,
    market_caps: pd.Series | None = None,
    cap_weight: float = 0.50,
    inv_vol_weight: float = 0.35,
) -> np.ndarray:
    """
    Construye un prior que no dependa completamente de market cap.

    Si hay market caps, usa una mezcla market-cap/equal/inverse-vol.
    Si no hay market caps, usa equal/inverse-vol. Esto evita que el prior quede
    mal especificado cuando el universo fue elegido por market cap, liquidez o
    una mezcla arbitraria.
    """
    tickers = train_returns.columns
    n = len(tickers)
    equal = np.ones(n) / n

    vol = train_returns.std().replace(0, np.nan).values.astype(float)
    inv_vol = 1.0 / np.nan_to_num(vol, nan=np.nanmedian(vol))
    inv_vol = np.clip(inv_vol, 0, np.nanpercentile(inv_vol, 95))
    inv_vol = inv_vol / inv_vol.sum() if inv_vol.sum() > 0 else equal

    if market_caps is not None:
        caps = market_caps.reindex(tickers).fillna(0).values.astype(float)
        cap_prior = caps / caps.sum() if caps.sum() > 0 else equal
        equal_weight = max(0.0, 1.0 - cap_weight - inv_vol_weight)
        w = cap_weight * cap_prior + inv_vol_weight * inv_vol + equal_weight * equal
    else:
        w = inv_vol_weight * inv_vol + (1.0 - inv_vol_weight) * equal

    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    return w / w.sum() if w.sum() > 0 else equal


def _factor_views(
    train_returns: pd.DataFrame,
    lookback: int,
    skip: int,
    include_reversal: bool,
    view_cap_daily: float,
    view_shrink: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Crea views sobre factores, no sobre cada accion.

    P tiene pocas filas: mercado, momentum, low-vol y opcionalmente reversal.
    Q es el retorno esperado diario del factor, capado y shrunk para evitar
    que un periodo extremo domine todo el posterior.
    """
    n = len(train_returns.columns)
    views: list[np.ndarray] = []
    q_values: list[float] = []
    names: list[str] = []

    recent = _window(train_returns, lookback, skip)
    if recent.empty:
        recent = train_returns
    mean_recent = recent.mean().values

    market_w = np.ones(n) / n
    views.append(market_w)
    q_values.append(float(np.clip(view_shrink * recent.mean(axis=1).mean(), -view_cap_daily, view_cap_daily)))
    names.append("market")

    momentum = _momentum_signal(train_returns, lookback=lookback, skip=skip)
    mom_w = _normalize_view_weights(_safe_zscore(momentum), long_short=True)
    if np.abs(mom_w).sum() > 0:
        views.append(mom_w)
        q_values.append(float(np.clip(view_shrink * np.dot(mom_w, mean_recent), -view_cap_daily, view_cap_daily)))
        names.append("momentum")

    vol = recent.std().values
    low_vol_signal = -_safe_zscore(vol)
    low_vol_w = _normalize_view_weights(low_vol_signal, long_short=True)
    if np.abs(low_vol_w).sum() > 0:
        views.append(low_vol_w)
        q_values.append(float(np.clip(view_shrink * np.dot(low_vol_w, mean_recent), -view_cap_daily, view_cap_daily)))
        names.append("low_vol")

    if include_reversal:
        rev_window = _window(train_returns, lookback=min(21, max(1, lookback // 6)), skip=0)
        if not rev_window.empty:
            short_ret = (1 + rev_window).prod().values - 1
            reversal_w = _normalize_view_weights(-_safe_zscore(short_ret), long_short=True)
            if np.abs(reversal_w).sum() > 0:
                views.append(reversal_w)
                q_values.append(float(np.clip(view_shrink * np.dot(reversal_w, mean_recent), -view_cap_daily, view_cap_daily)))
                names.append("reversal")

    return np.vstack(views), np.asarray(q_values), names


def black_litterman(
    train_returns: pd.DataFrame,
    market_caps: pd.Series | None = None,
    delta: float = 2.5,
    tau: float | None = None,
    lookback: int = 252,
    skip: int = 21,
    conf_base: float = 0.05,
) -> np.ndarray:
    """
    Retornos esperados diarios via Black-Litterman con views absolutas de momentum.

    Parámetros
    ----------
    train_returns : DataFrame (T x N) de retornos diarios
    market_caps   : Series con capitalización de mercado por ticker.
                    Si es None, usa pesos iguales como proxy de equilibrio.
    delta         : aversión al riesgo implícita del mercado (default 2.5)
    tau           : incertidumbre del prior; si es None usa 1/T
    lookback      : días de historia para el momentum (default 252 ≈ 1 año)
    skip          : días recientes a excluir del momentum (default 21 ≈ 1 mes)
    conf_base     : fracción de τΣ usada como varianza de cada view (default 0.05)

    Retorna
    -------
    mu_bl : ndarray (N,) — retorno diario esperado ajustado por Black-Litterman
    """
    T, N = train_returns.shape
    tickers = train_returns.columns

    # 1. Covarianza LedoitWolf (garantiza matriz positivo-definida)
    lw = LedoitWolf()
    lw.fit(train_returns.values)
    Sigma = lw.covariance_  # (N x N), escala diaria

    # 2. Pesos de mercado para el equilibrio
    if market_caps is not None:
        caps  = market_caps.reindex(tickers).fillna(0).values.astype(float)
        total = caps.sum()
        w_mkt = caps / total if total > 0 else np.ones(N) / N
    else:
        w_mkt = np.ones(N) / N

    # 3. Retornos de equilibrio: Π = δ · Σ · w_mkt  (escala diaria)
    tau = tau if tau is not None else 1.0 / T
    Pi  = delta * Sigma @ w_mkt  # (N,)

    # 4. Views de momentum en escala diaria (P = I: una view absoluta por activo)
    Q = _momentum_signal(train_returns, lookback, skip)  # (N,) escala diaria

    # 5. Incertidumbre de views: Ω = conf_base · diag(τΣ)
    tau_Sigma = tau * Sigma
    Omega     = np.diag(conf_base * np.diag(tau_Sigma))  # (N x N) diagonal

    # 6. Fórmula Black-Litterman — forma Woodbury (P = I)
    #    μ_BL = Π + τΣ (τΣ + Ω)⁻¹ (Q - Π)
    #    Evita invertir matrices mal condicionadas; (τΣ + Ω) está regularizada por Ω.
    M     = tau_Sigma + Omega                           # (N x N), bien condicionada
    mu_bl = Pi + tau_Sigma @ np.linalg.solve(M, Q - Pi)  # (N,)

    return mu_bl


def robust_factor_black_litterman(
    train_returns: pd.DataFrame,
    market_caps: pd.Series | None = None,
    delta: float = 2.5,
    tau: float | None = None,
    lookback: int = 126,
    skip: int = 21,
    conf_base: float = 1.0,
    cap_weight: float = 0.50,
    inv_vol_weight: float = 0.35,
    include_reversal: bool = True,
    view_cap_daily: float = 0.0015,
    view_shrink: float = 0.50,
    hist_blend: float = 0.15,
) -> np.ndarray:
    """
    Black-Litterman robusto basado en factores.

    Diferencias con el BL preliminar:
    - No necesita que el universo sea una muestra perfecta del mercado.
    - Si hay market caps los usa, pero los mezcla con equal-weight e inverse-vol.
    - Si no hay market caps, mantiene un prior estable equal/inverse-vol.
    - Usa pocas views factoriales en vez de una view por accion.
    - Capa y shrinkea las views para reducir sobreajuste.
    """
    T, _ = train_returns.shape
    train_r = train_returns.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if train_r.empty:
        return np.zeros(len(train_returns.columns))

    lw = LedoitWolf()
    lw.fit(train_r.values)
    sigma = lw.covariance_

    tau = tau if tau is not None else 1.0 / max(T, 1)
    tau_sigma = tau * sigma

    w_prior = _robust_prior_weights(
        train_r,
        market_caps=market_caps,
        cap_weight=cap_weight,
        inv_vol_weight=inv_vol_weight,
    )
    pi = delta * sigma @ w_prior

    p_matrix, q_values, _ = _factor_views(
        train_r,
        lookback=lookback,
        skip=skip,
        include_reversal=include_reversal,
        view_cap_daily=view_cap_daily,
        view_shrink=view_shrink,
    )

    view_cov = p_matrix @ tau_sigma @ p_matrix.T
    omega_diag = conf_base * np.maximum(np.diag(view_cov), 1e-10)
    omega = np.diag(omega_diag)

    middle = view_cov + omega
    adjustment = tau_sigma @ p_matrix.T @ np.linalg.solve(middle, q_values - p_matrix @ pi)
    mu_bl = pi + adjustment

    if hist_blend > 0:
        hist_mu = train_r.mean().values
        hist_mu = np.clip(hist_mu, -view_cap_daily, view_cap_daily)
        mu_bl = (1.0 - hist_blend) * mu_bl + hist_blend * hist_mu

    return np.nan_to_num(mu_bl, nan=0.0, posinf=0.0, neginf=0.0)
