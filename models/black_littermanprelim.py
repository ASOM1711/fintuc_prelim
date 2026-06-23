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
