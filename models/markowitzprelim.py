import numpy as np
import gurobipy as gp
from gurobipy import GRB
from sklearn.covariance import LedoitWolf


def calibrar_lambda(
    train_returns,
    target_vol_anual: float,
    max_weight: float = 0.15,
    n_iter: int = 30,
) -> float:
    """
    Calibra lambda para que el portafolio óptimo (sin restricción QCQP)
    tenga una volatilidad anualizada igual a target_vol_anual.

    Usa búsqueda binaria en escala log sobre lambda ∈ [0.01, 5000].
    La volatilidad del portafolio es monótonamente decreciente en lambda:
    mayor lambda → más penalización de varianza → portafolio más conservador.

    Retorna el lambda calibrado.
    """
    mu    = train_returns.mean().values
    lw    = LedoitWolf()
    lw.fit(train_returns.values)
    Sigma = lw.covariance_
    n     = len(train_returns.columns)

    def _vol(lam: float) -> float:
        model = gp.Model("calib")
        model.Params.OutputFlag = 0
        w = model.addVars(n, lb=0, ub=max_weight, name="w")
        model.addConstr(gp.quicksum(w[i] for i in range(n)) == 1)
        port_var = gp.quicksum(
            w[i] * Sigma[i, j] * w[j] for i in range(n) for j in range(n)
        )
        port_ret = gp.quicksum(mu[i] * w[i] for i in range(n))
        model.setObjective(port_ret - lam * port_var, GRB.MAXIMIZE)
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            return float("inf")
        w_vals = np.array([w[i].X for i in range(n)])
        return float(np.sqrt(252 * w_vals @ Sigma @ w_vals))

    log_min, log_max = np.log(0.01), np.log(5000.0)
    for _ in range(n_iter):
        log_mid = (log_min + log_max) / 2
        vol = _vol(np.exp(log_mid))
        if vol > target_vol_anual:
            log_min = log_mid   # aumentar lambda para reducir vol
        else:
            log_max = log_mid   # reducir lambda para aumentar vol

    return float(np.exp((log_min + log_max) / 2))


def optimizar_markowitz(train_returns, lam, perdida_max_anual, perfil,
                        mu_personalizado=None, max_weight=0.15,
                        lambda_l2=0, usar_l2=False, full_invest=False):
    """
    Optimiza un portafolio de Markowitz para un perfil de riesgo dado.

    Parámetros
    ----------
    train_returns     : DataFrame (fechas x tickers) con retornos diarios
    lam               : parámetro de aversión al riesgo (lambda)
    perdida_max_anual : tolerancia máxima de pérdida anual del perfil (ej. 0.05)
    perfil            : nombre del perfil (string, para nombrar el modelo)
    mu_personalizado  : vector de retornos esperados (Black-Litterman u otro).
                        Si es None se usa la media histórica.
    max_weight        : peso máximo por activo (default 0.15)
    lambda_l2         : coeficiente de regularización L2
    usar_l2           : si True, agrega penalización L2 al objetivo
    full_invest       : si True, el cliente invierte el 100% (Σw = 1).
                        La caja chica solo acumula dividendos.
                        Si False (default), Σw <= 1 y el resto queda en cash.

    Retorna
    -------
    model : modelo Gurobi resuelto
    w     : variables de decisión (pesos por activo)
    """
    mu = mu_personalizado if mu_personalizado is not None else train_returns.mean().values

    lw = LedoitWolf()
    lw.fit(train_returns.values)
    Sigma = lw.covariance_

    n = len(train_returns.columns)

    model = gp.Model(f"Markowitz_{perfil}")
    model.Params.OutputFlag = 0

    w = model.addVars(n, lb=0, ub=max_weight, name="w")

    # restricción de presupuesto
    if full_invest:
        # 100% invertido: caja chica solo recibe dividendos
        model.addConstr(gp.quicksum(w[i] for i in range(n)) == 1,
                        name="presupuesto")
    else:
        # versión original: lo que no se invierte queda en caja chica
        model.addConstr(gp.quicksum(w[i] for i in range(n)) <= 1,
                        name="presupuesto")

    # varianza diaria del portafolio
    portfolio_variance = gp.quicksum(
        w[i] * Sigma[i, j] * w[j]
        for i in range(n)
        for j in range(n)
    )

    # restricción de riesgo: volatilidad anualizada <= tolerancia del perfil
    # sqrt(252 * w'Σw) <= perdida_max_anual  →  252 * w'Σw <= perdida_max_anual²
    if perdida_max_anual > 0:
        model.addQConstr(
            252 * portfolio_variance <= perdida_max_anual ** 2,
            name="volatilidad_maxima"
        )

    # retorno esperado del portafolio
    portfolio_return = gp.quicksum(mu[i] * w[i] for i in range(n))

    # función objetivo
    if usar_l2:
        l2 = gp.quicksum(w[i] * w[i] for i in range(n))
        objective = portfolio_return - lam * portfolio_variance - lambda_l2 * l2
    else:
        objective = portfolio_return - lam * portfolio_variance

    model.setObjective(objective, GRB.MAXIMIZE)
    model.optimize()

    return model, w
