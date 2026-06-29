"""
CVaR portfolio optimization via LP (Rockafellar & Uryasev, 2000).

Maximize expected return subject to CVaR_alpha(w) <= cvar_limit.

LP formulation with variable vector x = [w(N), v(1), u(T)]:
    max   mu' w
    s.t.  v + 1/(T(1-alpha)) * sum(u) <= cvar_limit    [CVaR limit]
          u_t + r_t' w + v >= 0  for all t              [scenario constraints]
          u_t >= 0                for all t
          sum(w) = 1  (or <= 1)
          0 <= w_i <= max_weight
"""

import numpy as np
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB

_CLIP = 0.50          # winsorización igual que runner.py
_VOL_FACTOR = 0.50    # misma fracción de tolerancia que Markowitz
_CVAR_SCALE = 2.063   # CVaR_0.95 / sigma bajo normalidad = phi(1.645)/0.05


def cvar_limit_para_perfil(tolerancia: float) -> float:
    """
    Devuelve el límite diario de CVaR_0.95 equivalente al constraint de
    volatilidad de Markowitz (Tol * VOL_FACTOR anualizado).

    Bajo normalidad: CVaR_0.95 ≈ 2.063 * sigma, por lo que:
        cvar_limit_diario = Tol * VOL_FACTOR * CVAR_SCALE / sqrt(252)
    """
    if tolerancia <= 0:
        return 0.0
    return tolerancia * _VOL_FACTOR * _CVAR_SCALE / np.sqrt(252)


def optimizar_cvar(
    train_returns,
    cvar_limit: float,
    perfil: str,
    mu_personalizado=None,
    max_weight: float = 0.025,
    alpha: float = 0.95,
    full_invest: bool = False,
) -> np.ndarray:
    """
    Resuelve el problema de optimización CVaR usando Gurobi LP.

    Parámetros
    ----------
    train_returns  : DataFrame (T x N), retornos diarios de entrenamiento
    cvar_limit     : límite diario de CVaR_alpha (en fracción, e.g. 0.003)
    perfil         : string para nombrar el modelo Gurobi
    mu_personalizado : vector de retornos esperados (BL u otro); si None usa media histórica
    alpha          : nivel de confianza (default 0.95 → CVaR sobre el peor 5%)
    full_invest    : si True, sum(w) = 1; si False, sum(w) <= 1

    Retorna
    -------
    w_opt : np.ndarray de forma (N,) con los pesos óptimos
    """
    N = len(train_returns.columns)

    if cvar_limit <= 0:
        return np.zeros(N)

    R = train_returns.clip(-_CLIP, _CLIP).values  # T x N
    T, _ = R.shape
    mu = mu_personalizado if mu_personalizado is not None else R.mean(axis=0)

    model = gp.Model(f"CVaR_{perfil}")
    model.Params.OutputFlag = 0

    # Variables: x = [w(0..N-1) | v(N) | u(N+1..N+T)]
    lb = np.concatenate([np.zeros(N), [-GRB.INFINITY], np.zeros(T)])
    ub = np.concatenate([np.full(N, max_weight), [GRB.INFINITY], np.full(T, GRB.INFINITY)])
    x = model.addMVar(N + 1 + T, lb=lb, ub=ub, name="x")

    # Restricciones de escenario: u_t + r_t'w + v >= 0  para todo t
    # Forma matricial: [R | ones_col | I_T] x >= 0   (T x (N+1+T))
    A_scen = sp.hstack([
        sp.csr_matrix(R),
        sp.csr_matrix(np.ones((T, 1))),
        sp.eye(T, format="csr"),
    ], format="csr")
    model.addMConstr(A_scen, x, ">", np.zeros(T))

    # Restricción CVaR: v + 1/(T(1-alpha)) * sum(u) <= cvar_limit
    a_lim = np.zeros(N + 1 + T)
    a_lim[N] = 1.0
    a_lim[N + 1:] = 1.0 / (T * (1.0 - alpha))
    model.addMConstr(
        sp.csr_matrix(a_lim.reshape(1, -1)), x, "<", np.array([cvar_limit])
    )

    # Restricción de presupuesto
    a_budget = np.zeros(N + 1 + T)
    a_budget[:N] = 1.0
    budget_sense = "=" if full_invest else "<"
    model.addMConstr(
        sp.csr_matrix(a_budget.reshape(1, -1)), x, budget_sense, np.array([1.0])
    )

    # Objetivo: maximizar mu'w
    c = np.concatenate([mu, [0.0], np.zeros(T)])
    model.setObjective(c @ x, GRB.MAXIMIZE)

    model.optimize()

    if model.Status == GRB.OPTIMAL:
        return x.X[:N]

    # Fallback: pesos iguales si el LP es infactible
    return np.ones(N) / N
