import numpy as np


def p1_abandono(
    drawdown: float,
    tolerancia: float,
    k: float = 10.0,
    exceso_critico: float = 0.15,
) -> float:
    """
    Probabilidad de que el cliente abandone el sistema.

    drawdown        : pérdida actual desde el peak (fracción positiva, e.g. 0.10 = 10%)
    tolerancia      : pérdida máxima tolerable del perfil (e.g. 0.05 para conservador)
    k               : pendiente de la sigmoide sobre el exceso
    exceso_critico  : exceso sobre tolerancia donde P1 = 0.5 (default 15%)

    Mientras drawdown <= tolerancia, P1 = 0: el cliente acepta pérdidas dentro de
    su tolerancia declarada y no considera abandonar.
    Cuando drawdown > tolerancia, P1 sube como sigmoide del exceso:
      P1 = 0.5 cuando exceso == exceso_critico
      P1 -> 1  cuando exceso >> exceso_critico
    """
    if tolerancia == 0.0:
        return 0.0  # todo en caja chica: pérdidas de comisión no generan abandono
    exceso = drawdown - tolerancia
    if exceso <= 0:
        return 0.0
    return 1.0 / (1.0 + np.exp(-k * (exceso - exceso_critico)))


def p2_aceptacion(turnover: float, umbral: float = 0.10, k: float = 20.0) -> float:
    """
    Probabilidad de que el cliente acepte la recomendación de rebalanceo mensual.

    turnover : fracción del portafolio que cambiaría si se rebalancea (L1/2, entre 0 y 1)
    umbral   : drift mínimo a partir del cual el cliente empieza a aceptar (default 10%)
    k        : pendiente de la sigmoide

    P2 -> 0   cuando el portafolio está cerca del óptimo  (cliente no ve urgencia)
    P2 = 0.5 cuando turnover == umbral
    P2 -> 1   cuando el drift es grande  (cliente ve valor en rebalancear)
    """
    return 1.0 / (1.0 + np.exp(-k * (turnover - umbral)))
