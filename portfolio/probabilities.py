import numpy as np


def p1_abandono(drawdown: float, tolerancia: float, lineal: bool = False) -> float:
    """
    Probabilidad de que el cliente abandone el sistema.

    Modo sigmoid (default, lineal=False):
      P1 = 0                                         si drawdown <= tolerancia
      P1 = 1 / (1 + exp(-((drawdown - tolerancia) * 100)))  si drawdown > tolerancia
      P1 = 50% cuando drawdown = tolerancia

    Modo lineal (lineal=True):
      P1 = 0                                         si drawdown <= tolerancia
      P1 = min(1.0, (drawdown - tolerancia) * 15)   si drawdown > tolerancia
      → +1pp sobre tolerancia = 15% abandono
      → +2pp sobre tolerancia = 30% abandono
      → +6.67pp sobre tolerancia = 100% abandono

    Caso especial: tolerancia == 0 → P1 = 0 siempre.
    """
    if tolerancia == 0.0 or drawdown <= tolerancia:
        return 0.0
    exceso = drawdown - tolerancia
    if lineal:
        return min(1.0, exceso * 15)
    return 1.0 / (1.0 + np.exp(-(exceso * 100)))


def p2_aceptacion(retorno_esperado: float, tolerancia: float) -> float:
    """
    Probabilidad de que el cliente acepte la recomendación de rebalanceo.

    P2 = 1 / (1 + exp(-(x2 - x̂2)))
      x2  = retorno anual esperado del portafolio propuesto, en puntos porcentuales
      x̂2  = tolerancia del perfil (máxima pérdida esperada), en puntos porcentuales

    P2 = 0.5 cuando el retorno esperado iguala la tolerancia del perfil.
    P2 > 0.5 cuando el retorno esperado supera la tolerancia (cliente ve valor).
    P2 < 0.5 cuando el retorno esperado es menor que la tolerancia (cliente duda).
    """
    return 1.0 / (1.0 + np.exp(-(retorno_esperado * 100 - tolerancia * 100)))
