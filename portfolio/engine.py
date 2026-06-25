from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config import COMMISSION
from portfolio.probabilities import p1_abandono, p2_aceptacion


@dataclass
class PortfolioState:
    """Estado completo del portafolio en un momento dado."""
    tickers: list[str]
    valor: float          # valor total en $
    pesos: np.ndarray     # fracción del valor invertida en cada activo
    caja_chica: float     # fracción del valor en efectivo (dividendos + no invertido)
    peak: float           # máximo valor histórico (para calcular drawdown)

    @classmethod
    def crear(cls, tickers: list[str], capital: float, pesos: np.ndarray) -> "PortfolioState":
        """Inicializa el portafolio con un capital y pesos óptimos iniciales."""
        pesos = np.clip(pesos, 0, None)
        return cls(
            tickers=list(tickers),
            valor=capital,
            pesos=pesos,
            caja_chica=max(0.0, 1.0 - pesos.sum()),
            peak=capital,
        )

    def drawdown(self) -> float:
        """Pérdida actual desde el peak (positivo = pérdida, e.g. 0.10 = 10%)."""
        return max(0.0, 1.0 - self.valor / self.peak)

    def actualizar_dia(self, price_ret: np.ndarray, div_yield: np.ndarray) -> None:
        """
        Aplica retornos de precio y dividendos de un día.
        Los dividendos se acumulan en caja chica (no se reinvierten en acciones).
        """
        val_stocks = self.pesos * self.valor
        val_cash   = self.caja_chica * self.valor

        # dividendos salen de la posición en acciones y van a caja
        div_cash   = (val_stocks * div_yield).sum()
        val_stocks = val_stocks * (1 + price_ret)
        val_cash  += div_cash

        nuevo_valor     = val_stocks.sum() + val_cash
        self.pesos      = val_stocks / nuevo_valor
        self.caja_chica = val_cash / nuevo_valor
        self.valor      = nuevo_valor

        if nuevo_valor > self.peak:
            self.peak = nuevo_valor

    def cobrar_comision_mensual(self, commission_rate: float = COMMISSION) -> float:
        """
        Descuenta la comisión de gestión mensual (commission_rate / 12 sobre AUM).
        Retorna el monto descontado en $.

        El peak se reduce en el mismo monto para que la comisión no afecte
        el cálculo de drawdown — el cliente entiende que hay un fee de gestión
        y el abandono solo debe reflejar pérdidas de mercado.
        """
        monto       = self.valor * (commission_rate / 12)
        self.valor -= monto
        self.peak  -= monto  # exime la comisión del drawdown
        return monto

    def aplicar_pesos(self, pesos_nuevos: np.ndarray) -> float:
        """
        Actualiza los pesos al nuevo óptimo (rebalanceo).
        Retorna el turnover: fracción del portafolio que cambia de manos.
        """
        turnover        = float(np.abs(pesos_nuevos - self.pesos).sum() / 2)
        self.pesos      = pesos_nuevos.copy()
        self.caja_chica = max(0.0, 1.0 - pesos_nuevos.sum())
        return turnover


def paso_mensual(
    state: PortfolioState,
    price_ret_mes: pd.DataFrame,
    div_yield_mes: pd.DataFrame,
    pesos_optimos: np.ndarray,
    tolerancia_perfil: float,
    mu_bl: Optional[np.ndarray] = None,
    commission_rate: float = COMMISSION,
    rng: Optional[np.random.Generator] = None,
    p1_lineal: bool = False,
) -> dict:
    """
    Ejecuta un mes completo del portafolio:
      1. Aplica retornos diarios (precio + dividendos).
      2. Cobra comisión mensual de gestión.
      3. Decide si el cliente acepta el rebalanceo (P2).
      4. Calcula la probabilidad de abandono del cliente (P1).

    P2 se basa en el retorno anual esperado del portafolio propuesto vs la tolerancia.
    P1 se basa en el drawdown actual vs la tolerancia del perfil.
    """
    if rng is None:
        rng = np.random.default_rng()

    tickers = state.tickers
    pr_mes  = price_ret_mes.reindex(columns=tickers).fillna(0).values
    dv_mes  = div_yield_mes.reindex(columns=tickers).fillna(0).values

    for t in range(len(pr_mes)):
        state.actualizar_dia(pr_mes[t], dv_mes[t])

    comision = state.cobrar_comision_mensual(commission_rate)

    # 3. P2: ¿acepta el cliente el rebalanceo?
    # retorno esperado anual = dot(mu_bl, pesos) * 252 (mu_bl en escala diaria)
    if mu_bl is not None and pesos_optimos.sum() > 0:
        ret_diario_esperado = float(np.dot(mu_bl, pesos_optimos))
        retorno_esperado    = (1 + ret_diario_esperado) ** 252 - 1
    else:
        retorno_esperado = 0.0

    turnover   = float(np.abs(pesos_optimos - state.pesos).sum() / 2)
    prob_p2    = p2_aceptacion(retorno_esperado, tolerancia_perfil)
    rebalanceo = bool(rng.random() < prob_p2)
    if rebalanceo:
        state.aplicar_pesos(pesos_optimos)

    # 4. P1: ¿abandona el cliente?
    dd       = state.drawdown()
    prob_p1  = p1_abandono(dd, tolerancia_perfil, lineal=p1_lineal)
    abandona = bool(rng.random() < prob_p1)

    return {
        "valor":             state.valor,
        "drawdown":          dd,
        "turnover":          turnover,
        "retorno_esperado":  retorno_esperado,
        "prob_p2":           prob_p2,
        "prob_p1":           prob_p1,
        "rebalanceo":        rebalanceo,
        "abandona":          abandona,
        "caja_chica":        state.caja_chica,
        "comision":          comision,
    }
