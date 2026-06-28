"""
Asesor interactivo mes a mes usando el modelo final BL + Markowitz.

El flujo:
1. El usuario elige perfil de riesgo.
2. Parte con un capital inicial de USD 1.000.
3. Antes de cada mes, el modelo calcula un portafolio recomendado.
4. El usuario decide si cambia su portafolio actual por el recomendado.
5. Luego se actualiza el valor con los retornos reales historicos del mes.
6. La terminal muestra portafolio actual vs recomendado, prediccion,
   varianza anual, volatilidad anual y escenarios favorable/desfavorable.

Ejemplo:
    python interactive_portfolio_advisor.py
    python interactive_portfolio_advisor.py --perfil arriesgado --start 2024-01-01 --end 2024-06-30
    python interactive_portfolio_advisor.py --perfil neutro --auto no --max-months 3
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from gurobipy import GRB
from sklearn.covariance import LedoitWolf

import data.loader as loader
from backtesting.runner import (
    _CLIP,
    _calcular_mu_bl,
    _max_weight_for_profile,
    _train_window,
)
from config import (
    BL_CONF_BASE,
    BL_LOOKBACK,
    BL_METHOD,
    BL_SKIP,
    COMMISSION,
    PROFILE_MAX_WEIGHTS,
    RISK_PROFILES,
    TRAIN_WINDOW_YEARS,
)
from models.markowitzprelim import optimizar_markowitz
from portfolio.engine import PortfolioState
from portfolio.probabilities import p1_abandono, p2_aceptacion


CAPITAL_DEFAULT = 1_000.0
START_YEAR_DEFAULT = 2019
END_YEAR_DEFAULT = 2025
LAM_DEFAULT = 0.0
VOL_FACTOR = 0.5


@dataclass
class PortfolioDiagnostics:
    expected_return: float
    variance: float
    volatility: float
    favorable: float
    unfavorable: float
    prob_accept: float


@dataclass
class OptimizationResult:
    weights: np.ndarray | None
    status: int
    is_optimal: bool
    message: str


def _load_universe_with_fallback():
    price_returns, div_yields = loader.load_universe()
    if not price_returns.empty:
        return price_returns, div_yields

    root = Path.cwd()
    candidates = [
        root.parent.parent / "Capstone" / "universo_300_acciones" / "universo_300_acciones",
    ]
    for candidate in candidates:
        if list(candidate.glob("stock_return_*.csv")):
            loader.DATA_DIR = candidate
            return loader.load_universe()

    return price_returns, div_yields


def _load_market_caps():
    try:
        info = loader.load_stock_info()
    except FileNotFoundError:
        return None
    return info["marketCap"] if "marketCap" in info.columns else None


def _format_pct(value: float) -> str:
    return f"{value * 100:,.2f}%"


def _format_usd(value: float) -> str:
    return f"USD {value:,.2f}"


def _portfolio_diagnostics(
    weights: np.ndarray,
    train_returns: pd.DataFrame,
    mu_daily: np.ndarray,
    tolerancia: float,
) -> PortfolioDiagnostics:
    if len(weights) == 0 or float(weights.sum()) <= 1e-12:
        expected = 0.0
        variance = 0.0
    else:
        train_r = train_returns.clip(lower=-_CLIP, upper=_CLIP)
        lw = LedoitWolf()
        lw.fit(train_r.values)
        sigma = lw.covariance_
        ret_daily = float(np.dot(mu_daily, weights))
        expected = (1.0 + ret_daily) ** 252 - 1.0
        variance = float(252.0 * weights @ sigma @ weights)

    volatility = float(np.sqrt(max(variance, 0.0)))
    return PortfolioDiagnostics(
        expected_return=expected,
        variance=variance,
        volatility=volatility,
        favorable=expected + volatility,
        unfavorable=expected - volatility,
        prob_accept=p2_aceptacion(expected, tolerancia),
    )


def _status_label(status: int) -> str:
    labels = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }
    return labels.get(status, f"STATUS_{status}")


def _optimize_portfolio(
    train_returns: pd.DataFrame,
    tolerancia: float,
    perfil: str,
    mu_bl: np.ndarray,
    lam: float,
    max_weight: float,
    vol_factor: float,
    full_invest: bool,
) -> OptimizationResult:
    n = len(train_returns.columns)
    if tolerancia == 0.0:
        return OptimizationResult(
            weights=np.zeros(n),
            status=GRB.OPTIMAL,
            is_optimal=True,
            message="Perfil con tolerancia 0%; se mantiene en caja.",
        )

    train_r = train_returns.clip(lower=-_CLIP, upper=_CLIP)
    model, w_vars = optimizar_markowitz(
        train_r,
        lam=lam,
        perdida_max_anual=tolerancia * vol_factor,
        perfil=perfil,
        mu_personalizado=mu_bl,
        max_weight=max_weight,
        full_invest=full_invest,
    )
    if model.Status == GRB.OPTIMAL:
        return OptimizationResult(
            weights=np.array([w_vars[i].X for i in range(n)]),
            status=model.Status,
            is_optimal=True,
            message="Solucion optima encontrada.",
        )

    return OptimizationResult(
        weights=None,
        status=model.Status,
        is_optimal=False,
        message=(
            "Con las acciones disponibles, el riesgo estimado del mercado supera "
            "lo que este perfil esta dispuesto a tolerar bajo las restricciones actuales. "
            "Por eso no hay un portafolio recomendado que cumpla el limite de riesgo. "
            f"Estado tecnico: {_status_label(model.Status)}."
        ),
    )


def _top_holdings(
    tickers: list[str],
    weights: np.ndarray,
    total_value: float,
    top_n: int,
) -> pd.DataFrame:
    rows = []
    for ticker, weight in zip(tickers, weights):
        if weight > 1e-8:
            rows.append({
                "ticker": ticker,
                "peso": float(weight),
                "monto": float(weight * total_value),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["ticker", "peso", "monto"])
    return df.sort_values("peso", ascending=False).head(top_n)


def _print_holdings(title: str, tickers: list[str], weights: np.ndarray, total_value: float, top_n: int) -> None:
    print(f"\n{title}")
    holdings = _top_holdings(tickers, weights, total_value, top_n)
    if holdings.empty:
        print("  Sin posiciones en acciones; capital queda en caja.")
        return
    for _, row in holdings.iterrows():
        print(f"  {row['ticker']:<12} peso={_format_pct(row['peso']):>9}  monto={_format_usd(row['monto']):>14}")
    invested = float(weights.sum())
    cash = max(0.0, 1.0 - invested)
    print(f"  {'CAJA':<12} peso={_format_pct(cash):>9}  monto={_format_usd(cash * total_value):>14}")


def _print_diagnostics(label: str, diag: PortfolioDiagnostics) -> None:
    print(f"\n{label}")
    print(f"  Prediccion anual esperada : {_format_pct(diag.expected_return)}")
    print(f"  Varianza anual            : {diag.variance:.6f}")
    print(f"  Volatilidad anual         : {_format_pct(diag.volatility)}")
    print(f"  Caso favorable            : {_format_pct(diag.favorable)}  (prediccion + volatilidad)")
    print(f"  Caso desfavorable         : {_format_pct(diag.unfavorable)}  (prediccion - volatilidad)")
    print(f"  Prob. aceptacion estimada : {_format_pct(diag.prob_accept)}")


def _prompt_profile(capital: float) -> str:
    perfiles = list(RISK_PROFILES.keys())
    print("\nSimulacion interactiva de portafolio")
    print(f"Capital inicial: {_format_usd(capital)}")
    print("\nPerfiles disponibles:")
    print("La tolerancia es la perdida maxima anual que acepta cada perfil.")
    for idx, perfil in enumerate(perfiles, start=1):
        print(f"  {idx}. {perfil}  tolerancia={_format_pct(RISK_PROFILES[perfil])}")

    while True:
        choice = input("Elige perfil (numero o nombre): ").strip().lower()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(perfiles):
                return perfiles[idx - 1]
        if choice in RISK_PROFILES:
            return choice
        print("Perfil no valido. Intenta de nuevo.")


def _prompt_test_period(
    default_start_year: int = START_YEAR_DEFAULT,
    default_end_year: int = END_YEAR_DEFAULT,
) -> tuple[str, str]:
    print("\nPeriodo de testing")
    print("Puedes elegir el año de inicio, por ejemplo 2018 o 2019.")
    print(f"El ultimo año completo disponible para testear es {default_end_year}.")
    print("No usamos 2026 porque la informacion disponible de ese año es parcial.")
    print(f"Si presionas ENTER, parte en {default_start_year} y termina en {default_end_year}.")
    while True:
        answer = input("Año de inicio del testing: ").strip()
        if answer == "":
            start_year = default_start_year
            break
        if answer.isdigit() and len(answer) == 4:
            year = int(answer)
            if 1900 <= year <= default_end_year:
                start_year = year
                break
        print(f"Ingresa un año valido entre 1900 y {default_end_year}, por ejemplo 2019.")

    while True:
        answer = input(f"Año final del testing [ENTER = {default_end_year}]: ").strip()
        if answer == "":
            end_year = default_end_year
            break
        if answer.isdigit() and len(answer) == 4:
            year = int(answer)
            if start_year <= year <= default_end_year:
                end_year = year
                break
        print(f"Ingresa un año final entre {start_year} y {default_end_year}.")

    return f"{start_year}-01-01", f"{end_year}-12-31"


def _ask_rebalance(auto: str) -> bool:
    if auto == "yes":
        return True
    if auto == "no":
        return False
    while True:
        answer = input("\nAceptar portafolio recomendado para el proximo mes? [s/N/q]: ").strip().lower()
        if answer in {"s", "si", "y", "yes"}:
            return True
        if answer in {"", "n", "no"}:
            return False
        if answer in {"q", "quit", "salir"}:
            raise KeyboardInterrupt
        print("Responde 's', 'n' o 'q'.")


def _ask_continue_or_exit(auto: str) -> bool:
    if auto in {"yes", "no"}:
        return True
    while True:
        answer = input(
            "\nEl riesgo de las acciones disponibles supera la tolerancia del perfil. "
            "Continuar con el portafolio actual o abandonar? [c/A]: "
        ).strip().lower()
        if answer in {"c", "continuar", "s", "si", "y", "yes"}:
            return True
        if answer in {"", "a", "abandonar", "q", "quit", "salir", "no", "n"}:
            return False
        print("Responde 'c' para continuar o 'a' para abandonar.")


def _month_return_from_state(
    state: PortfolioState,
    price_returns_month: pd.DataFrame,
    div_yields_month: pd.DataFrame,
    commission_rate: float,
) -> tuple[float, float]:
    start_value = state.valor
    tickers = state.tickers
    pr = price_returns_month.reindex(columns=tickers).fillna(0.0).values
    dv = div_yields_month.reindex(columns=tickers).fillna(0.0).values
    for idx in range(len(pr)):
        state.actualizar_dia(pr[idx], dv[idx])
    commission = state.cobrar_comision_mensual(commission_rate) if commission_rate > 0 else 0.0
    monthly_return = state.valor / start_value - 1.0 if start_value > 0 else 0.0
    return monthly_return, commission


def _prepare_data(start: str, train_years: int):
    price_returns, div_yields = _load_universe_with_fallback()
    if price_returns.empty:
        raise FileNotFoundError("No se encontraron retornos de acciones.")

    first_month = pd.Timestamp(start)
    train_init = _train_window(price_returns, first_month, train_years).dropna(axis=1, how="any")
    if train_init.empty:
        raise ValueError("No hay datos suficientes para entrenar en la fecha inicial.")

    tickers = list(train_init.columns)
    pr = price_returns[tickers]
    dv = div_yields.reindex(columns=tickers).fillna(0.0)
    return pr, dv, tickers


def run_advisor(args: argparse.Namespace) -> None:
    perfil = args.perfil or _prompt_profile(args.capital)
    if perfil not in RISK_PROFILES:
        raise ValueError(f"Perfil no valido: {perfil}")

    if args.start is None and args.end is None:
        eval_start, eval_end = _prompt_test_period()
    else:
        eval_start = args.start or f"{START_YEAR_DEFAULT}-01-01"
        eval_end = args.end or f"{END_YEAR_DEFAULT}-12-31"
    market_caps = _load_market_caps()
    price_returns, div_yields, tickers = _prepare_data(eval_start, args.train_years)
    tolerancia = RISK_PROFILES[perfil]
    max_weight = _max_weight_for_profile(None, perfil)
    meses = pd.date_range(eval_start, eval_end, freq="MS")
    if args.max_months:
        meses = meses[: args.max_months]

    print("\n" + "=" * 78)
    print("ASESOR INTERACTIVO BL + MARKOWITZ")
    print("=" * 78)
    print(f"Perfil elegido        : {perfil}")
    print(f"Tolerancia perfil     : {_format_pct(tolerancia)}")
    print(f"Capital inicial       : {_format_usd(args.capital)}")
    print(f"Periodo               : {meses[0].date()} -> {(meses[-1] + pd.offsets.MonthEnd(0)).date()}")
    print(f"BL method             : {args.bl_method}")
    print(f"Train window          : {args.train_years} años")
    print(f"Lambda                : {args.lam}")
    print(f"Max weight perfil     : {_format_pct(max_weight)}")
    print(f"Top holdings impresos : {args.top}")
    print("=" * 78)

    first_train = _train_window(price_returns, meses[0], args.train_years)
    first_train_r = first_train.clip(lower=-_CLIP, upper=_CLIP)
    first_mu = _calcular_mu_bl(
        first_train_r,
        market_caps=market_caps,
        conf_base=args.conf_base,
        lookback=args.lookback,
        skip=args.skip,
        bl_method=args.bl_method,
    )
    initial_result = _optimize_portfolio(
        first_train,
        tolerancia,
        perfil,
        first_mu,
        args.lam,
        max_weight,
        args.vol_factor,
        args.full_invest,
    )
    if not initial_result.is_optimal:
        print("\nAviso: no hay portafolio inicial compatible con la tolerancia del perfil.")
        print(initial_result.message)
        print("Se parte en caja y solo se recomendara cambiar si luego aparece una solucion factible.")
    initial_weights = initial_result.weights if initial_result.weights is not None else np.zeros(len(tickers))
    state = PortfolioState.crear(tickers, args.capital, initial_weights)
    print("\nPortafolio inicial recomendado por el modelo:")
    _print_holdings("Posiciones iniciales", tickers, state.pesos, state.valor, args.top)

    for month_idx, mes in enumerate(meses, start=1):
        mes_end = mes + pd.offsets.MonthEnd(0)
        start_value = state.valor
        train = _train_window(price_returns, mes, args.train_years)
        train_r = train.clip(lower=-_CLIP, upper=_CLIP)
        mu_bl = _calcular_mu_bl(
            train_r,
            market_caps=market_caps,
            conf_base=args.conf_base,
            lookback=args.lookback,
            skip=args.skip,
            bl_method=args.bl_method,
        )
        recommendation = _optimize_portfolio(
            train,
            tolerancia,
            perfil,
            mu_bl,
            args.lam,
            max_weight,
            args.vol_factor,
            args.full_invest,
        )
        recommended_weights = recommendation.weights if recommendation.weights is not None else state.pesos.copy()

        current_diag = _portfolio_diagnostics(state.pesos, train, mu_bl, tolerancia)
        recommended_diag = _portfolio_diagnostics(recommended_weights, train, mu_bl, tolerancia)
        turnover = float(np.abs(recommended_weights - state.pesos).sum() / 2.0)
        prob_abandono = p1_abandono(state.drawdown(), tolerancia)
        same_as_current = recommendation.is_optimal and np.allclose(
            recommended_weights,
            state.pesos,
            atol=1e-8,
        )
        skip_duplicate_first_month = month_idx == 1 and same_as_current

        print("\n" + "-" * 78)
        print(f"Mes {month_idx}: {mes.strftime('%Y-%m')}  decision al inicio del mes")
        print(f"Periodo de retorno real posterior: {mes.date()} -> {mes_end.date()}")
        print("-" * 78)
        print(f"Valor antes de decidir : {_format_usd(start_value)}")
        print(f"Drawdown actual        : {_format_pct(state.drawdown())}")
        print(f"Prob. abandono actual  : {_format_pct(prob_abandono)}")
        print(f"Turnover si rebalancea : {_format_pct(turnover)}")

        _print_holdings("Portafolio actual", tickers, state.pesos, state.valor, args.top)
        _print_diagnostics("Escenarios portafolio actual", current_diag)

        if skip_duplicate_first_month:
            print("\nPortafolio recomendado rebalanceado")
            print("  Primer mes: el portafolio actual ya corresponde a la recomendacion inicial del modelo.")
            print("  No se ofrece rebalanceo porque ambos portafolios son iguales.")
        elif recommendation.is_optimal:
            _print_holdings("Portafolio recomendado rebalanceado", tickers, recommended_weights, state.valor, args.top)
            _print_diagnostics("Escenarios portafolio recomendado", recommended_diag)
            print("\nComparacion recomendacion vs actual")
            print(
                "  Diferencia prediccion : "
                f"{_format_pct(recommended_diag.expected_return - current_diag.expected_return)}"
            )
            print(
                "  Diferencia varianza   : "
                f"{recommended_diag.variance - current_diag.variance:+.6f}"
            )
        else:
            print("\nPortafolio recomendado rebalanceado")
            print(f"  {recommendation.message}")
            print("  No se ofrece cambio este mes.")

        if skip_duplicate_first_month:
            rebalance = False
        elif recommendation.is_optimal:
            try:
                rebalance = _ask_rebalance(args.auto)
            except KeyboardInterrupt:
                print("\nSimulacion interrumpida por el usuario.")
                break
        else:
            if not _ask_continue_or_exit(args.auto):
                print("\nEl usuario decide abandonar la simulacion.")
                break
            rebalance = False

        if rebalance:
            applied_turnover = state.aplicar_pesos(recommended_weights)
            print(f"Rebalanceo aplicado. Turnover efectivo: {_format_pct(applied_turnover)}")
        else:
            print("No se rebalancea. Se mantiene el portafolio actual.")

        pr_mes = price_returns.loc[mes:mes_end]
        dv_mes = div_yields.loc[mes:mes_end]
        if pr_mes.empty:
            print("No hay retornos disponibles para este mes; se avanza sin cambios.")
            continue
        monthly_return, commission = _month_return_from_state(state, pr_mes, dv_mes, args.commission)
        print("\nResultado al cierre del mes")
        print(f"  Retorno real del mes : {_format_pct(monthly_return)}")
        print(f"  Comision cobrada     : {_format_usd(commission)}")
        print(f"  Valor fin mes        : {_format_usd(state.valor)}")
        print(f"  Drawdown fin mes     : {_format_pct(state.drawdown())}")

    print("\n" + "=" * 78)
    print("SIMULACION TERMINADA")
    print("=" * 78)
    print(f"Perfil        : {perfil}")
    print(f"Valor final   : {_format_usd(state.valor)}")
    print(f"Retorno total : {_format_pct(state.valor / args.capital - 1.0)}")
    print(f"Drawdown final: {_format_pct(state.drawdown())}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Asesor interactivo mensual BL + Markowitz.")
    parser.add_argument("--perfil", choices=list(RISK_PROFILES.keys()), default=None)
    parser.add_argument("--capital", type=float, default=CAPITAL_DEFAULT)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--max-months", type=int, default=None)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--auto", choices=["ask", "yes", "no"], default="ask")
    parser.add_argument("--lam", type=float, default=LAM_DEFAULT)
    parser.add_argument("--train-years", type=int, default=TRAIN_WINDOW_YEARS)
    parser.add_argument("--full-invest", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--commission", type=float, default=COMMISSION)
    parser.add_argument("--vol-factor", type=float, default=VOL_FACTOR)
    parser.add_argument("--bl-method", default=BL_METHOD)
    parser.add_argument("--conf-base", type=float, default=BL_CONF_BASE)
    parser.add_argument("--lookback", type=int, default=BL_LOOKBACK)
    parser.add_argument("--skip", type=int, default=BL_SKIP)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_advisor(args)


if __name__ == "__main__":
    main()
