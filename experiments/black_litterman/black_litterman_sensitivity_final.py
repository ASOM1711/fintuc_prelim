"""
Sensibilidad de Black-Litterman para la configuracion final.

Etapa 1:
    Barre parametros de la senal BL sin optimizar portafolios, usando error
    predictivo a nivel activo. Esto permite descartar combinaciones malas rapido.

Etapa 2:
    Toma los mejores candidatos y corre backtesting anual independiente completo
    con train_years=5, max_weight=2.5%, lambda=0 y full_invest=True.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import data.loader as loader
from backtesting.prediction_error import (
    run_prediction_error,
    run_profile_prediction_error,
    resumen_prediction_error,
    resumen_prediction_error_por_perfil,
)
from backtesting.runner import run_all_profiles, resumen_backtest


CAPITAL = 1_000_000
START_YEAR = 2019
ERROR_END = "2024-12-31"
SEED = 42
LAM = 0.0
FULL_INVEST = True
TRAIN_YEARS = 5
MAX_WEIGHT = 0.025

CONF_BASES = [0.05, 0.10, 0.25, 0.50, 1.00, 2.00]
LOOKBACKS = [63, 126, 252, 504]
SKIPS = [0, 21, 63]
TOP_N = 4
BASELINE = {"conf_base": 0.05, "lookback": 252, "skip": 21}

OUT_DIR = Path("resultados_diagnostico")
SIGNAL_GRID = OUT_DIR / "bl_sensitivity_signal_grid.csv"
CANDIDATES = OUT_DIR / "bl_sensitivity_candidates.csv"
ANNUAL_METRICS = OUT_DIR / "bl_sensitivity_annual_metrics.csv"
PROFILE_ERROR = OUT_DIR / "bl_sensitivity_profile_error.csv"
SUMMARY = OUT_DIR / "bl_sensitivity_summary.csv"

PROFILE_ORDER = [
    "muy_conservador",
    "conservador",
    "neutro",
    "arriesgado",
    "muy_arriesgado",
]


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
            print(f"  DATA_DIR no encontro datos; usando fallback: {candidate}")
            loader.DATA_DIR = candidate
            return loader.load_universe()

    return price_returns, div_yields


def _load_market_caps():
    try:
        info = loader.load_stock_info()
    except FileNotFoundError:
        print("  stocks_info.txt no encontrado; usando pesos iguales en Black-Litterman.")
        return None

    return info["marketCap"] if "marketCap" in info.columns else None


def _year_end_for_data(year: int, last_date: pd.Timestamp) -> str:
    if year == last_date.year:
        return last_date.strftime("%Y-%m-%d")
    return f"{year}-12-31"


def _config_id(conf_base: float, lookback: int, skip: int) -> str:
    return f"conf={conf_base:g}|lb={lookback}|skip={skip}"


def _signal_score(row: pd.Series) -> float:
    # Menor es mejor. Penaliza error, premia correlacion e hit rate.
    corr = 0.0 if pd.isna(row["corr_pred_real"]) else row["corr_pred_real"]
    hit = 0.5 if pd.isna(row["hit_rate_signo"]) else row["hit_rate_signo"]
    return float(row["mae"] - 0.03 * corr - 0.02 * (hit - 0.5))


def run_signal_grid(price_returns: pd.DataFrame, market_caps) -> pd.DataFrame:
    rows = []
    total = len(CONF_BASES) * len(LOOKBACKS) * len(SKIPS)
    i = 0
    for conf_base in CONF_BASES:
        for lookback in LOOKBACKS:
            for skip in SKIPS:
                i += 1
                config_id = _config_id(conf_base, lookback, skip)
                print(f"  Senal BL {i}/{total}: {config_id}", flush=True)
                df_error = run_prediction_error(
                    price_returns,
                    market_caps=market_caps,
                    eval_start=f"{START_YEAR}-01-01",
                    eval_end=ERROR_END,
                    train_years=TRAIN_YEARS,
                    conf_base=conf_base,
                    lookback=lookback,
                    skip=skip,
                )
                summary = resumen_prediction_error(df_error)
                rows.append({
                    "config_id": config_id,
                    "conf_base": conf_base,
                    "lookback": lookback,
                    "skip": skip,
                    **summary,
                })
                pd.DataFrame(rows).to_csv(SIGNAL_GRID, index=False)

    out = pd.DataFrame(rows)
    out["signal_score"] = out.apply(_signal_score, axis=1)
    out = out.sort_values("signal_score")
    out.to_csv(SIGNAL_GRID, index=False)
    return out


def select_candidates(signal_grid: pd.DataFrame) -> pd.DataFrame:
    candidates = signal_grid.head(TOP_N).copy()
    baseline_mask = (
        (signal_grid["conf_base"] == BASELINE["conf_base"])
        & (signal_grid["lookback"] == BASELINE["lookback"])
        & (signal_grid["skip"] == BASELINE["skip"])
    )
    baseline = signal_grid[baseline_mask]
    candidates = pd.concat([candidates, baseline], ignore_index=True)
    candidates = candidates.drop_duplicates(["conf_base", "lookback", "skip"])
    candidates = candidates.sort_values("signal_score").reset_index(drop=True)
    candidates.to_csv(CANDIDATES, index=False)
    return candidates


def run_annual_backtests(price_returns, div_yields, market_caps, end_year: int, candidates: pd.DataFrame) -> pd.DataFrame:
    registros = []
    for _, cfg in candidates.iterrows():
        conf_base = float(cfg["conf_base"])
        lookback = int(cfg["lookback"])
        skip = int(cfg["skip"])
        config_id = _config_id(conf_base, lookback, skip)
        print(f"\nBacktesting candidato {config_id}", flush=True)

        for year in range(START_YEAR, end_year + 1):
            eval_start = f"{year}-01-01"
            eval_end = _year_end_for_data(year, price_returns.index[-1])
            print(f"  Ano {year}: {eval_start} -> {eval_end}", flush=True)

            resultados = run_all_profiles(
                price_returns,
                div_yields,
                capital_inicial=CAPITAL,
                eval_start=eval_start,
                eval_end=eval_end,
                train_years=TRAIN_YEARS,
                market_caps=market_caps,
                lam=LAM,
                max_weight=MAX_WEIGHT,
                full_invest=FULL_INVEST,
                conf_base=conf_base,
                lookback=lookback,
                skip=skip,
                seed=SEED,
                verbose=False,
                use_bl=True,
            )

            for perfil, df in resultados.items():
                r = resumen_backtest(df)
                registros.append({
                    "config_id": config_id,
                    "conf_base": conf_base,
                    "lookback": lookback,
                    "skip": skip,
                    "train_years": TRAIN_YEARS,
                    "max_weight": MAX_WEIGHT,
                    "anio": year,
                    "perfil": perfil,
                    "prob_aceptacion_prom": float(df["prob_p2"].mean()) if len(df) else pd.NA,
                    "prob_abandono_prom": float(df["prob_p1"].mean()) if len(df) else pd.NA,
                    **r,
                })
            pd.DataFrame(registros).to_csv(ANNUAL_METRICS, index=False)

    out = pd.DataFrame(registros)
    out.to_csv(ANNUAL_METRICS, index=False)
    return out


def run_portfolio_errors(price_returns, market_caps, candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, cfg in candidates.iterrows():
        conf_base = float(cfg["conf_base"])
        lookback = int(cfg["lookback"])
        skip = int(cfg["skip"])
        config_id = _config_id(conf_base, lookback, skip)
        print(f"\nError portafolio candidato {config_id}", flush=True)

        df_error = run_profile_prediction_error(
            price_returns,
            market_caps=market_caps,
            eval_start=f"{START_YEAR}-01-01",
            eval_end=ERROR_END,
            train_years=TRAIN_YEARS,
            conf_base=conf_base,
            lookback=lookback,
            skip=skip,
            lam=LAM,
            max_weight=MAX_WEIGHT,
            full_invest=FULL_INVEST,
            use_bl=True,
        )
        resumen = resumen_prediction_error_por_perfil(df_error).reset_index()
        resumen["config_id"] = config_id
        resumen["conf_base"] = conf_base
        resumen["lookback"] = lookback
        resumen["skip"] = skip
        rows.append(resumen)
        pd.concat(rows, ignore_index=True).to_csv(PROFILE_ERROR, index=False)

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(PROFILE_ERROR, index=False)
    return out


def build_summary(annual: pd.DataFrame, errors: pd.DataFrame) -> pd.DataFrame:
    annual_summary = annual.groupby(["config_id", "conf_base", "lookback", "skip", "perfil"]).agg(
        retorno_promedio=("retorno_anual", "mean"),
        drawdown_promedio=("max_drawdown", "mean"),
        tasa_aceptacion=("frecuencia_rebalanceo", "mean"),
        tasa_abandono=("abandono", "mean"),
        prob_aceptacion_promedio=("prob_aceptacion_prom", "mean"),
        prob_abandono_promedio=("prob_abandono_prom", "mean"),
        meses_activos_promedio=("meses_activos", "mean"),
        peor_retorno=("retorno_anual", "min"),
    ).reset_index()

    error_cols = [
        "config_id", "perfil", "mae", "rmse", "bias", "hit_rate_signo",
        "n_activos_promedio",
    ]
    summary = annual_summary.merge(errors[error_cols], on=["config_id", "perfil"], how="left")
    summary["score"] = (
        summary["retorno_promedio"]
        - 0.50 * summary["drawdown_promedio"]
        - 0.50 * summary["mae"]
        - 0.30 * summary["tasa_abandono"]
    )
    summary.to_csv(SUMMARY, index=False)
    return summary


def _print_best(summary: pd.DataFrame) -> None:
    best = summary.sort_values(["perfil", "score"], ascending=[True, False]).groupby("perfil").head(1)
    best = best.set_index("perfil").reindex(PROFILE_ORDER).reset_index()
    print("\n-- Mejor BL por perfil ------------------------------------------")
    for _, row in best.iterrows():
        print(
            f"  {row['perfil']:<20} {row['config_id']:<24} "
            f"ret={row['retorno_promedio']:.1%} dd={row['drawdown_promedio']:.1%} "
            f"acept={row['tasa_aceptacion']:.1%} aband={row['tasa_abandono']:.1%} "
            f"mae={row['mae']:.1%} score={row['score']:.1%}"
        )

    avg = summary.groupby("config_id").agg(
        retorno_promedio=("retorno_promedio", "mean"),
        drawdown_promedio=("drawdown_promedio", "mean"),
        tasa_aceptacion=("tasa_aceptacion", "mean"),
        tasa_abandono=("tasa_abandono", "mean"),
        mae=("mae", "mean"),
        score=("score", "mean"),
    ).sort_values("score", ascending=False)
    print("\n-- Ranking promedio de candidatos -------------------------------")
    print(avg.to_string(formatters={
        col: (lambda x: "" if pd.isna(x) else f"{x:.1%}")
        for col in avg.columns
    }))


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Cargando datos...")
    price_returns, div_yields = _load_universe_with_fallback()
    market_caps = _load_market_caps()
    if price_returns.empty:
        raise FileNotFoundError("No se encontraron retornos de acciones.")

    last_date = price_returns.index[-1]
    end_year = last_date.year if last_date.month == 12 else last_date.year - 1
    print(f"  Tickers cargados : {price_returns.shape[1]}")
    print(f"  Periodo de datos : {price_returns.index[0].date()} -> {last_date.date()}")
    print(f"  Anos metricas    : {START_YEAR} -> {end_year}")
    print(f"  Error predictivo : {START_YEAR} -> 2024")

    print("\nEtapa 1: sensibilidad de senal BL")
    signal_grid = run_signal_grid(price_returns, market_caps)
    candidates = select_candidates(signal_grid)
    print("\nCandidatos seleccionados:")
    print(candidates[["config_id", "mae", "rmse", "hit_rate_signo", "corr_pred_real", "signal_score"]].to_string(index=False))

    print("\nEtapa 2: backtesting anual de candidatos")
    annual = run_annual_backtests(price_returns, div_yields, market_caps, end_year, candidates)
    errors = run_portfolio_errors(price_returns, market_caps, candidates)
    summary = build_summary(annual, errors)
    _print_best(summary)

    print("\nArchivos guardados:")
    for path in [SIGNAL_GRID, CANDIDATES, ANNUAL_METRICS, PROFILE_ERROR, SUMMARY]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
