"""
Monte Carlo anual independiente para aceptacion y abandono por perfil.

Cada anio se simula desde capital inicial, por lo que las tasas de abandono
son comparables anio a anio y no quedan arrastradas por eventos previos.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import data.loader as loader
from backtesting.runner import run_all_profiles, run_monte_carlo, resumen_monte_carlo
from config import BL_METHOD, PROFILE_MAX_WEIGHTS, TRAIN_WINDOW_YEARS


CAPITAL = 1_000_000
START_YEAR = 2019
N_CLIENTES = 5_000
LAM = 0.0
SEED = 42
OUT_DIR = Path("resultados_diagnostico")
SUMMARY_OUT = OUT_DIR / "montecarlo_probabilidades_por_anio_resumen.csv"
DETAIL_OUT = OUT_DIR / "montecarlo_probabilidades_por_anio_detalle.csv"
CONFIG_OUT = OUT_DIR / "montecarlo_probabilidades_por_anio_config.csv"


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
        print("  stocks_info.txt no encontrado; usando pesos iguales/inverse-vol en BL robusto.")
        return None

    return info["marketCap"] if "marketCap" in info.columns else None


def _year_end_for_data(year: int, last_date: pd.Timestamp) -> str:
    if year == last_date.year:
        return last_date.strftime("%Y-%m-%d")
    return f"{year}-12-31"


def _base_probabilities(resultados: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for perfil, df in resultados.items():
        rows.append({
            "perfil": perfil,
            "max_weight": float(df["max_weight"].iloc[0]) if "max_weight" in df and len(df) else np.nan,
            "prob_aceptacion_prom_modelo": float(df["prob_p2"].mean()) if len(df) else np.nan,
            "prob_abandono_prom_modelo": float(df["prob_p1"].mean()) if len(df) else np.nan,
            "abandono_trayectoria_base": bool(df["abandona"].any()) if len(df) else False,
            "meses_trayectoria_base": int(len(df)),
        })
    return pd.DataFrame(rows)


def _mc_summary(year: int, mc: dict[str, pd.DataFrame], resultados: dict[str, pd.DataFrame]) -> pd.DataFrame:
    resumen = resumen_monte_carlo(mc).reset_index()
    base = _base_probabilities(resultados)
    rows = []
    for perfil, df in mc.items():
        p_abandono = float(df["abandona"].mean())
        rows.append({
            "perfil": perfil,
            "aceptacion_realizada_mc": float(df["frecuencia_rebalanceo"].mean()),
            "abandono_anual_mc": p_abandono,
            "se_abandono": float(np.sqrt(p_abandono * (1 - p_abandono) / len(df))) if len(df) else np.nan,
            "se_aceptacion": float(df["frecuencia_rebalanceo"].std(ddof=1) / np.sqrt(len(df))) if len(df) > 1 else np.nan,
        })
    extra = pd.DataFrame(rows)
    out = resumen.merge(extra, on="perfil", how="left").merge(base, on="perfil", how="left")
    out.insert(0, "anio", year)
    return out


def main():
    print("Cargando datos...")
    price_returns, div_yields = _load_universe_with_fallback()
    market_caps = _load_market_caps()

    if price_returns.empty:
        raise FileNotFoundError("No se encontraron retornos de acciones.")

    last_date = price_returns.index[-1]
    end_year = last_date.year if last_date.month == 12 else last_date.year - 1
    years = list(range(START_YEAR, end_year + 1))

    print(f"  Tickers cargados : {price_returns.shape[1]}")
    print(f"  Periodo datos    : {price_returns.index[0].date()} -> {last_date.date()}")
    print(f"  Anios MC         : {years[0]} -> {years[-1]}")
    print(f"  Clientes/perfil  : {N_CLIENTES:,}")
    print(f"  BL method        : {BL_METHOD}")
    print(f"  train_years      : {TRAIN_WINDOW_YEARS}")
    print(f"  lambda           : {LAM}")
    print(f"  max_weight perfil: {PROFILE_MAX_WEIGHTS}")

    summary_rows = []
    detail_rows = []

    for year in years:
        eval_start = f"{year}-01-01"
        eval_end = _year_end_for_data(year, last_date)
        print(f"\nAnio {year}: {eval_start} -> {eval_end}")

        resultados, mu_bl_cache, tickers = run_all_profiles(
            price_returns,
            div_yields,
            capital_inicial=CAPITAL,
            eval_start=eval_start,
            eval_end=eval_end,
            train_years=TRAIN_WINDOW_YEARS,
            market_caps=market_caps,
            lam=LAM,
            full_invest=True,
            seed=SEED,
            return_cache=True,
            verbose=False,
        )

        mc = run_monte_carlo(
            price_returns,
            div_yields,
            n_clientes=N_CLIENTES,
            mu_bl_cache=mu_bl_cache,
            tickers=tickers,
            capital_inicial=CAPITAL,
            eval_start=eval_start,
            eval_end=eval_end,
            lam=LAM,
            full_invest=True,
            seed_base=year,
            verbose=False,
        )

        summary = _mc_summary(year, mc, resultados)
        summary_rows.append(summary)

        detail = pd.concat(mc, names=["perfil", "fila"]).reset_index(level="fila", drop=True).reset_index()
        detail.insert(0, "anio", year)
        detail_rows.append(detail)

        cols = ["perfil", "max_weight", "aceptacion_realizada_mc", "abandono_anual_mc", "meses_activos_prom"]
        print(summary[cols].to_string(index=False, formatters={
            "max_weight": "{:.1%}".format,
            "aceptacion_realizada_mc": "{:.1%}".format,
            "abandono_anual_mc": "{:.1%}".format,
        }))

    OUT_DIR.mkdir(exist_ok=True)
    summary_all = pd.concat(summary_rows, ignore_index=True)
    detail_all = pd.concat(detail_rows, ignore_index=True)
    summary_all.to_csv(SUMMARY_OUT, index=False)
    detail_all.to_csv(DETAIL_OUT, index=False)
    pd.DataFrame([{
        "start_year": START_YEAR,
        "end_year": years[-1],
        "n_clientes": N_CLIENTES,
        "train_years": TRAIN_WINDOW_YEARS,
        "lambda": LAM,
        "bl_method": BL_METHOD,
        **{f"max_weight_{k}": v for k, v in PROFILE_MAX_WEIGHTS.items()},
    }]).to_csv(CONFIG_OUT, index=False)

    print(f"\nGuardado en:\n  {SUMMARY_OUT}\n  {DETAIL_OUT}\n  {CONFIG_OUT}")


if __name__ == "__main__":
    main()
