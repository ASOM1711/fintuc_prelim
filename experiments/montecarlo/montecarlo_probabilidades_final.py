"""
Monte Carlo final para estimar tasas realizadas de aceptacion y abandono.

Configuracion:
- Black-Litterman default actual de config.py.
- train_years = 5.
- lambda = 0.
- max_weight por perfil desde PROFILE_MAX_WEIGHTS.
- full_invest = True.
- anios completos 2019-2025.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import data.loader as loader
from backtesting.runner import run_all_profiles, run_monte_carlo, resumen_monte_carlo
from config import BL_METHOD, PROFILE_MAX_WEIGHTS, TRAIN_WINDOW_YEARS


CAPITAL = 1_000_000
EVAL_START = "2019-01-01"
EVAL_END = "2025-12-31"
N_CLIENTES = 5_000
LAM = 0.0
SEED = 42
OUT_DIR = Path("resultados_diagnostico")


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


def _resumen_probabilidades_deterministicas(resultados: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for perfil, df in resultados.items():
        rows.append({
            "perfil": perfil,
            "max_weight": float(df["max_weight"].iloc[0]) if "max_weight" in df and len(df) else np.nan,
            "prob_aceptacion_prom_modelo": float(df["prob_p2"].mean()) if len(df) else np.nan,
            "prob_abandono_prom_modelo": float(df["prob_p1"].mean()) if len(df) else np.nan,
            "meses_trayectoria_base": int(len(df)),
            "abandono_trayectoria_base": bool(df["abandona"].any()) if len(df) else False,
        })
    return pd.DataFrame(rows)


def _resumen_mc_con_error(resumen_mc: pd.DataFrame, mc: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = resumen_mc.reset_index().copy()
    rows = []
    for perfil, df in mc.items():
        p_abandono = float(df["abandona"].mean())
        p_aceptacion = float(df["frecuencia_rebalanceo"].mean())
        n = len(df)
        rows.append({
            "perfil": perfil,
            "se_abandono": float(np.sqrt(p_abandono * (1 - p_abandono) / n)) if n else np.nan,
            "se_aceptacion": float(df["frecuencia_rebalanceo"].std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
            "aceptacion_realizada_mc": p_aceptacion,
            "abandono_acumulado_mc": p_abandono,
        })
    extra = pd.DataFrame(rows)
    out = out.merge(extra, on="perfil", how="left")
    return out


def main():
    print("Cargando datos...")
    price_returns, div_yields = _load_universe_with_fallback()
    market_caps = _load_market_caps()

    if price_returns.empty:
        raise FileNotFoundError("No se encontraron retornos de acciones.")

    print(f"  Tickers cargados : {price_returns.shape[1]}")
    print(f"  Periodo datos    : {price_returns.index[0].date()} -> {price_returns.index[-1].date()}")
    print(f"  Periodo MC       : {EVAL_START} -> {EVAL_END}")
    print(f"  Clientes/perfil  : {N_CLIENTES:,}")
    print(f"  BL method        : {BL_METHOD}")
    print(f"  train_years      : {TRAIN_WINDOW_YEARS}")
    print(f"  lambda           : {LAM}")
    print(f"  max_weight perfil: {PROFILE_MAX_WEIGHTS}")

    print("\nPrecomputando BL y trayectoria base...")
    resultados, mu_bl_cache, tickers = run_all_profiles(
        price_returns,
        div_yields,
        capital_inicial=CAPITAL,
        eval_start=EVAL_START,
        eval_end=EVAL_END,
        train_years=TRAIN_WINDOW_YEARS,
        market_caps=market_caps,
        lam=LAM,
        full_invest=True,
        seed=SEED,
        return_cache=True,
        verbose=True,
    )

    print(f"\nSimulando Monte Carlo ({N_CLIENTES:,} clientes por perfil)...")
    mc = run_monte_carlo(
        price_returns,
        div_yields,
        n_clientes=N_CLIENTES,
        mu_bl_cache=mu_bl_cache,
        tickers=tickers,
        capital_inicial=CAPITAL,
        eval_start=EVAL_START,
        eval_end=EVAL_END,
        lam=LAM,
        full_invest=True,
        seed_base=0,
        verbose=True,
    )

    resumen_mc = resumen_monte_carlo(mc)
    resumen_final = _resumen_mc_con_error(resumen_mc, mc)
    probs_base = _resumen_probabilidades_deterministicas(resultados)
    resumen_final = resumen_final.merge(probs_base, on="perfil", how="left")

    OUT_DIR.mkdir(exist_ok=True)
    resumen_path = OUT_DIR / "montecarlo_final_probabilidades_resumen.csv"
    detalle_path = OUT_DIR / "montecarlo_final_probabilidades_detalle.csv"
    config_path = OUT_DIR / "montecarlo_final_probabilidades_config.csv"

    resumen_final.to_csv(resumen_path, index=False)
    pd.concat(mc, names=["perfil", "fila"]).reset_index(level="fila", drop=True).reset_index().to_csv(
        detalle_path, index=False
    )
    pd.DataFrame([{
        "eval_start": EVAL_START,
        "eval_end": EVAL_END,
        "n_clientes": N_CLIENTES,
        "train_years": TRAIN_WINDOW_YEARS,
        "lambda": LAM,
        "bl_method": BL_METHOD,
        **{f"max_weight_{k}": v for k, v in PROFILE_MAX_WEIGHTS.items()},
    }]).to_csv(config_path, index=False)

    cols = [
        "perfil",
        "max_weight",
        "aceptacion_realizada_mc",
        "abandono_acumulado_mc",
        "se_aceptacion",
        "se_abandono",
        "meses_activos_prom",
        "retorno_prom",
        "retorno_std",
        "prob_aceptacion_prom_modelo",
        "prob_abandono_prom_modelo",
    ]
    print("\nResumen Monte Carlo final:")
    print(resumen_final[cols].to_string(index=False, formatters={
        "max_weight": "{:.1%}".format,
        "aceptacion_realizada_mc": "{:.1%}".format,
        "abandono_acumulado_mc": "{:.1%}".format,
        "se_aceptacion": "{:.2%}".format,
        "se_abandono": "{:.2%}".format,
        "retorno_prom": "{:.1%}".format,
        "retorno_std": "{:.1%}".format,
        "prob_aceptacion_prom_modelo": "{:.1%}".format,
        "prob_abandono_prom_modelo": "{:.1%}".format,
    }))
    print(f"\nGuardado en:\n  {resumen_path}\n  {detalle_path}\n  {config_path}")


if __name__ == "__main__":
    main()
