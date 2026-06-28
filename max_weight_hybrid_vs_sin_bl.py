"""
Compara max_weight con el Black-Litterman final vs sin Black-Litterman.

Configuracion:
- lambda = 0
- train_years = 5
- full_invest = True
- max_weight = 1%, 2.5%, 5%, 10%, 15%, 20%
- con BL usa los defaults actuales de config.py:
  robust_factor_hybrid, conf=1, lookback=126, skip=21.

El script guarda parciales para poder reanudar si la corrida se interrumpe.
"""
from pathlib import Path

import pandas as pd

import data.loader as loader
from backtesting.prediction_error import (
    run_profile_prediction_error,
    resumen_prediction_error_por_perfil,
)
from backtesting.runner import run_all_profiles, resumen_backtest
from config import BL_CONF_BASE, BL_LOOKBACK, BL_METHOD, BL_SKIP


CAPITAL = 1_000_000
START_YEAR = 2019
ERROR_END = "2024-12-31"
SEED = 42
LAM = 0.0
FULL_INVEST = True
TRAIN_YEARS = 5
MAX_WEIGHTS = [0.01, 0.025, 0.05, 0.10, 0.15, 0.20]
MODELS = [
    {
        "modelo": "hybrid_bl",
        "use_bl": True,
        "bl_method": BL_METHOD,
        "conf_base": BL_CONF_BASE,
        "lookback": BL_LOOKBACK,
        "skip": BL_SKIP,
    },
    {
        "modelo": "sin_bl",
        "use_bl": False,
        "bl_method": BL_METHOD,
        "conf_base": BL_CONF_BASE,
        "lookback": BL_LOOKBACK,
        "skip": BL_SKIP,
    },
]

OUT_DIR = Path("resultados_diagnostico")
ANNUAL_OUT = OUT_DIR / "max_weight_hybrid_vs_sin_bl_metricas_anuales.csv"
ERROR_OUT = OUT_DIR / "max_weight_hybrid_vs_sin_bl_error_predictivo.csv"
SUMMARY_OUT = OUT_DIR / "max_weight_hybrid_vs_sin_bl_resumen.csv"


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


def _existing_keys(path: Path, key_cols: list[str]) -> set[tuple]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if df.empty:
        return set()
    return set(tuple(row[col] for col in key_cols) for _, row in df[key_cols].drop_duplicates().iterrows())


def _append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    if path.exists():
        prev = pd.read_csv(path)
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(path, index=False)


def run_annual(price_returns, div_yields, market_caps, end_year: int) -> pd.DataFrame:
    existing = _existing_keys(ANNUAL_OUT, ["modelo", "max_weight", "anio"])

    for model_cfg in MODELS:
        for max_weight in MAX_WEIGHTS:
            for year in range(START_YEAR, end_year + 1):
                key = (model_cfg["modelo"], max_weight, year)
                if key in existing:
                    print(f"  Saltando existente {key}")
                    continue

                eval_start = f"{year}-01-01"
                eval_end = _year_end_for_data(year, price_returns.index[-1])
                print(
                    f"\n{model_cfg['modelo']} max_weight={max_weight:.1%} "
                    f"ano {year}: {eval_start} -> {eval_end}",
                    flush=True,
                )

                resultados = run_all_profiles(
                    price_returns,
                    div_yields,
                    capital_inicial=CAPITAL,
                    eval_start=eval_start,
                    eval_end=eval_end,
                    train_years=TRAIN_YEARS,
                    market_caps=market_caps,
                    lam=LAM,
                    max_weight=max_weight,
                    full_invest=FULL_INVEST,
                    conf_base=model_cfg["conf_base"],
                    lookback=model_cfg["lookback"],
                    skip=model_cfg["skip"],
                    bl_method=model_cfg["bl_method"],
                    seed=SEED,
                    verbose=False,
                    use_bl=model_cfg["use_bl"],
                )

                rows = []
                for perfil, df in resultados.items():
                    r = resumen_backtest(df)
                    rows.append({
                        "modelo": model_cfg["modelo"],
                        "use_bl": model_cfg["use_bl"],
                        "bl_method": model_cfg["bl_method"],
                        "conf_base": model_cfg["conf_base"],
                        "lookback": model_cfg["lookback"],
                        "skip": model_cfg["skip"],
                        "train_years": TRAIN_YEARS,
                        "max_weight": max_weight,
                        "anio": year,
                        "perfil": perfil,
                        "prob_aceptacion_prom": float(df["prob_p2"].mean()) if len(df) else pd.NA,
                        "prob_abandono_prom": float(df["prob_p1"].mean()) if len(df) else pd.NA,
                        **r,
                    })
                _append_csv(ANNUAL_OUT, rows)
                existing.add(key)

    return pd.read_csv(ANNUAL_OUT)


def run_errors(price_returns, market_caps) -> pd.DataFrame:
    existing = _existing_keys(ERROR_OUT, ["modelo", "max_weight"])

    for model_cfg in MODELS:
        for max_weight in MAX_WEIGHTS:
            key = (model_cfg["modelo"], max_weight)
            if key in existing:
                print(f"  Saltando error existente {key}")
                continue

            print(f"\nError {model_cfg['modelo']} max_weight={max_weight:.1%}", flush=True)
            df_error = run_profile_prediction_error(
                price_returns,
                market_caps=market_caps,
                eval_start=f"{START_YEAR}-01-01",
                eval_end=ERROR_END,
                train_years=TRAIN_YEARS,
                conf_base=model_cfg["conf_base"],
                lookback=model_cfg["lookback"],
                skip=model_cfg["skip"],
                lam=LAM,
                max_weight=max_weight,
                full_invest=FULL_INVEST,
                use_bl=model_cfg["use_bl"],
                bl_method=model_cfg["bl_method"],
            )
            resumen = resumen_prediction_error_por_perfil(df_error).reset_index()
            resumen["modelo"] = model_cfg["modelo"]
            resumen["use_bl"] = model_cfg["use_bl"]
            resumen["bl_method"] = model_cfg["bl_method"]
            resumen["conf_base"] = model_cfg["conf_base"]
            resumen["lookback"] = model_cfg["lookback"]
            resumen["skip"] = model_cfg["skip"]
            resumen["train_years"] = TRAIN_YEARS
            resumen["max_weight"] = max_weight
            _append_csv(ERROR_OUT, resumen.to_dict("records"))
            existing.add(key)

    return pd.read_csv(ERROR_OUT)


def build_summary(annual: pd.DataFrame, errors: pd.DataFrame) -> pd.DataFrame:
    metricas = annual.groupby(["modelo", "max_weight", "perfil"]).agg(
        retorno_promedio=("retorno_anual", "mean"),
        retorno_mediano=("retorno_anual", "median"),
        drawdown_promedio=("max_drawdown", "mean"),
        abandono_promedio=("abandono", "mean"),
        prob_abandono_promedio=("prob_abandono_prom", "mean"),
        prob_aceptacion_promedio=("prob_aceptacion_prom", "mean"),
        frecuencia_rebalanceo_promedio=("frecuencia_rebalanceo", "mean"),
        meses_activos_promedio=("meses_activos", "mean"),
    ).reset_index()

    resumen = metricas.merge(
        errors[[
            "modelo", "max_weight", "perfil", "mae", "rmse", "bias",
            "hit_rate_signo", "n_activos_promedio",
        ]],
        on=["modelo", "max_weight", "perfil"],
        how="left",
    )
    resumen["score"] = (
        resumen["retorno_promedio"]
        - 0.50 * resumen["drawdown_promedio"]
        - 0.50 * resumen["mae"]
        - 0.30 * resumen["abandono_promedio"]
    )

    delta = []
    for (max_weight, perfil), group in resumen.groupby(["max_weight", "perfil"]):
        pivot = group.set_index("modelo")
        if {"hybrid_bl", "sin_bl"}.issubset(pivot.index):
            row = {
                "modelo": "delta_hybrid_minus_sin_bl",
                "max_weight": max_weight,
                "perfil": perfil,
            }
            for col in [
                "retorno_promedio", "drawdown_promedio", "abandono_promedio",
                "prob_abandono_promedio", "prob_aceptacion_promedio",
                "frecuencia_rebalanceo_promedio", "mae", "rmse", "bias",
                "hit_rate_signo", "n_activos_promedio", "score",
            ]:
                row[col] = pivot.loc["hybrid_bl", col] - pivot.loc["sin_bl", col]
            row["retorno_mediano"] = pivot.loc["hybrid_bl", "retorno_mediano"] - pivot.loc["sin_bl", "retorno_mediano"]
            row["meses_activos_promedio"] = pivot.loc["hybrid_bl", "meses_activos_promedio"] - pivot.loc["sin_bl", "meses_activos_promedio"]
            delta.append(row)

    if delta:
        resumen = pd.concat([resumen, pd.DataFrame(delta)], ignore_index=True)
    resumen.to_csv(SUMMARY_OUT, index=False)
    return resumen


def _format_pivot(tabla: pd.DataFrame, value_col: str) -> str:
    pivot = tabla.pivot_table(index=["modelo", "perfil"], columns="max_weight", values=value_col, aggfunc="first")
    return pivot.to_string(formatters={
        col: (lambda x: "" if pd.isna(x) else f"{x:.1%}")
        for col in pivot.columns
    })


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
    print(f"  BL final         : {BL_METHOD}, conf={BL_CONF_BASE}, lookback={BL_LOOKBACK}, skip={BL_SKIP}")

    annual = run_annual(price_returns, div_yields, market_caps, end_year)
    errors = run_errors(price_returns, market_caps)
    resumen = build_summary(annual, errors)

    print("\n-- Retorno promedio --------------------------------------------")
    print(_format_pivot(resumen[resumen["modelo"].isin(["hybrid_bl", "sin_bl"])], "retorno_promedio"))
    print("\n-- Delta retorno hybrid - sin BL -------------------------------")
    print(_format_pivot(resumen[resumen["modelo"] == "delta_hybrid_minus_sin_bl"], "retorno_promedio"))
    print(f"\nResultados guardados en {OUT_DIR}/")


if __name__ == "__main__":
    main()
