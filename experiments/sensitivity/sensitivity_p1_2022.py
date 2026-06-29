"""
Sensibilidad de la función P1 (abandono) para el año 2022.

Corre el backtest 2022 con abandono desactivado para obtener la secuencia
completa de drawdowns mensuales por perfil. Luego aplica 5 variantes de P1
y muestra cuándo habría ocurrido el abandono bajo cada una.

Sin modificar ningún archivo existente.
"""

import numpy as np
import pandas as pd

import portfolio.probabilities as probs
from data.loader import load_universe, load_stock_info
from backtesting.runner import run_all_profiles
from config import RISK_PROFILES, PROFILE_MAX_WEIGHTS, BL_METHOD, BL_CONF_BASE, BL_LOOKBACK, BL_SKIP

CAPITAL     = 1_000_000
EVAL_START  = "2022-01-01"
EVAL_END    = "2022-12-31"
TRAIN_YEARS = 5
SEED        = 42

PERFILES_ORDEN = ["muy_conservador", "conservador", "neutro", "arriesgado", "muy_arriesgado"]
LABELS = {
    "muy_conservador": "Muy conserv.",
    "conservador":     "Conservador",
    "neutro":          "Neutro",
    "arriesgado":      "Arriesgado",
    "muy_arriesgado":  "Muy arriesg.",
}

# ---------------------------------------------------------------------------
# Variantes de P1
# ---------------------------------------------------------------------------

def _sigmoid(exceso: float, steepness: float) -> float:
    return 1.0 / (1.0 + np.exp(-exceso * steepness))

def _lineal(exceso: float, slope: float = 15.0) -> float:
    return min(1.0, exceso * slope)

VARIANTES = {
    "muy_estricto": lambda exc: _sigmoid(exc, 500),   # abandona casi instantáneo al cruzar
    "actual":       lambda exc: _sigmoid(exc, 100),   # configuración actual
    "gradual":      lambda exc: _sigmoid(exc, 20),    # cliente más paciente
    "muy_gradual":  lambda exc: _sigmoid(exc, 5),     # cliente muy paciente
    "lineal":       lambda exc: _lineal(exc, 15),     # alternativa lineal
}

VARIANTE_LABELS = {
    "muy_estricto": "Muy estricto  (sigmoide k=500)",
    "actual":       "Actual        (sigmoide k=100)",
    "gradual":      "Gradual       (sigmoide k=20) ",
    "muy_gradual":  "Muy gradual   (sigmoide k=5)  ",
    "lineal":       "Lineal        (pendiente=15)  ",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _p1_variante(drawdown: float, tolerancia: float, fn) -> float:
    if tolerancia == 0.0 or drawdown <= tolerancia:
        return 0.0
    return fn(drawdown - tolerancia)


def _primer_mes_esperado(drawdowns: list, tolerancia: float, fn) -> int | None:
    """Primer mes donde P1 > 0.5 (abandono más probable que no)."""
    for i, dd in enumerate(drawdowns):
        if _p1_variante(dd, tolerancia, fn) > 0.5:
            return i + 1  # mes 1-indexado
    return None


def _primer_mes_seguro(drawdowns: list, tolerancia: float, fn, umbral=0.9) -> int | None:
    """Primer mes donde P1 > umbral (casi certeza de abandono)."""
    for i, dd in enumerate(drawdowns):
        if _p1_variante(dd, tolerancia, fn) > umbral:
            return i + 1
    return None

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Cargando datos...")
    pr, dv = load_universe()
    try:
        info        = load_stock_info()
        market_caps = info["marketCap"] if "marketCap" in info.columns else None
    except FileNotFoundError:
        market_caps = None

    # ----------------------------------------------------------------
    # 1. Correr backtest 2022 con P1 desactivada (siempre retorna 0)
    #    para obtener los 12 meses completos de drawdown por perfil.
    # ----------------------------------------------------------------
    print("\nCorriendo backtest 2022 con abandono desactivado...")
    _orig_p1 = probs.p1_abandono
    probs.p1_abandono = lambda dd, tol, lineal=False: 0.0   # monkey-patch

    resultados = run_all_profiles(
        pr, dv,
        capital_inicial  = CAPITAL,
        eval_start       = EVAL_START,
        eval_end         = EVAL_END,
        train_years      = TRAIN_YEARS,
        market_caps      = market_caps,
        bl_method        = BL_METHOD,
        conf_base        = BL_CONF_BASE,
        lookback         = BL_LOOKBACK,
        skip             = BL_SKIP,
        lam              = 0.0,
        full_invest      = True,
        seed             = SEED,
        verbose          = False,
    )

    probs.p1_abandono = _orig_p1  # restaurar

    # ----------------------------------------------------------------
    # 2. Extraer drawdown mensual por perfil
    # ----------------------------------------------------------------
    drawdowns_por_perfil = {}
    for perfil in PERFILES_ORDEN:
        df = resultados.get(perfil, pd.DataFrame())
        if df.empty or "drawdown" not in df.columns:
            drawdowns_por_perfil[perfil] = []
        else:
            drawdowns_por_perfil[perfil] = list(df["drawdown"].values)

    # ----------------------------------------------------------------
    # 3. Tabla: drawdown mensual + P1 por variante
    # ----------------------------------------------------------------
    print("\n" + "="*80)
    print("  DRAWDOWN MENSUAL 2022 Y PROBABILIDAD P1 SEGÚN VARIANTE")
    print("="*80)

    for perfil in PERFILES_ORDEN:
        tolerancia = RISK_PROFILES[perfil]
        dds        = drawdowns_por_perfil[perfil]
        if not dds:
            continue

        print(f"\n{'-'*80}")
        print(f"  Perfil: {LABELS[perfil]}   (tolerancia = {tolerancia:.0%})")
        print(f"{'-'*80}")

        header = f"  {'Mes':>4}  {'Drawdown':>10}  " + \
                 "  ".join(f"{k:>13}" for k in VARIANTES)
        print(header)
        print("  " + "-"*76)

        for i, dd in enumerate(dds):
            p1_vals = {k: _p1_variante(dd, tolerancia, fn)
                       for k, fn in VARIANTES.items()}
            fila = f"  {i+1:>4}  {dd:>9.1%}  " + \
                   "  ".join(f"{v:>13.1%}" for v in p1_vals.values())
            print(fila)

    # ----------------------------------------------------------------
    # 4. Tabla resumen: primer mes de abandono esperado por variante
    # ----------------------------------------------------------------
    print("\n\n" + "="*80)
    print("  RESUMEN: PRIMER MES DONDE P1 > 50%  (abandono 'esperado')")
    print("  y P1 > 90% (abandono 'casi seguro')")
    print("="*80)

    header2 = f"  {'Perfil':<16}  {'Tol':>5}  " + \
              "  ".join(f"{k:>13}" for k in VARIANTES)
    print(header2)
    print("  " + "-"*76)

    for perfil in PERFILES_ORDEN:
        tolerancia = RISK_PROFILES[perfil]
        dds        = drawdowns_por_perfil[perfil]
        if not dds:
            continue

        celdas = []
        for fn in VARIANTES.values():
            m50 = _primer_mes_esperado(dds, tolerancia, fn)
            m90 = _primer_mes_seguro(dds, tolerancia, fn)
            if m50 is None:
                celdas.append("     nunca")
            elif m90 is None:
                celdas.append(f" m{m50:02d}(>50%)")
            else:
                celdas.append(f"m{m50:02d}/m{m90:02d}")

        print(f"  {LABELS[perfil]:<16}  {tolerancia:>5.0%}  " +
              "  ".join(f"{c:>13}" for c in celdas))

    # ----------------------------------------------------------------
    # 5. Tabla: máximo drawdown y mes en que ocurre
    # ----------------------------------------------------------------
    print("\n\n" + "="*80)
    print("  MÁXIMO DRAWDOWN 2022 POR PERFIL")
    print("="*80)
    print(f"  {'Perfil':<16}  {'Tol':>5}  {'Max DD':>8}  {'En mes':>7}  {'Exceso':>8}")
    print("  " + "-"*50)
    for perfil in PERFILES_ORDEN:
        tolerancia = RISK_PROFILES[perfil]
        dds        = drawdowns_por_perfil[perfil]
        if not dds:
            continue
        max_dd  = max(dds)
        mes_max = dds.index(max_dd) + 1
        exceso  = max(0.0, max_dd - tolerancia)
        print(f"  {LABELS[perfil]:<16}  {tolerancia:>5.0%}  "
              f"{max_dd:>8.1%}  {'mes '+str(mes_max):>7}  {exceso:>8.1%}")

    # ----------------------------------------------------------------
    # 6. Curvas P1 teóricas (sin datos reales)
    # ----------------------------------------------------------------
    print("\n\n" + "="*80)
    print("  CURVAS P1 TEÓRICAS: P1 según exceso de drawdown sobre tolerancia")
    print("="*80)
    excesos = [0.00, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
    header3 = f"  {'Exceso':>8}  " + \
              "  ".join(f"{k:>13}" for k in VARIANTES)
    print(header3)
    print("  " + "-"*76)
    for exc in excesos:
        vals = [fn(exc) for fn in VARIANTES.values()]
        print(f"  {exc:>8.0%}  " +
              "  ".join(f"{v:>13.1%}" for v in vals))

    print("\nListo.\n")


if __name__ == "__main__":
    main()
