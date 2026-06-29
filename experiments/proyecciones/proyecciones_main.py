"""
Proyecciones forward 3 anios bajo 3 escenarios de mercado, con P1/P2.

Metodologia
-----------
1. Pesos optimos: BL + Markowitz entrenado en los ultimos 5 anios (2020-2024).
   Se fijan una vez y son los mismos para todos los clientes de un perfil.

2. Retornos futuros: block bootstrap del periodo de entrenamiento.
   Para cada trayectoria se remuestrean bloques de ~21 dias (1 mes) del
   historico, preservando la estructura de correlacion entre acciones y
   la autocorrelacion de corto plazo. Se suma un drift diario constante
   segun el escenario.

3. Simulacion mensual (36 meses = 3 anios):
   - Cada mes se aplican los retornos bootstrapeados al portafolio.
   - P2: el cliente acepta o rechaza el rebalanceo de vuelta a w_opt.
   - P1: el cliente abandona si el drawdown supera su tolerancia.
   El proceso se detiene si el cliente abandona.

4. Resultados: 500 trayectorias por (escenario, perfil).
   Se reportan retornos anualizados a 1, 2 y 3 anios (p10/p50/p90)
   y la tasa de abandono acumulada por anio.

Escenarios
----------
  Desfavorable : -10 % anual  =>  mercado sistematicamente peor
  Neutro       :   0 % anual  =>  retornos historicos sin ajuste
  Favorable    : +10 % anual  =>  mercado sistematicamente mejor

Los shocks corresponden a aprox. +/-0.67 sigma de la distribucion
historica anual del S&P 500 (sigma ~ 15 %), escenarios moderados.

Uso
---
    cd E3definitivo
    python proyecciones_main.py

Salidas
-------
    proyecciones_resumen.csv           p10/p50/p90 por escenario/perfil/anio
    output/pdf/proyecciones_fan.pdf    fan chart de trayectorias por perfil
    output/pdf/proyecciones_barras.pdf retornos p50 a 1, 2 y 3 anios
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

from data.loader import load_universe, load_stock_info
from backtesting.runner import (
    _train_window,
    _optimizar,
    _calcular_mu_bl,
    _max_weight_for_profile,
)
from portfolio.engine import PortfolioState, paso_mensual
from config import (
    RISK_PROFILES,
    TRAIN_WINDOW_YEARS,
    COMMISSION,
    BL_METHOD,
    BL_CONF_BASE,
    BL_LOOKBACK,
    BL_SKIP,
)

# ── Parametros ────────────────────────────────────────────────────────────────

TRAIN_END        = "2024-12-31"   # fin del periodo de entrenamiento
PROJECTION_YEARS = 3              # anios hacia adelante
MONTHS_PER_YEAR  = 12
MONTHS           = PROJECTION_YEARS * MONTHS_PER_YEAR  # 36 meses
DAYS_PER_MONTH   = 21             # dias habiles por mes (aprox.)
TOTAL_DAYS       = MONTHS * DAYS_PER_MONTH # 756 dias por trayectoria
BLOCK_SIZE       = 21             # bloque de bootstrap (~1 mes)
N_PATHS          = 500            # trayectorias por escenario/perfil
CAPITAL          = 1_000_000
SEED_BASE        = 42

ESCENARIOS: dict[str, float] = {
    "desfavorable": -0.10,
    "neutro":        0.00,
    "favorable":    +0.10,
}

PERFILES_ORDEN = [
    "muy_conservador", "conservador", "neutro", "arriesgado", "muy_arriesgado"
]

COLORES_ESC = {
    "desfavorable": "#d62728",
    "neutro":       "#1f77b4",
    "favorable":    "#2ca02c",
}

_CLIP = 0.50


# ── Preparacion del modelo ────────────────────────────────────────────────────

def preparar_modelo(
    price_returns: pd.DataFrame,
    div_yields: pd.DataFrame,
    market_caps,
) -> tuple[dict, np.ndarray, list, pd.DataFrame, pd.DataFrame]:
    """
    Entrena BL+Markowitz con la ventana 2020-2024 y devuelve:
      w_opt      : {perfil -> np.ndarray de pesos}
      mu_bl      : senal BL actual (retorno diario esperado por activo)
      tickers    : lista de tickers del universo
      train_pr   : retornos de precio del periodo de entrenamiento
      train_dv   : dividendos del periodo de entrenamiento
    """
    train_end_ts = pd.Timestamp(TRAIN_END)
    train = _train_window(price_returns, train_end_ts + pd.Timedelta(days=1), TRAIN_WINDOW_YEARS)
    train = train.dropna(axis=1, how="any")
    tickers = list(train.columns)

    train_r = train.clip(-_CLIP, _CLIP)

    print("  Calculando Black-Litterman...", end="", flush=True)
    mu_bl = _calcular_mu_bl(
        train_r=train_r,
        market_caps=market_caps,
        conf_base=BL_CONF_BASE,
        lookback=BL_LOOKBACK,
        skip=BL_SKIP,
        bl_method=BL_METHOD,
    )
    print(" listo.")

    print("  Optimizando pesos por perfil (Gurobi)...")
    w_opt = {}
    for perfil, tolerancia in RISK_PROFILES.items():
        max_w = _max_weight_for_profile(None, perfil)
        w_opt[perfil] = _optimizar(
            train_r, tolerancia, perfil,
            mu_bl=mu_bl, max_weight=max_w,
        )
        inversion = w_opt[perfil].sum()
        n_activos = int((w_opt[perfil] > 1e-6).sum())
        print(f"    {perfil:<22} inversion={inversion:.1%}  activos={n_activos}")

    train_dv = div_yields.reindex(columns=tickers).fillna(0)
    train_dv = train_dv.loc[train.index]

    return w_opt, mu_bl, tickers, train_r, train_dv


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def generar_trayectoria(
    train_pr: pd.DataFrame,
    train_dv: pd.DataFrame,
    daily_drift: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Genera una trayectoria de TOTAL_DAYS dias via block bootstrap.

    Aplica daily_drift a cada retorno de precio.
    Devuelve DataFrames con los mismos tickers como columnas.
    """
    T = len(train_pr)
    tickers = list(train_pr.columns)

    indices = []
    while len(indices) < TOTAL_DAYS:
        start = int(rng.integers(0, max(1, T - BLOCK_SIZE)))
        end   = min(start + BLOCK_SIZE, T)
        indices.extend(range(start, end))
    indices = indices[:TOTAL_DAYS]

    pr_vals = train_pr.values[indices] + daily_drift
    dv_vals = train_dv.values[indices]

    return (
        pd.DataFrame(pr_vals, columns=tickers),
        pd.DataFrame(dv_vals, columns=tickers),
    )


# ── Simulacion de una trayectoria ─────────────────────────────────────────────

def simular_cliente(
    w_opt: np.ndarray,
    mu_bl: np.ndarray,
    pr_path: pd.DataFrame,
    dv_path: pd.DataFrame,
    tolerancia: float,
    rng: np.random.Generator,
) -> dict:
    """
    Simula 36 meses de un cliente con pesos fijos w_opt, P1 y P2.

    Retorna dict con:
      valor_mes  : lista de valores al final de cada mes activo
      abandona   : True si el cliente abandono antes del mes 36
      mes_abandono: mes en que abandono (None si completo)
    """
    tickers = list(pr_path.columns)
    state   = PortfolioState.crear(tickers, CAPITAL, w_opt)

    valores   = []
    abandona  = False
    mes_abnd  = None

    for mes in range(MONTHS):
        ini    = mes * DAYS_PER_MONTH
        fin    = ini + DAYS_PER_MONTH
        pr_mes = pr_path.iloc[ini:fin]
        dv_mes = dv_path.iloc[ini:fin]

        m = paso_mensual(
            state, pr_mes, dv_mes, w_opt, tolerancia,
            mu_bl=mu_bl,
            commission_rate=COMMISSION,
            rng=rng,
        )
        valores.append(state.valor)

        if m["abandona"]:
            abandona = True
            mes_abnd = mes + 1
            break

    return {
        "valor_mes":    valores,
        "abandona":     abandona,
        "mes_abandono": mes_abnd,
    }


# ── Correr proyecciones ───────────────────────────────────────────────────────

def correr_proyecciones(
    w_opt: dict,
    mu_bl: np.ndarray,
    train_pr: pd.DataFrame,
    train_dv: pd.DataFrame,
) -> dict:
    """
    Corre N_PATHS trayectorias por (escenario, perfil).

    Retorna:
      resultados[escenario][perfil] = lista de dicts (uno por trayectoria)
    """
    resultados: dict = {}

    for escenario, shock_anual in ESCENARIOS.items():
        daily_drift = (1.0 + shock_anual) ** (1.0 / 252) - 1
        print(f"\n  Escenario {escenario} (shock={shock_anual:+.0%}/anio, "
              f"drift diario={daily_drift:+.5f})...")

        resultados[escenario] = {}

        for perfil in PERFILES_ORDEN:
            tolerancia = RISK_PROFILES[perfil]
            w          = w_opt[perfil]
            trayectorias = []

            for path_i in range(N_PATHS):
                rng = np.random.default_rng(SEED_BASE + path_i * 31 + list(ESCENARIOS).index(escenario) * 7)
                pr_path, dv_path = generar_trayectoria(train_pr, train_dv, daily_drift, rng)
                res = simular_cliente(w, mu_bl, pr_path, dv_path, tolerancia, rng)
                trayectorias.append(res)

            resultados[escenario][perfil] = trayectorias
            n_abnd = sum(t["abandona"] for t in trayectorias)
            print(f"    {perfil:<22} abandono={n_abnd/N_PATHS:.0%}")

    return resultados


# ── Calcular metricas ─────────────────────────────────────────────────────────

def retorno_anualizado(valor_final: float, capital: float, n_anios: float) -> float:
    if valor_final <= 0 or n_anios <= 0:
        return -1.0
    return (valor_final / capital) ** (1.0 / n_anios) - 1.0


def extraer_valor_a_anio(trayectoria: dict, anio: int) -> float:
    """
    Valor del portafolio al final del anio dado.
    Si el cliente abandono antes, retorna el valor en el mes de abandono.
    """
    mes_objetivo = anio * MONTHS_PER_YEAR
    valores = trayectoria["valor_mes"]
    if len(valores) >= mes_objetivo:
        return valores[mes_objetivo - 1]
    # Abandono antes: usar ultimo valor disponible
    return valores[-1] if valores else CAPITAL


def tabla_resumen(resultados: dict) -> pd.DataFrame:
    """
    Construye tabla con columnas:
      escenario, perfil, anio, p10, p25, p50, p75, p90, media,
      tasa_abandono_acumulada
    """
    filas = []
    for escenario, res_esc in resultados.items():
        for perfil in PERFILES_ORDEN:
            trayectorias = res_esc.get(perfil, [])
            if not trayectorias:
                continue
            for anio in [1, 2, 3]:
                mes_objetivo = anio * 12
                retornos = []
                n_abnd   = 0
                for t in trayectorias:
                    val = extraer_valor_a_anio(t, anio)
                    ret = retorno_anualizado(val, CAPITAL, anio)
                    retornos.append(ret)
                    if t["abandona"] and (t["mes_abandono"] or 37) <= mes_objetivo:
                        n_abnd += 1
                retornos = np.array(retornos)
                filas.append({
                    "escenario":        escenario,
                    "perfil":           perfil,
                    "anio":             anio,
                    "media":            round(float(np.mean(retornos)), 4),
                    "p10":              round(float(np.percentile(retornos, 10)), 4),
                    "p25":              round(float(np.percentile(retornos, 25)), 4),
                    "p50":              round(float(np.percentile(retornos, 50)), 4),
                    "p75":              round(float(np.percentile(retornos, 75)), 4),
                    "p90":              round(float(np.percentile(retornos, 90)), 4),
                    "tasa_abandono":    round(n_abnd / len(trayectorias), 4),
                })
    return pd.DataFrame(filas).set_index(["escenario", "perfil", "anio"])


def imprimir_tabla(df: pd.DataFrame) -> None:
    sep = "=" * 90
    for escenario in ESCENARIOS:
        print(f"\n{sep}")
        print(f"  PROYECCION ESCENARIO {escenario.upper()}  "
              f"(drift={ESCENARIOS[escenario]:+.0%}/anio)")
        print(sep)
        print(f"  {'Perfil':<22} {'Anio':>5} {'p10':>8} {'p25':>8} "
              f"{'p50':>8} {'p75':>8} {'p90':>8} {'Media':>8} {'T.Abnd':>8}")
        print(f"  {'-'*22} {'-'*5} {'-'*8} {'-'*8} "
              f"{'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for perfil in PERFILES_ORDEN:
            for anio in [1, 2, 3]:
                try:
                    r = df.loc[(escenario, perfil, anio)]
                except KeyError:
                    continue
                label = perfil.replace("_", " ").title() if anio == 1 else ""
                print(
                    f"  {label:<22} {anio:>5}"
                    f" {r['p10']:>8.1%} {r['p25']:>8.1%}"
                    f" {r['p50']:>8.1%} {r['p75']:>8.1%}"
                    f" {r['p90']:>8.1%} {r['media']:>8.1%}"
                    f" {r['tasa_abandono']:>8.1%}"
                )
    print(f"\n{sep}")


# ── Graficos ──────────────────────────────────────────────────────────────────

def _trayectoria_mediana(trayectorias: list, escenario_drift: float) -> np.ndarray:
    """Valor relativo mediano mes a mes (rellenando con NaN si abandona)."""
    matrix = np.full((len(trayectorias), MONTHS), np.nan)
    for i, t in enumerate(trayectorias):
        vals = t["valor_mes"]
        matrix[i, :len(vals)] = np.array(vals) / CAPITAL
        if len(vals) < MONTHS:
            matrix[i, len(vals):] = vals[-1] / CAPITAL
    return np.nanpercentile(matrix, [10, 50, 90], axis=0)


def plot_fan_chart(resultados: dict, out_path: Path) -> None:
    """
    Fan chart: evolucion del valor relativo del portafolio mes a mes.
    4 subplots (conservador, neutro, arriesgado, muy_arriesgado).
    Por subplot: 3 bandas de color (una por escenario).
    """
    perfiles_plot = ["conservador", "neutro", "arriesgado", "muy_arriesgado"]
    meses_eje = np.arange(1, MONTHS + 1)
    anios_ticks = [12, 24, 36]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, perfil in zip(axes, perfiles_plot):
        for escenario, shock in ESCENARIOS.items():
            trayectorias = resultados.get(escenario, {}).get(perfil, [])
            if not trayectorias:
                continue
            pcts = _trayectoria_mediana(trayectorias, shock)
            color = COLORES_ESC[escenario]
            ax.fill_between(meses_eje, pcts[0], pcts[2],
                            alpha=0.15, color=color)
            ax.plot(meses_eje, pcts[1], color=color, linewidth=2.0,
                    label=escenario.capitalize())

        ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.set_title(perfil.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel("Valor relativo (inicial = 1)")
        ax.set_xlabel("Mes")
        ax.set_xticks(anios_ticks)
        ax.set_xticklabels(["Anio 1", "Anio 2", "Anio 3"])
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Proyeccion 3 anios por escenario de mercado  "
        f"(n={N_PATHS} trayectorias, banda=p10/p90)",
        fontsize=13,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Grafico guardado en: {out_path}")


def plot_barras(df: pd.DataFrame, out_path: Path) -> None:
    """
    Barras agrupadas: retorno p50 a 1, 2 y 3 anios por perfil y escenario.
    """
    perfiles_plot = ["conservador", "neutro", "arriesgado", "muy_arriesgado"]
    escenarios    = list(ESCENARIOS.keys())
    anios         = [1, 2, 3]
    n_anios       = len(anios)
    n_esc         = len(escenarios)
    width         = 0.22
    x             = np.arange(n_anios)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for ax, perfil in zip(axes, perfiles_plot):
        for j, escenario in enumerate(escenarios):
            vals = []
            for anio in anios:
                try:
                    vals.append(df.loc[(escenario, perfil, anio), "p50"])
                except KeyError:
                    vals.append(0.0)
            offset = (j - n_esc / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=escenario.capitalize(),
                          color=COLORES_ESC[escenario], alpha=0.8)

        ax.axhline(0.0, color="grey", linewidth=0.8)
        ax.set_title(perfil.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel("Retorno anualizado (p50)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Anio {a}" for a in anios])
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"Retorno p50 proyectado a 1, 2 y 3 anios por escenario  "
        f"(n={N_PATHS} trayectorias)",
        fontsize=13,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Grafico guardado en: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    print("Cargando datos...")
    price_returns, div_yields = load_universe()
    if price_returns.empty:
        raise FileNotFoundError("No se encontraron datos de precio.")

    try:
        info        = load_stock_info()
        market_caps = info["marketCap"] if "marketCap" in info.columns else None
    except FileNotFoundError:
        market_caps = None

    print(f"  Periodo historico: {price_returns.index[0].date()} -> "
          f"{price_returns.index[-1].date()}")
    print(f"  Entrenamiento    : ultimos {TRAIN_WINDOW_YEARS} anios hasta {TRAIN_END}")
    print(f"  Proyeccion       : {PROJECTION_YEARS} anios ({MONTHS} meses, "
          f"{N_PATHS} trayectorias por escenario/perfil)")

    # 1. Preparar modelo
    print("\n[1/3] Preparando modelo BL + Markowitz...")
    w_opt, mu_bl, tickers, train_pr, train_dv = preparar_modelo(
        price_returns, div_yields, market_caps
    )

    # 2. Correr proyecciones
    print("\n[2/3] Corriendo proyecciones (bootstrap + simulacion)...")
    resultados = correr_proyecciones(w_opt, mu_bl, train_pr, train_dv)

    # 3. Tablas y graficos
    print("\n[3/3] Calculando metricas y generando graficos...")
    df = tabla_resumen(resultados)
    imprimir_tabla(df)

    root = Path(__file__).parent
    df.to_csv(str(root / "proyecciones_resumen.csv"))
    print("\nTabla guardada en: proyecciones_resumen.csv")

    plot_fan_chart(resultados, root / "output" / "pdf" / "proyecciones_fan.pdf")
    plot_barras(df, root / "output" / "pdf" / "proyecciones_barras.pdf")


if __name__ == "__main__":
    main()
