"""Corre los backtests anuales independientes solo para lambda = 0."""
from pathlib import Path

import pandas as pd

from retornos_anuales_independientes import (
    _load_market_caps,
    _load_universe_with_fallback,
    correr_caso,
)


def main():
    print("Cargando datos...")
    price_returns, div_yields = _load_universe_with_fallback()
    market_caps = _load_market_caps()

    if price_returns.empty:
        raise FileNotFoundError(
            "No se encontraron retornos de acciones. Revisa DATA_DIR en config.py."
        )

    last_date = price_returns.index[-1]
    end_year = last_date.year

    print(f"  Tickers cargados : {price_returns.shape[1]}")
    print(f"  Periodo de datos : {price_returns.index[0].date()} -> {last_date.date()}")
    print(f"  Años evaluados   : 2019 -> {end_year}")
    if last_date.month < 12:
        print(f"  Nota: {end_year} es parcial; datos hasta {last_date.date()}.")

    out_dir = Path("resultados_diagnostico")
    out_dir.mkdir(exist_ok=True)

    tabla = correr_caso("lambda_0", 0.0, price_returns, div_yields, market_caps, end_year)
    tabla.to_csv(out_dir / "lambda_0_anios_independientes.csv", index=False)

    combinado_path = out_dir / "retornos_anuales_independientes.csv"
    if combinado_path.exists():
        combinado = pd.read_csv(combinado_path)
        combinado = combinado[combinado["caso"] != "lambda_0"]
        combinado = pd.concat([combinado, tabla], ignore_index=True)
    else:
        combinado = tabla
    combinado.to_csv(combinado_path, index=False)

    print(f"\nResultados guardados en {out_dir}/")


if __name__ == "__main__":
    main()
