import ast
import pandas as pd
import numpy as np
from pathlib import Path
from config import DATA_DIR, INFO_FILE, DIV_CAP


def load_universe() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga los CSVs históricos de las 300 acciones.

    Retorna
    -------
    price_returns : DataFrame (fechas x tickers)
        Retorno diario de precio puro: (Close_t - Close_{t-1}) / Close_{t-1}
    div_yields : DataFrame (fechas x tickers)
        Dividend yield diario: Dividends_t / Close_{t-1}, capado en DIV_CAP
    """
    price_returns = {}
    div_yields = {}

    for csv_path in sorted(DATA_DIR.glob("stock_return_*.csv")):
        ticker = csv_path.stem.replace("stock_return_", "")

        df = pd.read_csv(csv_path, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True).tz_convert(None).normalize()
        df = df[["Close", "Dividends"]].sort_index()
        df = df[df["Close"] > 0].dropna()

        prev_close = df["Close"].shift(1)
        price_returns[ticker] = (df["Close"] - prev_close) / prev_close
        div_yields[ticker]    = (df["Dividends"] / prev_close).clip(upper=DIV_CAP)

    price_returns = pd.DataFrame(price_returns).dropna(how="all")
    div_yields    = pd.DataFrame(div_yields).dropna(how="all")

    # alinear índices
    idx = price_returns.index.intersection(div_yields.index)
    return price_returns.loc[idx], div_yields.loc[idx]


def load_stock_info() -> pd.DataFrame:
    """
    Carga stocks_info.txt y retorna un DataFrame con una fila por ticker.
    Columnas útiles: sector, industry, marketCap, beta, dividendYield, etc.
    """
    records = {}
    with open(INFO_FILE, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ticker, raw = line.split(";", 1)
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, dict):
                    records[ticker] = parsed
            except Exception:
                continue

    return pd.DataFrame.from_dict(records, orient="index")


def get_window(
    price_returns: pd.DataFrame,
    eval_year: int,
    train_years: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide los retornos en ventana de entrenamiento y out-of-sample.

    Parámetros
    ----------
    eval_year  : año a evaluar (ej. 2019)
    train_years: cuántos años anteriores usar para entrenar (default 4)

    Retorna
    -------
    train : retornos del período de entrenamiento
    oos   : retornos del año eval_year completo
    """
    train_start = f"{eval_year - train_years}-01-01"
    train_end   = f"{eval_year - 1}-12-31"
    oos_start   = f"{eval_year}-01-01"
    oos_end     = f"{eval_year}-12-31"

    train = price_returns.loc[train_start:train_end].dropna(axis=1, how="any")
    oos   = price_returns.loc[oos_start:oos_end][train.columns]

    return train, oos
