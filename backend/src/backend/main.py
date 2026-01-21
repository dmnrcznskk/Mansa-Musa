from backend.ml.data.fetchers.yahoo_fetcher import fetch_history
from backend.ml.data.mappers.map_ohlcv_to_features import map_ohlcv_to_features
from backend.ml.data.create_target import create_market_target

import pandas as pd


def start_api() -> None:
    """
    Uruchamia serwer FastAPI
    """
    pass


def start_dev() -> None:
    """
    Funkcja uruchamiająca aplikację w formie dla szkolenia
    i testowania modelów sztucznej intelignecji
    """
    df = fetch_history("AAPL")
    results = map_ohlcv_to_features(df)
    target = create_market_target(df)
    dataset = pd.concat([results, target], axis=1)
    final_dataset = dataset.dropna()

    print(final_dataset.head(30))
    print(final_dataset.tail(0))
