from backend.ml.data.fetchers.yahoo_fetcher import fetch_history
from backend.ml.data.mappers.map_ohlcv_to_features import map_ohlcv_to_features


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
    print(results)
