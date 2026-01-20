from pandas import DataFrame
import numpy as np
from ta.momentum import rsi
from ta.trend import macd, macd_diff
from ta.volatility import average_true_range


def map_ohlcv_to_features(df: DataFrame) -> DataFrame:
    """
    Funkcja mapująca dane OHLCV na zestaw cech (Input X).

    Oblicza zwroty logarytmiczne oraz metryki trendów dla każdego indexu wraz z godziną i dniem w forcmacie cyklicznym (sin/cos)

    Args:
        df (DataFrame): [Open, High, Low, Close, Volume] z indexem Datetime.

    Returns:
        DataFrame: [Date, log_returns, rsi, macd, macd_hist, atr, hour_sin, hour_cos, day_sin, day_cos].
    """

    df = df.sort_index()

    X = DataFrame(index=df.index)

    X["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))

    X["rsi"] = rsi(close=df["Close"], window=14)

    X["macd"] = macd(close=df["Close"])
    X["macd_hist"] = macd_diff(close=df["Close"])

    X["atr"] = average_true_range(
        high=df["High"], low=df["Low"], close=df["Close"], window=14
    )

    def time_function(value: int, devider: int) -> float:
        return 2 * np.pi * value / devider

    hour = df.index.hour
    hour_func = time_function(hour, 24)
    X["hour_sin"] = np.sin(hour_func)
    X["hour_cos"] = np.cos(hour_func)

    day_of_week = df.index.dayofweek
    day_of_week_func = time_function(day_of_week, 7)
    X["day_sin"] = np.sin(day_of_week_func)
    X["day_cos"] = np.cos(day_of_week_func)

    return X.dropna().copy()
