import pytest
import pandas as pd
import numpy as np
from backend.ml.data.mappers.map_ohlcv_to_features import map_ohlcv_to_features
from pandas import DataFrame


@pytest.fixture
def mock_ohlcv_data_long() -> DataFrame:
    """
    Tworzy wystarczająco 50 wierszy danych testowych,
    aby wskaźniki techniczne mogły się wyliczyć.
    """
    periods = 50
    data = {
        "Open": np.linspace(100, 150, periods),
        "High": np.linspace(102, 152, periods),
        "Low": np.linspace(98, 148, periods),
        "Close": np.linspace(101, 151, periods),
        "Volume": np.random.randint(1000, 5000, periods),
    }
    df = pd.DataFrame(data)
    df.index = pd.date_range(start="2024-01-01", periods=periods, freq="h")
    return df


def test_map_ohlcv_to_features_success(mock_ohlcv_data_long):
    """
    Testuje czy mapper poprawnie wylicza wszystkie kolumny i usuwa NaN.
    """

    result = map_ohlcv_to_features(mock_ohlcv_data_long)

    assert not result.empty
    assert len(result) > 0

    expected_columns = [
        "log_returns",
        "rsi",
        "macd",
        "macd_hist",
        "atr",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
    ]
    assert all(col in result.columns for col in expected_columns)

    assert result.isna().sum().sum() == 0


def test_map_ohlcv_to_features_cyclical_bounds(mock_ohlcv_data_long):
    """
    Testuje czy wartości sin/cos mieszczą się w poprawnym zakresie rynkowym [-1, 1].
    """
    result = map_ohlcv_to_features(mock_ohlcv_data_long)

    time_cols = ["hour_sin", "hour_cos", "day_sin", "day_cos"]
    for col in time_cols:
        assert result[col].max() <= 1.0
        assert result[col].min() >= -1.0


def test_map_ohlcv_log_returns_exact_value():
    periods = 100

    prices = [100.0] * (periods - 1) + [110.0]
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": prices,
            "Low": prices,
            "Close": prices,
            "Volume": [1000] * periods,
        },
        index=pd.date_range("2024-01-01", periods=periods, freq="D"),
    )

    result = map_ohlcv_to_features(df)

    expected_log_return = np.log(110 / 100)
    actual_value = result["log_returns"].iloc[-1]

    assert actual_value == pytest.approx(expected_log_return, rel=1e-5)


def test_map_ohlcv_time_exact_midnight():
    periods = 40
    df = pd.DataFrame(
        {
            "Open": [100] * periods,
            "High": [100] * periods,
            "Low": [100] * periods,
            "Close": [100] * periods,
            "Volume": [100] * periods,
        }
    )

    df.index = pd.to_datetime(["2024-01-01 00:00:00"] * periods)

    result = map_ohlcv_to_features(df)

    assert result["hour_sin"].iloc[-1] == pytest.approx(0.0, abs=1e-9)
    assert result["hour_cos"].iloc[-1] == pytest.approx(1.0, abs=1e-9)


def test_map_ohlcv_rsi_behavior():
    periods = 50
    prices_up = [100.0 + i for i in range(periods)]

    df_up = pd.DataFrame(
        {
            "Open": prices_up,
            "High": prices_up,
            "Low": prices_up,
            "Close": prices_up,
            "Volume": [1000] * periods,
        },
        index=pd.date_range("2024-01-01", periods=periods, freq="D"),
    )

    result = map_ohlcv_to_features(df_up)

    assert result["rsi"].iloc[-1] > 70
    assert result["rsi"].max() <= 100


def test_map_ohlcv_atr_volatility_jump():
    periods = 50
    prices = [100.0 + (i % 2) for i in range(40)]
    prices += [200.0 + (i * 50) for i in range(10)]

    df = pd.DataFrame(
        {
            "Open": prices,
            "High": [p + 2 for p in prices],
            "Low": [p - 2 for p in prices],
            "Close": prices,
            "Volume": [1000] * 50,
        },
        index=pd.date_range("2024-01-01", periods=50, freq="D"),
    )

    result = map_ohlcv_to_features(df)

    assert result["atr"].iloc[-1] > result["atr"].iloc[10]
    assert (result["atr"] > 0).all()


def test_map_ohlcv_macd_direction():
    periods = 50
    prices_down = [200.0 - (i * 2) for i in range(periods)]
    df_down = pd.DataFrame(
        {
            "Open": prices_down,
            "High": prices_down,
            "Low": prices_down,
            "Close": prices_down,
            "Volume": [1000] * periods,
        },
        index=pd.date_range("2024-01-01", periods=periods, freq="D"),
    )

    result = map_ohlcv_to_features(df_down)

    assert result["macd"].iloc[-1] < 0
    assert result["macd_hist"].iloc[-1] < 0
