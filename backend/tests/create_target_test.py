import pytest
import pandas as pd
import numpy as np
from pandas import DataFrame
from backend.ml.data.create_target import create_market_target


@pytest.fixture
def sample_ohlcv():
    """Tworzy bazowy DataFrame do testów."""
    data = {"High": [100.0] * 10, "Low": [100.0] * 10, "Close": [100.0] * 10}
    df = DataFrame(data)
    df.index = pd.date_range("2024-01-01", periods=10, freq="h")
    return df


def test_target_take_profit(sample_ohlcv):
    """Testuje, czy cena uderzająca w TP daje 1.0."""
    sample_ohlcv.loc[sample_ohlcv.index[1], "High"] = 102.0

    result = create_market_target(sample_ohlcv, tp_pct=0.015, sl_pct=0.01, window=3)

    assert result.iloc[0] == 1.0


def test_target_stop_loss(sample_ohlcv):
    """Testuje, czy cena uderzająca w SL daje -1.0."""
    sample_ohlcv.loc[sample_ohlcv.index[1], "Low"] = 98.0

    result = create_market_target(sample_ohlcv, tp_pct=0.015, sl_pct=0.01, window=3)

    assert result.iloc[0] == -1.0


def test_target_vertical_barrier(sample_ohlcv):
    """Testuje, czy brak ruchu w oknie window daje 0.0."""
    result = create_market_target(sample_ohlcv, tp_pct=0.015, sl_pct=0.01, window=3)

    # Pierwszy rekord nie dotknął TP ani SL przez 3 świece
    assert result.iloc[0] == 0.0


def test_target_nan_at_end(sample_ohlcv):
    """Testuje, czy ostatnie rekordy (window) są NaN."""
    window_size = 3
    result = create_market_target(sample_ohlcv, window=window_size)

    # Ostatnie 3 rekordy muszą być NaN
    assert result.tail(window_size).isna().all()
    # Rekord tuż przed oknem końcowym powinien być jeszcze liczbą
    assert not np.isnan(result.iloc[-(window_size + 1)])


def test_target_priority_tp_over_sl(sample_ohlcv):
    """Testuje, co się stanie, gdy w jednej świecy cena dotknie obu barier."""
    # W tej samej świecy High=110 i Low=90.
    # Kod sprawdza TP jako pierwszy, więc powinien zwrócić 1.0.
    sample_ohlcv.loc[sample_ohlcv.index[1], "High"] = 110.0
    sample_ohlcv.loc[sample_ohlcv.index[1], "Low"] = 90.0

    result = create_market_target(sample_ohlcv, tp_pct=0.015, sl_pct=0.01, window=3)
    assert result.iloc[0] == 1.0
