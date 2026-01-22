import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from backend.ml.data.fetchers.yahoo_fetcher import fetch_history


@pytest.fixture
def mock_valid_data():
    data = {
        'Open': [100.0, 101.0],
        'High': [102.0, 103.0],
        'Low': [99.0, 100.0],
        'Close': [101.0, 102.0],
        'Volume': [1000, 2000],
        'Adj Close': [101.0, 102.0] 
    }
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(['2023-01-01', '2023-01-02'])
    return df

def test_fetch_history_success(mock_valid_data):
    """
    Scenariusz pozytywny: yfinance zwraca dane, my otrzymujemy czysty DataFrame.
    """

    with patch('backend.ml.data.fetchers.yahoo_fetcher.yf.download') as mock_download:
        mock_download.return_value = mock_valid_data
        
        result = fetch_history("AAPL")
        
        assert not result.empty
        assert len(result) == 2
        assert list(result.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']
        mock_download.assert_called_once()

def test_fetch_history_empty_raises_error():
    """
    Scenariusz negatywny: yfinance zwraca pusty DataFrame -> funkcja ma rzucić ValueError.
    """
    with patch('backend.ml.data.fetchers.yahoo_fetcher.yf.download') as mock_download:
        mock_download.return_value = pd.DataFrame()
        
        with pytest.raises(ValueError, match="CRITICAL"):
            fetch_history("BLEDNY_TICKER")
