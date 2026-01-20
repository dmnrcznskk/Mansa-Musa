import yfinance as yf
import pandas as pd
from typing import Optional

def fetch_history(ticker: str, interval: str = '1d', start_date: Optional[str] = None, period: str = 'max') -> pd.DataFrame:
    """
        Pobiera dane historyczne OHLCV.
        
        Args:
            ticker (str): np. 'AAPL', 'BTC-USD'
            interval (str): '1d', '1h', '15m'
            start_date (str): Format 'YYYY-MM-DD'. Jeśli None, bierze 'period'.
            period (str): '1y', '5y', 'max' (używane gdy brak start_date)
            
        Returns:
            pd.DataFrame: Kolumny [Open, High, Low, Close, Volume] z indeksem Datetime
    """

    if start_date is not None:
        df = yf.download(tickers=ticker, interval=interval, start=start_date, progress=False)
    else:
        df = yf.download(tickers=ticker, interval=interval, period=period, progress=False)

    if df.empty:
        raise ValueError(f"CRITICAL: Nie udało się pobrać danych.")
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[required_cols].copy()
    df.dropna(inplace=True)

    return df

