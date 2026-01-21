import pandas as pd
import numpy as np
from pandas import DataFrame, Series


def create_market_target(
    df: DataFrame, tp_pct: float = 0.015, sl_pct: float = 0.01, window: int = 5
) -> Series:
    """
    Generuje klasy dla modelu przy użyciu metody Triple Barrier, symulując realne warunki handlowe (TP/SL/Czas).
    Args:
        df (DataFrame): Dane wejściowe OHLCV. Muszą zawierać kolumny High, Low oraz Close.
        tp_pct (float): Procentowy próg dla bariery górnej (Take Profit).
        sl_pct (float): Procentowy próg dla bariery dolnej (Stop Loss).
        window (int): Maksymalny czas oczekiwania na rozstrzygnięcie transakcji (w godzinach).

    Returns:
        Series (Kolumna target z indeksem Datetime):
            - 1: Zysk.
            - -1: Strata.
            - 0: Nic.
            - NaN: dla braku danych.
    """

    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values

    labels = np.zeros(len(close))

    for i in range(len(close) - window):
        entry_price = close[i]

        upper_barrier = entry_price * (1 + tp_pct)
        lower_barrier = entry_price * (1 - sl_pct)

        for j in range(1, window + 1):
            curr_high = high[i + j]
            curr_low = low[i + j]

            if curr_high >= upper_barrier:
                labels[i] = 1
                break

            if curr_low <= lower_barrier:
                labels[i] = -1
                break

    labels[-window:] = np.nan

    return Series(labels, index=df.index, name="target")
