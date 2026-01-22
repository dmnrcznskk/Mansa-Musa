import pytest
import pandas as pd
import numpy as np
import os
from backend.ml.data.feature_scaler  import FeatureScaler

@pytest.fixture
def market_data():
    """Tworzy DataFrame imitujący dane OHLCV + wskaźniki."""
    dates = pd.date_range(start="2024-01-01", periods=10, freq="1h")
    data = {
        'Close': [42000, 42100, 41900, 42500, 43000, 42800, 43200, 43100, 43500, 44000],
        'Volume': [150, 200, 100, 400, 350, 250, 300, 150, 500, 450],
        'RSI': [30, 45, 40, 60, 70, 65, 75, 55, 80, 85],
        'Symbol': ['BTCUSDT'] * 10
    }
    return pd.DataFrame(data, index=dates)

def test_financial_features_scaling(market_data):
    """
    Sprawdza, czy wolumen i RSI są poprawnie zeskalowane.
    W tradingu to kluczowe, żeby duże liczby (Volume) nie zdominowały modelu.
    """
    scaler = FeatureScaler()
    features = ['Volume', 'RSI']
    
    scaled_df = scaler.fit_transform(market_data, features)
    
    assert np.isclose(scaled_df['Volume'].mean(), 0, atol=1e-7)
    assert np.isclose(scaled_df['Volume'].std(ddof=0), 1, atol=1e-7)
    
    assert not scaled_df['Volume'].equals(market_data['Volume'])

def test_index_preservation(market_data):
    """
    KRYTYCZNE DLA GIEŁDY: Sprawdza, czy po skalowaniu nie zgubiliśmy dat (indeksu).
    StandardScaler z sklearna zwraca numpy array i kasuje indeksy - Twoja klasa ma to naprawiać.
    """
    scaler = FeatureScaler()
    scaled_df = scaler.fit_transform(market_data, ['Close'])
    
    pd.testing.assert_index_equal(scaled_df.index, market_data.index)

def test_price_raw_vs_scaled_separation(market_data):
    """Sprawdza, czy jak skalujemy wskaźniki, to cena (Close) zostaje nietknięta."""
    scaler = FeatureScaler()
    cols_to_scale = ['Volume', 'RSI'] 
    
    scaled_df = scaler.fit_transform(market_data, cols_to_scale)
    
    pd.testing.assert_series_equal(scaled_df['Close'], market_data['Close'])
    pd.testing.assert_series_equal(scaled_df['Symbol'], market_data['Symbol'])

def test_train_test_split_consistency(market_data):
    """
    Symulacja backtestu:
    Trenujemy na danych z "przeszłości", testujemy na "przyszłości".
    Zapobiega Data Leakage (nie używamy średniej z przyszłości do skalowania przeszłości).
    """
    scaler = FeatureScaler()
    features = ['Close', 'Volume']
    
    train_size = 6
    train_df = market_data.iloc[:train_size]
    test_df = market_data.iloc[train_size:]
    
    scaler.fit_transform(train_df, features)
    
    scaled_test = scaler.transform(test_df)
    
    raw_val = test_df.iloc[0]['Volume']
    
    train_mean = train_df['Volume'].mean()
    train_std = train_df['Volume'].std(ddof=0)
    expected_scaled_val = (raw_val - train_mean) / train_std
    
    actual_scaled_val = scaled_test.iloc[0]['Volume']
    
    assert np.isclose(actual_scaled_val, expected_scaled_val)

def test_save_load_model_weights(market_data, tmp_path):
    """Sprawdza czy możemy zapisać wagi skalera (mean/std) i użyć ich na produkcji."""
    scaler = FeatureScaler()
    features = ['RSI']
    scaler.fit_transform(market_data, features)
    
    path = tmp_path / "production_scaler.joblib"
    scaler.save_scaler(str(path))
    
    new_scaler = FeatureScaler()
    new_scaler.load_scaler(str(path), features)
    
    result_original = scaler.transform(market_data)
    result_loaded = new_scaler.transform(market_data)
    
    pd.testing.assert_frame_equal(result_original, result_loaded)
