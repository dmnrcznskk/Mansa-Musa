import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

class FeatureScaler:
    def __init__(self):
        self.scaler = StandardScaler()
        self.columns_to_scale = []
        self.is_fitted = False

    def fit_transform(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        Używać TYLKO na zbiorze treningowym (X_train).
        Oblicza średnią i odchylenie, a potem skaluje dane.
        """
        
        self.columns_to_scale = columns
        df_scaled = df.copy()
        scaled_values = self.scaler.fit_transform(df[columns])
        df_scaled[columns] = scaled_values
        self.is_fitted = True
        return df_scaled

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Używać na zbiorze testowym (X_test) lub nowych danych live.
        Używa średniej zapamiętanej z fit_transform.
        """

        if not self.is_fitted:
            raise ValueError("Skaler nie został wytrenowany! Użyj najpierw fit_transform na danych treningowych.")
            
        df_scaled = df.copy()
        scaled_values = self.scaler.transform(df[self.columns_to_scale])
        df_scaled[self.columns_to_scale] = scaled_values
        
        return df_scaled

    def save_scaler(self, path: str):
        """Zapisuje skaler do pliku"""

        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.scaler, path)
        print(f"Skaler zapisany w: {path}")

    def load_scaler(self, path: str, columns: list[str]):
        """Wczytuje skaler"""

        self.scaler = joblib.load(path)
        self.columns_to_scale = columns
        self.is_fitted = True
