import joblib
import numpy as np
from typing import Literal, Optional
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError


class MusaRandomForestTreeClassifier:
    """

    ## Args:
        :n_estimators (int): Liczba drzew w lesie. Więcej drzew to lepsza stabilność, ale wolniejszy trening (domyślnie 100).
        :max_depth (int): Maksymalna głębokość drzew, chroni przed overfittingiem (domyślnie 10).
        :random_state (int): ustawienie losowości (domyślnie 50)
        :class_weight (Optional[Literal["balanced", "balanced_subsample"]]): Automatyczne doważanie klas (domyślnie'balanced').

    ## Attributes:
        :model (RandomForestClassifier): Rdzeń modelu ze Scikit-Learn.
        :is_trained (bool): Flaga stanu informująca, czy model przeszedł trening.

    ## Methods:
        :train(X_train, y_train):
            Trenuje model na danych historycznych i ustawia flagę is_trained.
        :predict(X) -> ndarray:
            Przewiduje kierunek ruchu ceny (target):
        :predict_proba(X) -> ndarray:
            Zwraca pewność modelu dla każdej z trzech powyższych klas.
        :save(file_path):
            Zapisuje surowy modelu (nie klasę) do pliku .joblib.
        :load(file_path) -> MusaRandomForestTreeClassifier:
            Wczytuje model z dysku i odtwarza stan instancji klasy.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        random_state: int = 50,
        class_weight: Optional[Literal["balanced", "balanced_subsample"]] = "balanced",
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=-1,
        )
        self.is_trained = False

    def train(self, X_train: DataFrame, y_train: Series):
        """Trenuje model i ustawia flagi gotowości."""
        self.model.fit(X_train, y_train)

        self.model.is_trained = True
        self.is_trained = True
        print("Model skończył trening")

    def _check_if_trained(self):
        """Wewnętrzny bezpiecznik sprawdzający stan modelu."""
        if not self.is_trained:
            raise NotFittedError(
                "Ten model nie jest jeszcze wytrenowany! Wywołaj .train() przed predykcją."
            )

    def predict(self, X: DataFrame) -> ndarray:
        """Przewiduje klasy rynkowe (1, 0, -1)."""
        self._check_if_trained()
        return self.model.predict(X)

    def predict_proba(self, X: DataFrame) -> ndarray:
        """Zwraca prawdopodobieństwo dla każdej z klas."""
        self._check_if_trained()
        return self.model.predict_proba(X)

    def save(self, file_path: str):
        """Zapisuje model wewnątrz klasy do pliku .joblib."""
        joblib.dump(self.model, file_path)
        print(f"Model zapisany w: {file_path}")

    @classmethod
    def load(cls, file_path: str):
        """Wczytuje model i odtwarza stan is_trained."""
        instance = cls()
        instance.model = joblib.load(file_path)

        instance.is_trained = getattr(instance.model, "is_trained", False)

        return instance
