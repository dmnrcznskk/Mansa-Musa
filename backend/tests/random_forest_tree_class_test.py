import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.exceptions import NotFittedError
from backend.ml.architectures.random_forest_tree_class import (
    MusaRandomForestTreeClassifier,
)
import os


@pytest.fixture
def dummy_data():
    """Tworzy minimalny zestaw danych do testu."""
    X = pd.DataFrame(np.random.rand(10, 4), columns=["f1", "f2", "f3", "f4"])
    y = pd.Series([1, 0, -1, 0, 1, 0, -1, 0, 1, 0])
    return X, y


def test_initialization():
    """Sprawdza czy model startuje jako niewytrenowany."""
    bot = MusaRandomForestTreeClassifier()
    assert bot.is_trained is False


def test_training_flow(dummy_data):
    """Sprawdza czy proces treningu poprawnie ustawia flagi."""
    X, y = dummy_data
    bot = MusaRandomForestTreeClassifier()

    bot.train(X, y)

    assert bot.is_trained is True
    assert bot.model.is_trained is True


def test_prediction_safety(dummy_data):
    """Sprawdza czy model blokuje predykcję przed treningiem."""
    X, _ = dummy_data
    bot = MusaRandomForestTreeClassifier()

    with pytest.raises(NotFittedError):
        bot.predict(X)


def test_save_and_load_persistence(dummy_data):
    """Sprawdza czy model 'pamięta' stan po wczytaniu z dysku (wersja os)."""
    X, y = dummy_data
    file_name = "test_model_temporary.joblib"

    bot = MusaRandomForestTreeClassifier()
    bot.train(X, y)

    try:
        bot.save(file_name)
        loaded_bot = MusaRandomForestTreeClassifier.load(file_name)

        assert loaded_bot.is_trained is True

        predictions = loaded_bot.predict(X)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)

    finally:
        if os.path.exists(file_name):
            os.remove(file_name)
