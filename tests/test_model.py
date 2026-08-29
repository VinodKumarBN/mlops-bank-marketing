import pandas as pd
from sklearn.model_selection import train_test_split

from src.pipeline import load_data


def test_data_loading():
    X, y = load_data.entrypoint()

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape[0] == y.shape[0]


def test_target_values():
    X, y = load_data.entrypoint()

    assert set(y.unique()).issubset({0, 1})


def test_train_test_split():
    X, y = load_data.entrypoint()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    assert len(X_train) > len(X_test)
    assert len(y_train) > len(y_test)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)