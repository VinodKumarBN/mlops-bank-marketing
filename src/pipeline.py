import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from zenml import pipeline, step


@step
def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv("data/bank-full.csv", sep=";")

    X = df.drop("y", axis=1)
    y = df["y"].map({"no": 0, "yes": 1})

    return X, y


@step
def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2
) -> tuple[Pipeline, pd.DataFrame, pd.Series]:

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_features
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )

    model_pipeline.fit(X_train, y_train)

    return model_pipeline, X_test, y_test


@step
def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> tuple[float, float, float, float]:

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    return accuracy, precision, recall, f1


@pipeline
def bank_marketing_pipeline(test_size: float = 0.2):
    X, y = load_data()

    model, X_test, y_test = train_model(
        X,
        y,
        test_size=test_size
    )

    evaluate_model(
        model,
        X_test,
        y_test
    )


if __name__ == "__main__":
    bank_marketing_pipeline(test_size=0.2)