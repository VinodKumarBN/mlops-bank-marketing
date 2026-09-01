from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "bank-full.csv"
DVC_FILE_PATH = BASE_DIR / "data" / "bank-full.csv.dvc"

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "bank_marketing_model.pkl"


# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH, sep=";")


# --------------------------------------------------
# Get DVC dataset version/hash
# --------------------------------------------------

with open(DVC_FILE_PATH, "r") as f:
    dvc_metadata = yaml.safe_load(f)

dvc_data_version = dvc_metadata["outs"][0]["md5"]


# --------------------------------------------------
# Separate features and target
# --------------------------------------------------

X = df.drop("y", axis=1)
y = df["y"].map({"no": 0, "yes": 1})


# --------------------------------------------------
# Identify feature types
# --------------------------------------------------

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


# --------------------------------------------------
# Train / test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features,
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
    ]
)


# --------------------------------------------------
# Final selected model
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
)


# --------------------------------------------------
# Complete ML pipeline
# --------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", model),
    ]
)


# --------------------------------------------------
# MLflow experiment
# --------------------------------------------------

mlflow.set_experiment("Bank Marketing Classification")


with mlflow.start_run():

    # DVC → MLflow lineage
    mlflow.set_tag(
        "dvc_data_version",
        dvc_data_version,
    )

    mlflow.log_artifact(
        str(DVC_FILE_PATH)
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Parameters
    mlflow.log_params(
        {
            "n_estimators": 200,
            "max_depth": "None",
            "min_samples_split": 5,
            "min_samples_leaf": 1,
            "class_weight": "balanced",
        }
    )

    # Metrics
    mlflow.log_metrics(
        {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )

    # Log model to MLflow
    mlflow.sklearn.log_model(
        pipeline,
        name="bank_marketing_model",
    )

    # Save model for FastAPI / Docker
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        pipeline,
        MODEL_PATH,
    )

    # Output
    print("Model Performance")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    print(
        f"\nModel saved to: {MODEL_PATH}"
    )

    print(
        f"DVC data version: {dvc_data_version}"
    )