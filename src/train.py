import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import yaml

df = pd.read_csv("data/bank-full.csv", sep=";")

with open("data/bank-full.csv.dvc", "r") as f:
    dvc_metadata = yaml.safe_load(f)

dvc_data_version = dvc_metadata["outs"][0]["md5"]

X = df.drop("y", axis=1)
y = df["y"].map({"no": 0, "yes": 1})

numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
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

pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", model)
    ]
)

mlflow.set_experiment("Bank Marketing Classification")

with mlflow.start_run():

    mlflow.set_tag("dvc_data_version", dvc_data_version)
    mlflow.log_artifact("data/bank-full.csv.dvc")


    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_params({
        "n_estimators": 200,
        "max_depth": "None",
        "min_samples_split": 5,
        "min_samples_leaf": 1,
        "class_weight": "balanced"
    })

    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

    mlflow.sklearn.log_model(
        pipeline,
        name="bank_marketing_model"
    )

    os.makedirs("models", exist_ok=True)

    joblib.dump(
    pipeline,
    "models/bank_marketing_model.pkl"
)

    print("Model saved to models/bank_marketing_model.pkl")

    print("Model Performance")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)