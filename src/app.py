from fastapi import FastAPI
import joblib
import pandas as pd


app = FastAPI(
    title="Bank Marketing Prediction API",
    description="Predict whether a customer will subscribe to a term deposit.",
    version="1.0.0"
)

model = joblib.load("models/bank_marketing_model.pkl")


@app.get("/")
def home():
    return {
        "message": "Bank Marketing Prediction API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(data: dict):

    input_data = pd.DataFrame([data])

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    result = "yes" if prediction == 1 else "no"

    return {
        "prediction": result,
        "probability_yes": float(probability)
    }