from fastapi import FastAPI, Request
import joblib
import pandas as pd
import logging
import time


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


app = FastAPI(
    title="Bank Marketing Prediction API",
    description="Predict whether a customer will subscribe to a term deposit.",
    version="1.0.0"
)

model = joblib.load("models/bank_marketing_model.pkl")


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.perf_counter()

    response = await call_next(request)

    duration_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "method=%s path=%s status=%s latency_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms
    )

    return response


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

    logger.info(
        "prediction=%s probability_yes=%.4f",
        result,
        probability
    )

    return {
        "prediction": result,
        "probability_yes": float(probability)
    }