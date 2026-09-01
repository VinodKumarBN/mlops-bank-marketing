# Bank Marketing MLOps Project

An end-to-end MLOps project that predicts whether a bank customer will subscribe to a term deposit and demonstrates the complete machine learning lifecycle.

## Problem Statement

Predict whether a bank customer will subscribe to a term deposit using customer and campaign information.

## Machine Learning

- Classification problem
- Target: `y`
- Baseline: Logistic Regression
- Models compared: Logistic Regression, Decision Tree, Random Forest, KNN
- Final candidate: Balanced Random Forest
- F1 Score: 0.6161
- Recall: 0.7486

## MLOps Stack

- Git & GitHub — version control
- DVC — dataset versioning
- MLflow — experiment tracking and model registry
- ZenML — ML pipeline orchestration
- Pytest — automated testing
- GitHub Actions — CI/CD
- Docker — containerization
- GHCR — container registry
- FastAPI — model serving
- Render — cloud deployment

## Project Workflow

```text
Data
 ↓
DVC
 ↓
Preprocessing
 ↓
Model Training
 ↓
MLflow Tracking
 ↓
Model Registry
 ↓
ZenML Pipeline
 ↓
Pytest
 ↓
GitHub Actions CI
 ↓
Docker
 ↓
GHCR
 ↓
Render Deployment
 ↓
FastAPI
 ↓
Monitoring
 ↓
Data Drift Detection
 ↓
Automated Retraining
