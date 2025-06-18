from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn

app = FastAPI(title="Churn Prediction API")

# Load model
model = joblib.load("models/churn_model.pkl")

# Define input schema
class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def read_root():
    return {"message": "Churn Prediction API is running."}

@app.post("/predict")
def predict_churn(features: CustomerFeatures):
    data = pd.DataFrame([features.dict()])
    data['customerID'] = "Dummy"
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]
    return {
        "prediction": prediction,
        "churn_probability": round(probability, 3)
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
