# Project: End-to-End ML Pipeline for Churn Prediction

This project builds a complete machine learning pipeline to predict customer churn using telecom data. It includes data preprocessing, model training, experiment tracking, and deployment as a FastAPI web service.

---

## 📊 Dataset
- Source: Telco Customer Churn (e.g., Kaggle)
- Goal: Predict whether a customer will churn based on service and demographic attributes

---

## ⚙️ Features
- Data preprocessing with `ColumnTransformer`
- RandomForest classifier pipeline
- MLflow logging of metrics and models
- Trained model served with FastAPI

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python src/main.py -c configs/example_random_forest.yaml
```

### 3. Run the API
```bash
uvicorn api.app:app --reload
```

### 4. Access the API Docs
Go to: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📦 Example API Request
**POST** `/predict`
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 5,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.35,
  "TotalCharges": "350.5"
}
```

**Response:**
```json
{
  "prediction": "Yes",
  "churn_probability": 0.58
}
```

---

## 🧪 MLflow Tracking
- Metrics: precision, recall, F1-score, ROC AUC
- Parameters: model type, hyperparameters
- Artifacts: model pipeline

To start the MLflow UI:
```bash
mlflow ui
```
Access at: [http://localhost:5000](http://localhost:5000)
