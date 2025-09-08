import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Create a FastAPI app instance
app = FastAPI(title="Telco Prediction API", version="1.0")

# 2. Load our trained model
# Make sure you have a trained model file from your previous step
MODEL_PATH = "models/churn_model_v1.pkl"
model = joblib.load(MODEL_PATH)

# 3. Define the input data structure using Pydantic
# This creates data validation and a clear schema for our API
class ChurnFeatures(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    tenure_group: int
    MonthlyCharges_group: int

    class Config:
        schema_extra = {
            "example": {
                'gender': 0,
                'SeniorCitizen': 0,
                'Partner': 1,
                'Dependents': 1,
                'tenure': 59,
                'PhoneService': 1,
                'MultipleLines': 0,
                'InternetService': 0,
                'OnlineSecurity': 0,
                'OnlineBackup': 2,
                'DeviceProtection': 0,
                'TechSupport': 2,
                'Contract': 2,
                'PaperlessBilling': 1,
                'PaymentMethod': 1,
                'MonthlyCharges': 75.95,
                'tenure_group': 3,
                'MonthlyCharges_group': 2}
        }

# 4. Define the prediction endpoint
@app.post("/predict")
def predict_churn(features: ChurnFeatures):
    """
    Takes customer features as input and returns a churn prediction.
    """
    # Convert the Pydantic model to a pandas DataFrame
    # The model was trained on a DataFrame, so it expects that as input
    input_df = pd.DataFrame([features.model_dump()])
    
    # Make a prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0].tolist()

    # Return the result as JSON
    return {
        "prediction": int(prediction),
        "prediction_label": "Churn" if prediction == 1 else "No Churn",
        "class_probabilities": {"No Churn": probability[0], "Churn": probability[1]}
    }

# A simple root endpoint for checking if the API is running
@app.get("/")
def read_root():
    return {"status": "API is running"}
