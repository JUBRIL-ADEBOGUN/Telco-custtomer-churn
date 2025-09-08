from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_predict_endpoint():
    payload = {
        "tenure": 12,
        "MonthlyCharges": 59.99,
        "TotalCharges": 720.0,
        "gender": 1,
        "Contract": 0,
        "PaymentMethod": 1,
        "PaperlessBilling": 1,
        "InternetService": 2,
        "MultipleLines": 0,
        "OnlineSecurity": 1,
        "OnlineBackup": 0,
        "DeviceProtection": 1,
        "TechSupport": 0,
        "StreamingTV": 1,
        "StreamingMovies": 0,
        "Dependents": 0,
        "Partner": 1,
        "SeniorCitizen": 0,
        "PhoneService": 1,
        "tenure_group": 0,
        "MonthlyCharges_group": 1,
        "avg_monthly_spend": 60.0,
        "tenure_ratio": 0.16,
        "is_new_customer": 1,
        "high_monthly_charges": 0,
        "is_monthly_contract": 1,
        "fiber_new_customer": 0,
        "senior_echeck": 0,
        "paperless_echeck": 1,
        "has_family": 1,
        "protection_score": 2,
        "is_streaming_user": 1,
        "fiber_no_lines": 0,
        "num_services": 5
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
