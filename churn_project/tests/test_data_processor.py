
# Test cases for data_processor
import pandas as pd
import pytest
from churn_model.data_processor import preprocess_data  # We can import our code!

def test_preprocess_data_handles_na():
    """Tests that NaN values are dropped."""
    test_df = pd.DataFrame({
        'gender': ['Male', None, 'Female'],
        'Tenure': [10, 20, 30],
        'MonthlyCharges': [70.5, 80.0, None],
        'TotalCharges': [705, 1600, 2400],
        'tenure': [10, 20, 30],
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)'],
        'PaperlessBilling': ['Yes', 'No', 'Yes'],
        'InternetService': ['Fiber optic', 'DSL', 'No'],
        'MultipleLines': ['No', 'Yes', 'No'],
        'OnlineSecurity': ['Yes', 'No', 'No'],
        'OnlineBackup': ['No', 'Yes', 'No'],
        'DeviceProtection': ['No', 'Yes', 'No'],
        'TechSupport': ['No', 'Yes', 'No'],
        'StreamingTV': ['Yes', 'No', 'No'],
        'StreamingMovies': ['No', 'Yes', 'No'],
        'Dependents': ['No', 'Yes', 'No'],
        'Partner': ['Yes', 'No', 'Yes'],
        'SeniorCitizen': [1, 0, 1],
        'PhoneService': ['Yes', 'No', 'Yes'],
        'tenure_group': ['0-12', '13-24', '25-48'],
        'MonthlyCharges_group': ['Low', 'Medium', 'High'],
        # Engineered features
        'avg_monthly_spend': [70.5, 80.0, 80.0],
        'tenure_ratio': [0.14, 0.28, 0.42],
        'is_new_customer': [1, 0, 0],
        'high_monthly_charges': [0, 1, 1],
        'is_monthly_contract': [1, 0, 0],
        'fiber_new_customer': [1, 0, 0],
        'senior_echeck': [1, 0, 0],
        'paperless_echeck': [1, 0, 0],
        'has_family': [1, 0, 1],
        'protection_score': [1, 2, 0],
        'is_streaming_user': [1, 0, 0],
        'fiber_no_lines': [1, 0, 0],
        'num_services': [5, 3, 2],
        'Churn': [1, 0, 1],
        'customerID': ['0001', '0002', '0003']
    })
    processed_df = preprocess_data(test_df)
    # Should drop rows with any NaN
    assert processed_df.shape[0] == 1, "Should have dropped rows with NaN"

def test_preprocess_data_encodes_gender():
    """Tests that the Gender column is correctly encoded."""
    test_df = pd.DataFrame({
        'gender': ['Male', 'Female', 'Female', 'Male'],
        'Tenure': [5, 15, 25, 35],
        'MonthlyCharges': [60.0, 70.0, 80.0, 90.0],
        'TotalCharges': [300, 1050, 2000, 3150],
        'tenure': [5, 15, 25, 35],
        'Contract': ['Month-to-month', 'One year', 'Two year', 'Month-to-month'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
        'PaperlessBilling': ['Yes', 'No', 'Yes', 'No'],
        'InternetService': ['Fiber optic', 'DSL', 'No', 'Fiber optic'],
        'MultipleLines': ['No', 'Yes', 'No', 'Yes'],
        'OnlineSecurity': ['Yes', 'No', 'No', 'Yes'],
        'OnlineBackup': ['No', 'Yes', 'No', 'Yes'],
        'DeviceProtection': ['No', 'Yes', 'No', 'Yes'],
        'TechSupport': ['No', 'Yes', 'No', 'Yes'],
        'StreamingTV': ['Yes', 'No', 'No', 'Yes'],
        'StreamingMovies': ['No', 'Yes', 'No', 'Yes'],
        'Dependents': ['No', 'Yes', 'No', 'Yes'],
        'Partner': ['Yes', 'No', 'Yes', 'No'],
        'SeniorCitizen': [1, 0, 1, 0],
        'PhoneService': ['Yes', 'No', 'Yes', 'No'],
        'tenure_group': ['0-12', '13-24', '25-48', '49-72'],
        'MonthlyCharges_group': ['Low', 'Medium', 'High', 'Very High'],
        # Engineered features
        'avg_monthly_spend': [60.0, 70.0, 80.0, 90.0],
        'tenure_ratio': [0.07, 0.21, 0.35, 0.49],
        'is_new_customer': [1, 0, 0, 0],
        'high_monthly_charges': [0, 1, 1, 1],
        'is_monthly_contract': [1, 0, 0, 1],
        'fiber_new_customer': [1, 0, 0, 1],
        'senior_echeck': [1, 0, 0, 0],
        'paperless_echeck': [1, 0, 0, 0],
        'has_family': [1, 0, 1, 0],
        'protection_score': [1, 2, 0, 3],
        'is_streaming_user': [1, 0, 0, 1],
        'fiber_no_lines': [1, 0, 0, 1],
        'num_services': [5, 3, 2, 6],
        'Churn': [0, 1, 0, 1],
        'customerID': ['0004', '0005', '0006', '0007']
    })
    processed_df = preprocess_data(test_df)
    assert all(processed_df['gender'].isin([0, 1])), "Gender should be 0 or 1"