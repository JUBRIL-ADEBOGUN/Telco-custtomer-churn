# Data processor module
import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Performs preprocessing on the customer data."""
    df = df.copy()
    df = df.dropna()
    # Convert categorical variables to numeric
    df.loc[:, 'gender'] = df['gender'].apply(lambda x: 1 if x == 'Male' else 0)
    # df.loc[:, 'Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    df.loc[:, 'PaymentMethod'] = df['PaymentMethod'].astype('category').cat.codes
    df.loc[:, 'PaperlessBilling'] = df['PaperlessBilling'].apply(lambda x: 1 if x == 'Yes' else 0)
    df.loc[:, 'Contract'] = df['Contract'].astype('category').cat.codes
    df.loc[:, 'InternetService'] = df['InternetService'].astype('category').cat.codes
    df.loc[:, 'MultipleLines'] = df['MultipleLines'].astype('category').cat.codes
    df.loc[:, 'OnlineSecurity'] = df['OnlineSecurity'].astype('category').cat.codes
    df.loc[:, 'OnlineBackup'] = df['OnlineBackup'].astype('category').cat.codes
    df.loc[:, 'DeviceProtection'] = df['DeviceProtection'].astype('category').cat.codes
    df.loc[:, 'TechSupport'] = df['TechSupport'].astype('category').cat.codes
    df.loc[:, 'StreamingTV'] = df['StreamingTV'].astype('category').cat.codes
    df.loc[:, 'StreamingMovies'] = df['StreamingMovies'].astype('category').cat.codes
    df.loc[:, 'Dependents'] = df['Dependents'].apply(lambda x: 1 if x == 'Yes' else 0)
    df.loc[:, 'Partner'] = df['Partner'].apply(lambda x: 1 if x == 'Yes' else 0)
    df.loc[:, 'SeniorCitizen'] = df['SeniorCitizen'].apply(lambda x: 1 if x == 1 else 0)
    df.loc[:, 'TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.loc[:, 'PhoneService'] = df['PhoneService'].apply(lambda x: 1 if x == 'Yes' else 0)
    df.loc[:, 'tenure_group'] = df['tenure_group'].astype('category').cat.codes
    # df.loc[:, 'tenure_bin'] = df['tenure_bin'].astype('category').cat.codes
    df.loc[:, 'MonthlyCharges_group'] = df['MonthlyCharges_group'].astype('category').cat.codes
    return df


def get_features_and_target(df: pd.DataFrame, features: list, target: str):
    """Separates features and target variable."""
    features  = df.drop(columns=['customerID', 'Churn']).select_dtypes('number').columns.tolist()
    target = 'Churn'

    X = df[features]
    y = df[target]
    return X, y