# preprocessing.py
import pandas as pd
import joblib

def load_feature_columns(path="models/feature_columns.pkl"):
    return joblib.load(path)

def preprocess_input(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    # Drop customerID and gender
    df = df.drop(['customerID', 'gender'], axis=1, errors='ignore')

    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])

    # One-hot encoding
    one_hot_cols = ['Contract', 'PaymentMethod', 'InternetService']
    original_cols = df.columns.tolist()
    df_encoded = pd.get_dummies(df[one_hot_cols], prefix=one_hot_cols, drop_first=False)
    df = df.drop(columns=one_hot_cols)
    df = pd.concat([df, df_encoded], axis=1)

    # Convert boolean columns to integers
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Convert Yes/No to 1/0 and handle dtype
    df.replace({'Yes': 1, 'No': 0}, inplace=True)
    df = df.infer_objects(copy=False)

    # Add missing columns
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Ensure correct order
    df = df[feature_columns]

    return df

def determine_group_and_trim(df: pd.DataFrame):
    if df.loc[0, 'PhoneService'] == 1 and df.loc[0, 'InternetService_No'] == 0:
        group = "both"
        df = df.drop(columns=['PhoneService', 'InternetService_No'], errors='ignore')
    elif df.loc[0, 'PhoneService'] == 1 and df.loc[0, 'InternetService_No'] == 1:
        group = "only_phone"
        df = df.drop(columns=[
            'PhoneService', 'InternetService_DSL', 'InternetService_Fiber optic',
            'InternetService_No', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ], errors='ignore')
    elif df.loc[0, 'PhoneService'] == 0 and df.loc[0, 'InternetService_No'] == 0:
        group = "only_internet"
        df = df.drop(columns=[
            'PhoneService', 'InternetService_No', 'MultipleLines'
        ], errors='ignore')
    else:
        group = "unknown"
    
    return group, df
