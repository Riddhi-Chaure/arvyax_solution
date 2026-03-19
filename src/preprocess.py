import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def load_data(file_path):
    """Load data from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    return pd.read_csv(file_path)

def build_preprocessor(numeric_features, categorical_features):
    """
    Build a sklearn ColumnTransformer for preprocessing.
    Handles missing values, scaling, and categorical encoding.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    return preprocessor

def preprocess_main():
    """Main function to preprocess the data."""
    print("Starting preprocessing...")
    
    # Define features
    numeric_features = ['Tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = ['Contract']
    target = 'Churn'

    # Load data
    train_df = load_data('data/train.csv')
    
    X = train_df.drop(columns=[target, 'CustomerID'])
    y = train_df[target]

    # Initialize and fit preprocessor
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_processed = preprocessor.fit_transform(X)

    # Save preprocessor for reuse during prediction
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    joblib.dump(preprocessor, 'outputs/preprocessor.joblib')
    
    print("Preprocessing completed. Preprocessor saved to outputs/preprocessor.joblib")
    return X_processed, y

if __name__ == "__main__":
    preprocess_main()
