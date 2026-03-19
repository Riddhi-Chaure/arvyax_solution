import pandas as pd
import numpy as np
import os
import joblib

def load_resources():
    """Load pre-trained model and preprocessor."""
    model_path = 'outputs/model.joblib'
    preprocessor_path = 'outputs/preprocessor.joblib'
    
    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        raise FileNotFoundError("Model or preprocessor missing. Please run train.py first.")
    
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor

def generate_predictions(df):
    """Generate probability-based predictions using the loaded model."""
    print("Generating predictions...")
    model, preprocessor = load_resources()

    # Drop non-feature columns
    X = df.drop(columns=['CustomerID'])
    X_processed = preprocessor.transform(X)

    # Note: We use predict_proba for better decision making and uncertainty analysis
    probs = model.predict_proba(X_processed)[:, 1]  # Probability of class 1 (Churn)
    preds = (probs >= 0.5).astype(int)              # Baseline 0.5 threshold

    # Keep track of individual customer IDs
    results = pd.DataFrame({
        'CustomerID': df['CustomerID'],
        'Probability': probs,
        'Prediction': preds
    })
    
    return results

def main():
    """Main function to load test data and generate predictions."""
    test_path = 'data/test.csv'
    if not os.path.exists(test_path):
        print(f"Test data not found at {test_path}")
        return

    test_df = pd.read_csv(test_path)
    results = generate_predictions(test_df)

    # Save initial predictions (will be updated by uncertainty and decision engine)
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    results.to_csv('outputs/predictions.csv', index=False)
    print(f"Predictions saved to outputs/predictions.csv")

if __name__ == "__main__":
    main()
