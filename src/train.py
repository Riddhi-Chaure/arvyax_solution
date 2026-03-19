import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from src.preprocess import preprocess_main

def train_model(X, y):
    """Train a Random Forest model on preprocessed data."""
    print("Training ML model...")
    # Initialize classifier
    # Using Random Forest as it provides robust probabilities for uncertainty handling
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    
    # Simple validation on training data
    y_pred = model.predict(X)
    print(f"Training Accuracy: {accuracy_score(y, y_pred):.4f}")
    print("Classification Report:\n", classification_report(y, y_pred))
    
    # Save the trained model
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    joblib.dump(model, 'outputs/model.joblib')
    print("Model saved to outputs/model.joblib")
    return model

def main():
    """Execute preprocessing and training."""
    # Step 1: Preprocess data
    X_processed, y = preprocess_main()

    # Step 2: Train the model
    train_model(X_processed, y)

if __name__ == "__main__":
    main()
