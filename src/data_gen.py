import pandas as pd
import numpy as np
import os

def generate_synthetic_data():
    np.random.seed(42)
    n_samples = 1200
    
    # Simple churn prediction dataset
    data = {
        'CustomerID': range(1, n_samples + 1),
        'Tenure': np.random.randint(1, 72, n_samples),
        'MonthlyCharges': np.random.uniform(20.0, 110.0, n_samples),
        'TotalCharges': np.random.uniform(100.0, 5000.0, n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'Churn': np.random.randint(0, 2, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce some logical correlation
    # If MonthlyCharges > 90 and Contract is Month-to-month, Churn is more likely
    df.loc[(df['MonthlyCharges'] > 90) & (df['Contract'] == 'Month-to-month'), 'Churn'] = \
        np.random.choice([0, 1], size=len(df[(df['MonthlyCharges'] > 90) & (df['Contract'] == 'Month-to-month')]), p=[0.2, 0.8])
    
    # Split into train/test
    train_df = df.iloc[:1000]
    test_df = df.iloc[1000:]
    
    # Save to data directory
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Generated synthetic data in {train_path} and {test_path}")

if __name__ == "__main__":
    generate_synthetic_data()
