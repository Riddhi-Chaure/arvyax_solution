import pandas as pd
import numpy as np
import os
np.random.seed(42)
n = 1200

df = pd.DataFrame({
    'CustomerID': range(1, n + 1),
    'Tenure':np.random.randint(1, 72, n),
    'MonthlyCharges': np.random.uniform(20.0, 110.0, n),
    'TotalCharges':np.random.uniform(100.0, 5000.0, n),
    'Contract':np.random.choice(['Month-to-month', 'One year', 'Two year'], n),
    'Churn':np.random.randint(0, 2, n)
})

# high charges + no commitment = more likely to leave
mask = (df['MonthlyCharges'] > 90) & (df['Contract'] == 'Month-to-month')
df.loc[mask,'Churn'] = np.random.choice([0, 1],size=mask.sum(),p=[0.2, 0.8]).astype(df['Churn'].dtype)

train =df.iloc[:1000]
test=df.iloc[1000:]

os.makedirs('data', exist_ok=True)
train.to_csv('data/train.csv', index=False)
test.to_csv('data/test.csv', index=False)

print(f"train: {train.shape}, test: {test.shape}")
print(f"churn rate: {df['Churn'].mean():.2%}")