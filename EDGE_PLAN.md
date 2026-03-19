# Edge Case & Future Handling Plan

## 🧊 Potential Edge Cases
1. **New Categorical values**: If a record has a `Contract` type not seen in training.
   *Resolution*: Use `handle_unknown='ignore'` in `OneHotEncoder`.
2. **Missing TotalCharges**: Often happens for brand-new customers.
   *Resolution*: Implemented `SimpleImputer(strategy='median')` in `preprocess.py`.
3. **Data Drift**: Customer behavior changes over time.
   *Resolution*: Periodically retrain using `train.py`.

## 🚀 Scale Plan
1. **Containerization**: Add a `Dockerfile` for deployment to Cloud.
2. **API Endpoint**: Use `Flask` or `FastAPI` (not included in baseline) to expose `predict.py` as a service.
3. **Advanced Uncertainty**: Use **Conformal Prediction** or **Monte Carlo Dropout** (for Deep Learning) for more rigorous confidence scores.
