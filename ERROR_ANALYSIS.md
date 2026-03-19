# Error Analysis Plan

## 📊 Summary
This document outlines how to analyze gaps in performance of the Intelligent Decision Engine.

## 🔍 Key Performance Indicators
- **Accuracy / F1 Score**: Is the base model reliable?
- **Uncertainty Rate**: What percentage of predictions are being flagged as 'Uncertain' (IsUncertain == 1)?

## 🛠 Analysis Steps
1. **Misclassification Breakdown**:
   Compare `Prediction` vs `Actual` (once ground truth is known).
   Identify if certain segments (e.g., Short-term contracts) are consistently mispredicted.

2. **Confidence-Error Correlation**:
   *Ideally, records with high Confidence scores should have low error rates.*
   If high-confidence predictions are frequently wrong, the model's self-assessment is flawed.

3. **Feature Importance**:
   Use `model.feature_importances_` to see if the engine relies too much on a single feature.
