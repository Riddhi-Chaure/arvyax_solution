# Intelligent Decision Engine with Uncertainty Handling

Modular, production-ready machine learning system for automated decision making.

## 📌 Project Structure

- `data/`: Contains `train.csv` and `test.csv`.
- `notebooks/`: For exploratory data analysis.
- `src/`: Core logic (preprocessing, training, predicting, uncertainty, decision engine).
- `outputs/`: Prediction results and model artifacts.

## ⚙️ How to Run

1. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Synthetic Data** (if you don't have CSVs):
   ```bash
   python src/data_gen.py
   ```

3. **Train the Model**:
   ```bash
   python src/train.py
   ```

4. **Run Predictions**:
   ```bash
   python src/predict.py
   ```

5. **Analyze Uncertainty**:
   ```bash
   python src/uncertainty.py
   ```

6. **Convert Decisions**:
   ```bash
   python src/decision_engine.py
   ```

Check `outputs/predictions.csv` for the final RecommendedAction for each record.
