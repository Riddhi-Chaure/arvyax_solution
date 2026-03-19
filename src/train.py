# src/train.py

import pandas as pd
import numpy as np
import joblib
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import os

os.makedirs('models', exist_ok=True)

# ─────────────────────────────────────────
# LOAD DATA & MATRICES
# ─────────────────────────────────────────

def load_everything():
    train = pd.read_csv('data/train_clean.csv')

    X_full      = sp.load_npz('models/X_train_full.npz')
    X_text_only = sp.load_npz('models/X_train_text_only.npz')
    X_meta_only = sp.load_npz('models/X_train_meta_only.npz')

    # Encode targets
    le_state = LabelEncoder()
    y_state  = le_state.fit_transform(train['emotional_state'])
    # calm=0, focused=1, mixed=2, neutral=3, overwhelmed=4, restless=5

    y_intensity = train['intensity'].values  # already 1-5

    joblib.dump(le_state, 'models/le_state.pkl')

    return train, X_full, X_text_only, X_meta_only, y_state, y_intensity, le_state


# ─────────────────────────────────────────
# STEP 4A -- EMOTIONAL STATE CLASSIFIER
# ─────────────────────────────────────────

def train_state_model(X_full, y_state, le_state):
    print("=" * 50)
    print("TRAINING: Emotional State Classifier")
    print("=" * 50)

    # RandomForest -- robust to noisy text, handles mixed signals well
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    # CalibratedClassifierCV -- gives reliable probability scores
    # needed for confidence scores later in uncertainty.py
    clf_state = CalibratedClassifierCV(rf, cv=5, method='sigmoid')

    # Cross-validation -- honest estimate of real performance
    print("\nRunning 5-fold cross validation...")
    cv_scores = cross_val_score(
        clf_state, X_full, y_state,
        cv=5, scoring='f1_macro', n_jobs=-1
    )
    print(f"  CV F1 (macro) per fold : {[round(s,3) for s in cv_scores]}")
    print(f"  Mean F1                : {cv_scores.mean():.3f}")
    print(f"  Std                    : {cv_scores.std():.3f}")

    # Validation split -- to see per-class performance
    print("\nRunning validation split (80/20)...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_full, y_state, test_size=0.2, random_state=42, stratify=y_state
    )
    clf_state.fit(X_tr, y_tr)
    y_pred = clf_state.predict(X_val)

    print("\nClassification Report (Emotional State):")
    print(classification_report(y_val, y_pred,
                                 target_names=le_state.classes_))

    # Confusion matrix -- shows which classes get confused
    print("Confusion Matrix:")
    cm = confusion_matrix(y_val, y_pred)
    cm_df = pd.DataFrame(cm,
                          index=le_state.classes_,
                          columns=le_state.classes_)
    print(cm_df)

    # Final model -- retrain on ALL data for best predictions
    print("\nRetraining on full dataset...")
    clf_state.fit(X_full, y_state)

    joblib.dump(clf_state, 'models/clf_state.pkl')
    print("[OK] Saved -> models/clf_state.pkl")

    return clf_state


# ─────────────────────────────────────────
# STEP 4B -- INTENSITY CLASSIFIER
# ─────────────────────────────────────────

def train_intensity_model(X_full, y_intensity):
    print("\n" + "=" * 50)
    print("TRAINING: Intensity Classifier (1-5)")
    print("=" * 50)

    # Why classification not regression?
    # Intensity labels are subjective ordinal -- not mathematically equidistant
    # "3" vs "4" is a human judgment, not a measured distance

    # Switched from XGBoost to RandomForest here --
    # XGB with CalibratedClassifierCV(cv=5) was collapsing all predictions
    # to class 4 because the calibration folds were too small per-class
    rf_intensity = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=3,
        class_weight='balanced',   # handles any subtle imbalance
        random_state=42,
        n_jobs=-1
    )

    # cv=3 instead of 5 -- more data per fold for this small dataset
    # isotonic works better than sigmoid for ordinal-like targets
    clf_intensity = CalibratedClassifierCV(
        rf_intensity, cv=3, method='isotonic'
    )

    # RF handles 1-5 labels natively, no need to 0-index
    y_intensity_0 = y_intensity

    print("\nRunning 5-fold cross validation...")
    cv_scores = cross_val_score(
        clf_intensity, X_full, y_intensity_0,
        cv=5, scoring='f1_macro', n_jobs=-1
    )
    print(f"  CV F1 (macro) per fold : {[round(s,3) for s in cv_scores]}")
    print(f"  Mean F1                : {cv_scores.mean():.3f}")
    print(f"  Std                    : {cv_scores.std():.3f}")

    # Validation split
    print("\nRunning validation split (80/20)...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_full, y_intensity_0, test_size=0.2, random_state=42,
        stratify=y_intensity_0
    )
    clf_intensity.fit(X_tr, y_tr)
    y_pred = clf_intensity.predict(X_val)

    print("\nClassification Report (Intensity):")
    print(classification_report(y_val, y_pred,
                                 target_names=['1','2','3','4','5'],
                                 zero_division=0))

    # Retrain on full data
    print("\nRetraining on full dataset...")
    clf_intensity.fit(X_full, y_intensity_0)

    joblib.dump(clf_intensity, 'models/clf_intensity.pkl')
    print("[OK] Saved -> models/clf_intensity.pkl")

    return clf_intensity


# ─────────────────────────────────────────
# STEP 4C -- ABLATION STUDY
# ─────────────────────────────────────────

def run_ablation(X_full, X_text_only, X_meta_only, y_state, y_intensity):
    print("\n" + "=" * 50)
    print("ABLATION STUDY")
    print("=" * 50)

    configs = {
        'text_only'  : X_text_only,
        'meta_only'  : X_meta_only,
        'text + meta': X_full
    }

    rf_quick = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    )

    print(f"\n{'Config':<15} {'State F1':>10} {'Intensity F1':>14}")
    print("-" * 42)

    results = {}
    for name, X in configs.items():
        f1_state = cross_val_score(
            rf_quick, X, y_state,
            cv=5, scoring='f1_macro'
        ).mean()

        f1_intensity = cross_val_score(
            rf_quick, X, y_intensity - 1,
            cv=5, scoring='f1_macro'
        ).mean()

        results[name] = (f1_state, f1_intensity)
        print(f"{name:<15} {f1_state:>10.3f} {f1_intensity:>14.3f}")

    # Insight
    best = max(results, key=lambda k: results[k][0])
    print(f"\n-> Best config for state    : {best}")
    best_i = max(results, key=lambda k: results[k][1])
    print(f"-> Best config for intensity: {best_i}")

    return results


# ─────────────────────────────────────────
# STEP 4D -- FEATURE IMPORTANCE
# ─────────────────────────────────────────

def show_feature_importance(clf_state, n_top=20):
    print("\n" + "=" * 50)
    print("FEATURE IMPORTANCE (Top 20)")
    print("=" * 50)

    tfidf    = joblib.load('models/tfidf.pkl')
    encoders = joblib.load('models/encoders.pkl')

    tfidf_names = list(tfidf.get_feature_names_out())
    meta_names  = [
        'ambience_enc', 'time_enc', 'prev_mood_enc',
        'face_enc', 'refl_quality_enc',
        'duration_min', 'sleep_hours', 'energy_level', 'stress_level',
        'word_count', 'char_count', 'is_short',
        'signal_conflict', 'was_negative_yesterday', 'stress_energy_gap'
    ]
    all_names = tfidf_names + meta_names

    # Access underlying RF from calibrated wrapper
    # CalibratedClassifierCV stores fitted estimators inside calibrated_classifiers_
    # Average importances across all calibration folds for stability
    importances = np.mean([
        cc.estimator.feature_importances_
        for cc in clf_state.calibrated_classifiers_
    ], axis=0)

    top_idx = np.argsort(importances)[-n_top:][::-1]

    print(f"\n{'Rank':<6} {'Feature':<35} {'Importance':>10}")
    print("-" * 55)
    for rank, idx in enumerate(top_idx, 1):
        print(f"{rank:<6} {all_names[idx]:<35} {importances[idx]:>10.4f}")

    # Text vs metadata split
    text_importance = importances[:500].sum()
    meta_importance = importances[500:].sum()
    print(f"\nTotal importance -- Text    : {text_importance:.3f}")
    print(f"Total importance -- Metadata: {meta_importance:.3f}")
    print(f"-> {'Text' if text_importance > meta_importance else 'Metadata'} "
          f"contributes more overall")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == '__main__':

    # Load
    train, X_full, X_text_only, X_meta_only, \
        y_state, y_intensity, le_state = load_everything()

    # Train both models
    clf_state     = train_state_model(X_full, y_state, le_state)
    clf_intensity = train_intensity_model(X_full, y_intensity)

    # Ablation study
    ablation_results = run_ablation(
        X_full, X_text_only, X_meta_only, y_state, y_intensity
    )

    # Feature importance
    show_feature_importance(clf_state)

    print("\n[OK] Step 4 Complete -- All models saved to models/")