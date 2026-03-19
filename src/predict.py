# src/predict.py

import pandas as pd
import numpy as np
import joblib
import scipy.sparse as sp
import sys
import os

# make src imports work
sys.path.append(os.path.dirname(__file__))

from decision_engine import decide_what_to_do, decide_when_to_do, generate_message
from uncertainty import get_confidence, get_uncertain_flag

# ─────────────────────────────────────────
# LOAD ALL SAVED ARTIFACTS
# ─────────────────────────────────────────

def load_artifacts():
    clf_state     = joblib.load('models/clf_state.pkl')
    clf_intensity = joblib.load('models/clf_intensity.pkl')
    le_state      = joblib.load('models/le_state.pkl')
    tfidf         = joblib.load('models/tfidf.pkl')
    encoders      = joblib.load('models/encoders.pkl')
    return clf_state, clf_intensity, le_state, tfidf, encoders


# ─────────────────────────────────────────
# LOAD TEST FEATURES
# ─────────────────────────────────────────

def load_test_data():
    test         = pd.read_csv('data/test_clean.csv')
    X_test_full  = sp.load_npz('models/X_test_full.npz')
    return test, X_test_full


# ─────────────────────────────────────────
# GENERATE PREDICTIONS
# ─────────────────────────────────────────

def generate_predictions(test, X_test_full,
                          clf_state, clf_intensity,
                          le_state):

    # Predict emotional state
    state_preds  = clf_state.predict(X_test_full)
    state_labels = le_state.inverse_transform(state_preds)

    # Predict intensity (already 1-5, no offset needed)
    intensity_preds = clf_intensity.predict(X_test_full)

    # Confidence scores
    conf_state = get_confidence(clf_state, X_test_full)

    # Uncertain flags
    uncertain = get_uncertain_flag(
        confidence         = conf_state,
        is_short           = test['is_short'].values,
        signal_conflict    = test['signal_conflict'].values,
        reflection_quality = test['reflection_quality'].values,
        threshold          = 0.50
    )

    # Decision engine + message
    results = []
    for i in range(len(test)):
        row   = test.iloc[i]
        state = state_labels[i]
        inten = int(intensity_preds[i])

        what = decide_what_to_do(
            state, inten,
            int(row['stress_level']),
            int(row['energy_level'])
        )
        when = decide_when_to_do(
            row['time_of_day'], state, inten
        )
        msg = generate_message(state, what, when, inten)

        results.append({
            'id'                  : int(row['id']),
            'predicted_state'     : state,
            'predicted_intensity' : inten,
            'confidence'          : round(float(conf_state[i]), 3),
            'uncertain_flag'      : int(uncertain[i]),
            'what_to_do'          : what,
            'when_to_do'          : when,
            'supportive_message'  : msg
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────
# VERIFY OUTPUT
# ─────────────────────────────────────────

def verify_predictions(df):
    print("=== PREDICTION VERIFICATION ===\n")
    print(f"Total predictions       : {len(df)}")
    print(f"Unique states predicted : {df['predicted_state'].nunique()}")
    print(f"\nState distribution:")
    print(df['predicted_state'].value_counts())
    print(f"\nIntensity distribution:")
    print(df['predicted_intensity'].value_counts().sort_index())
    print(f"\nUncertain flags         : {df['uncertain_flag'].sum()} "
          f"({df['uncertain_flag'].mean()*100:.1f}%)")
    print(f"Avg confidence          : {df['confidence'].mean():.3f}")
    print(f"Min confidence          : {df['confidence'].min():.3f}")
    print(f"\nWhat to do distribution:")
    print(df['what_to_do'].value_counts())
    print(f"\nWhen to do distribution:")
    print(df['when_to_do'].value_counts())
    print(f"\nSample predictions (5 rows):")
    pd.set_option('display.max_colwidth', 60)
    print(df[['id','predicted_state','predicted_intensity',
              'confidence','uncertain_flag',
              'what_to_do','when_to_do']].head())


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == '__main__':
    import os
    os.makedirs('outputs', exist_ok=True)

    print("Loading artifacts...")
    clf_state, clf_intensity, le_state, tfidf, encoders = load_artifacts()

    print("Loading test data...")
    test, X_test_full = load_test_data()

    print("Generating predictions...\n")
    predictions = generate_predictions(
        test, X_test_full,
        clf_state, clf_intensity, le_state
    )

    verify_predictions(predictions)

    # Save final deliverable
    predictions.to_csv('outputs/predictions.csv', index=False)
    print("\n✅ Saved → outputs/predictions.csv")
