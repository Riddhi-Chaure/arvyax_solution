import pandas as pd
import numpy as np
import joblib
import scipy.sparse as sp
import sys
import os

sys.path.append(os.path.dirname(__file__))
from decision_engine import decide_what_to_do, decide_when_to_do, generate_message
from uncertainty import get_confidence, get_uncertain_flag


def load_artifacts():
    return (
        joblib.load('models/clf_state.pkl'),
        joblib.load('models/clf_intensity.pkl'),
        joblib.load('models/le_state.pkl'),
        joblib.load('models/tfidf.pkl'),
        joblib.load('models/encoders.pkl'),
    )


def run_predictions(test, X_test, clf_state, clf_intensity, le_state):

    state_labels=le_state.inverse_transform(clf_state.predict(X_test))
    intensity_preds=clf_intensity.predict(X_test)
    conf= get_confidence(clf_state, X_test)

    uncertain = get_uncertain_flag(
        confidence= conf,
        is_short= test['is_short'].values,
        signal_conflict= test['signal_conflict'].values,
        reflection_quality= test['reflection_quality'].values,
        threshold = 0.50
    )

    test = test.reset_index(drop=True)
    rows = []
    for i, row in test.iterrows():
        state= state_labels[i]
        inten= int(intensity_preds[i])
        what= decide_what_to_do(state, inten, int(row['stress_level']), int(row['energy_level']))
        when= decide_when_to_do(row['time_of_day'], state, inten)

        rows.append({
            'id':int(row['id']),
            'predicted_state': state,
            'predicted_intensity': inten,
            'confidence': round(float(conf[i]), 3),
            'uncertain_flag': int(uncertain[i]),
            'what_to_do': what,
            'when_to_do': when,
            'supportive_message': generate_message(state, what, when, inten)
        })

    return pd.DataFrame(rows)


if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)

    clf_state, clf_intensity, le_state, tfidf, encoders = load_artifacts()
    test = pd.read_csv('data/test_clean.csv')
    X_test = sp.load_npz('models/X_test_full.npz')

    df = run_predictions(test, X_test, clf_state, clf_intensity, le_state)

    # quick sanity check
    print(df['predicted_state'].value_counts())
    print(df['predicted_intensity'].value_counts().sort_index())
    print(f"uncertain: {df['uncertain_flag'].sum()}/{ len(df)} \n avg conf: {df['confidence'].mean():.3f}")
    print(df[['id','predicted_state','predicted_intensity','confidence','uncertain_flag','what_to_do','when_to_do']].head())

    df.to_csv('outputs/predictions.csv', index=False)
    print("saved predictions.csv")