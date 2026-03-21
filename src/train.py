import pandas as pd
import numpy as np
import joblib
import scipy.sparse as sp
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

os.makedirs('models', exist_ok=True)


def load_everything():
    train = pd.read_csv('data/train_clean.csv')

    X_full= sp.load_npz('models/X_train_full.npz')
    X_text_only= sp.load_npz('models/X_train_text_only.npz')
    X_meta_only= sp.load_npz('models/X_train_meta_only.npz')

    le_state= LabelEncoder()
    y_state= le_state.fit_transform(train['emotional_state'])
    y_intensity = train['intensity'].values

    joblib.dump(le_state, 'models/le_state.pkl')
    return train, X_full, X_text_only, X_meta_only, y_state, y_intensity, le_state


def train_state_model(X_full, y_state, le_state):
    print("TRAINING: Emotional State Classifier")

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=15,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    clf = CalibratedClassifierCV(rf, cv=5, method='sigmoid')

    print("\n5-fold CV...")
    cv = cross_val_score(clf, X_full, y_state, cv=5, scoring='f1_macro', n_jobs=-1)
    print(f"folds: {[round(s,3) for s in cv]}")
    print(f"mean: {cv.mean():.3f}  std: {cv.std():.3f}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_full, y_state, test_size=0.2, random_state=42, stratify=y_state
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_val)

    print("\n", classification_report(y_val, y_pred, target_names=le_state.classes_))

    cm = confusion_matrix(y_val, y_pred)
    print(pd.DataFrame(cm, index=le_state.classes_, columns=le_state.classes_))

    print("\nretraining on full data...")
    clf.fit(X_full, y_state)
    joblib.dump(clf, 'models/clf_state.pkl')
    print("saved clf_state.pkl")
    return clf


def train_intensity_model(X_full, y_intensity):
    print("TRAINING: Intensity Classifier (1-5)")

    # treating as classification not regression — labels are subjective ordinal
    # switched from XGBoost — was collapsing to class 4 with cv=5
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=3,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    clf = CalibratedClassifierCV(rf, cv=3, method='isotonic')

    print("\n5-fold CV...")
    cv = cross_val_score(clf, X_full, y_intensity, cv=5, scoring='f1_macro', n_jobs=-1)
    print(f"folds: {[round(s,3) for s in cv]}")
    print(f"mean: {cv.mean():.3f}  std: {cv.std():.3f}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_full, y_intensity, test_size=0.2, random_state=42, stratify=y_intensity
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_val)

    print("\n", classification_report(y_val, y_pred,
          target_names=['1','2','3','4','5'], zero_division=0))

    print("\nretraining on full data...")
    clf.fit(X_full, y_intensity)
    joblib.dump(clf, 'models/clf_intensity.pkl')
    print("saved clf_intensity.pkl")
    return clf


def run_ablation(X_full, X_text_only, X_meta_only, y_state, y_intensity):
    print("ABLATION STUDY")

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    configs = {
        'text_only':X_text_only,
        'meta_only':X_meta_only,
        'text+meta':X_full
    }

    print(f"\n{'config':<12} {'state_f1':>10} {'intensity_f1':>14}")
    print("-" * 38)

    results = {}
    for name, X in configs.items():
        f1_s = cross_val_score(rf, X, y_state,cv=5, scoring='f1_macro').mean()
        f1_i = cross_val_score(rf, X, y_intensity - 1,cv=5, scoring='f1_macro').mean()
        results[name] = (f1_s, f1_i)
        print(f"{name:<12} {f1_s:>10.3f} {f1_i:>14.3f}")

    print(f"\nbest for state:{max(results, key=lambda k: results[k][0])}")
    print(f"best for intensity:{max(results, key=lambda k: results[k][1])}")
    return results


def show_feature_importance(clf, n_top=20):
    print("FEATURE IMPORTANCE (Top 20)")

    tfidf = joblib.load('models/tfidf.pkl')
    names = list(tfidf.get_feature_names_out()) + [
        'ambience_enc','time_enc','prev_mood_enc','face_enc','refl_quality_enc',
        'duration_min','sleep_hours','energy_level','stress_level',
        'word_count','char_count','is_short',
        'signal_conflict','was_negative_yesterday','stress_energy_gap'
    ]

    importances = np.mean([
        cc.estimator.feature_importances_
        for cc in clf.calibrated_classifiers_
    ], axis=0)

    top = np.argsort(importances)[-n_top:][::-1]
    print(f"\n{'rank':<6} {'feature':<35} {'importance':>10}")
    for rank, idx in enumerate(top, 1):
        print(f"{rank:<6} {names[idx]:<35} {importances[idx]:>10.4f}")

    text_imp = importances[:500].sum()
    meta_imp = importances[500:].sum()
    print(f"\ntext: {text_imp:.3f}  metadata: {meta_imp:.3f}")
    print(f"{'text' if text_imp > meta_imp else 'metadata'} contributes more")


if __name__ == '__main__':
    train, X_full, X_text_only, X_meta_only, \
        y_state, y_intensity, le_state = load_everything()

    clf_state= train_state_model(X_full, y_state, le_state)
    clf_intensity= train_intensity_model(X_full, y_intensity)

    run_ablation(X_full, X_text_only, X_meta_only, y_state, y_intensity)
    show_feature_importance(clf_state)

    print("\nall models saved to models/")