# src/uncertainty.py

import numpy as np

def get_confidence(clf, X):
    """
    Returns max class probability for each sample.
    Higher = more confident.
    With 6 classes, random baseline = 0.167
    """
    proba = clf.predict_proba(X)       # shape: (n, n_classes)
    confidence = proba.max(axis=1)     # highest prob per row
    return confidence


def get_uncertain_flag(confidence, is_short, signal_conflict,
                       reflection_quality, threshold=0.50):
    """
    Flag a prediction as uncertain when ANY of these are true:
      1. Model confidence below threshold
      2. Input text is very short (<=5 words)
      3. Face emotion contradicts stress level
      4. Reflection quality is 'conflicted'

    threshold=0.50: below 50% confidence on 6-class → genuinely unsure
    """
    low_confidence  = confidence < threshold
    short_text      = np.array(is_short).astype(bool)
    conflict        = np.array(signal_conflict).astype(bool)
    conflicted_refl = np.array(
        [1 if q == 'conflicted' else 0 for q in reflection_quality]
    ).astype(bool)

    uncertain_flag = (
        low_confidence | short_text | conflict | conflicted_refl
    ).astype(int)

    return uncertain_flag