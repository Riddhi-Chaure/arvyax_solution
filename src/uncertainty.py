import numpy as np


def get_confidence(clf, X):
    proba = clf.predict_proba(X)
    return proba.max(axis=1)


def get_uncertain_flag(confidence, is_short, signal_conflict,
                       reflection_quality, threshold=0.50):
    # flag uncertain when model is unsure OR input is inherently noisy
    low_conf  = confidence < threshold
    short     = np.array(is_short).astype(bool)
    conflict  = np.array(signal_conflict).astype(bool)
    conflicted = np.array([q == 'conflicted' for q in reflection_quality])

    return (low_conf | short | conflict | conflicted).astype(int)