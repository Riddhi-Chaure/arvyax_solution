# src/visualize.py

import numpy as np
import pandas as pd
import joblib
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier

os.makedirs('outputs', exist_ok=True)


def plot_confusion_matrix(clf_state, X_full, y_state, le_state):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_full, y_state, test_size=0.2, random_state=42, stratify=y_state
    )
    clf_state.fit(X_tr, y_tr)
    y_pred = clf_state.predict(X_val)

    cm = confusion_matrix(y_val, y_pred)
    cm_df = pd.DataFrame(cm, index=le_state.classes_, columns=le_state.classes_)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues',
                linewidths=0.5, linecolor='gray')
    plt.title('Emotional State — Confusion Matrix', fontsize=14, pad=12)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=150)
    plt.close()
    print("saved confusion_matrix.png")


def plot_confidence_distribution(clf_state, X_full):
    proba      = clf_state.predict_proba(X_full)
    confidence = proba.max(axis=1)

    plt.figure(figsize=(8, 5))
    plt.hist(confidence, bins=30, color='steelblue', edgecolor='white', linewidth=0.6)
    plt.axvline(x=0.50, color='red', linestyle='--', linewidth=1.5, label='threshold (0.50)')
    plt.axvline(x=confidence.mean(), color='orange', linestyle='--',
                linewidth=1.5, label=f'mean ({confidence.mean():.2f})')

    uncertain_pct = (confidence < 0.50).mean() * 100
    plt.text(0.52, plt.ylim()[1] * 0.85,
             f'{uncertain_pct:.1f}% uncertain', color='red', fontsize=10)

    plt.title('Prediction Confidence Distribution', fontsize=14, pad=12)
    plt.xlabel('Confidence (max class probability)')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/confidence_distribution.png', dpi=150)
    plt.close()
    print("saved confidence_distribution.png")


def plot_accuracy_loss_curve(X_full, y_state):
    # simulate learning curve — train on growing subsets
    # shows how model improves as more data is added
    train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    train_f1s, val_f1s = [], []

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    for size in train_sizes:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_full, y_state, test_size=0.2, random_state=42, stratify=y_state
        )
        # use only `size` fraction of training data
        n = max(int(len(y_tr) * size), 10)
        rf.fit(X_tr[:n], y_tr[:n])

        from sklearn.metrics import f1_score
        train_f1s.append(f1_score(y_tr[:n], rf.predict(X_tr[:n]), average='macro'))
        val_f1s.append(f1_score(y_val,      rf.predict(X_val),    average='macro'))

    sizes_pct = [int(s * 100) for s in train_sizes]

    plt.figure(figsize=(8, 5))
    plt.plot(sizes_pct, train_f1s, 'o-', color='steelblue',
             linewidth=2, markersize=5, label='Train F1')
    plt.plot(sizes_pct, val_f1s,   'o-', color='coral',
             linewidth=2, markersize=5, label='Validation F1')

    # gap between train and val shows overfitting
    plt.fill_between(sizes_pct, train_f1s, val_f1s, alpha=0.1, color='gray')

    plt.title('Learning Curve — Emotional State Model', fontsize=14, pad=12)
    plt.xlabel('Training Data Used (%)')
    plt.ylabel('F1 Score (macro)')
    plt.xticks(sizes_pct)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('outputs/learning_curve.png', dpi=150)
    plt.close()
    print("saved learning_curve.png")


if __name__ == '__main__':
    clf_state = joblib.load('models/clf_state.pkl')
    le_state  = joblib.load('models/le_state.pkl')
    X_full    = sp.load_npz('models/X_train_full.npz')
    train     = pd.read_csv('data/train_clean.csv')
    y_state   = le_state.transform(train['emotional_state'])

    plot_confusion_matrix(clf_state, X_full, y_state, le_state)
    plot_confidence_distribution(clf_state, X_full)
    plot_accuracy_loss_curve(X_full, y_state)

    print("\nall plots saved to outputs/")
