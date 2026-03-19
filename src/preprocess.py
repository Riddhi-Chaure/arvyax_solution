# src/preprocess.py
import re
import nltk
import pandas as pd
import numpy as np

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

# ─────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────

def clean_text(text):
    # handle empty or missing
    if pd.isna(text) or str(text).strip() == '':
        return 'empty reflection'

    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)    # remove punctuation, numbers
    text = re.sub(r'\s+', ' ', text).strip()  # normalize whitespace

    # keep negation words -- they flip meaning ("not calm" != "calm")
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS or t in ['not', 'no', 'never']]

    return ' '.join(tokens) if tokens else 'empty reflection'


# ─────────────────────────────────────────
# MISSING VALUE HANDLER
# ─────────────────────────────────────────

def fill_missing(df):
    df = df.copy()

    # numerical — median is robust against outliers
    df['sleep_hours'] = df['sleep_hours'].fillna(df['sleep_hours'].median())

    # categorical — 'unknown' tells the model "this was missing"
    # better than dropping the row or guessing
    df['previous_day_mood'] = df['previous_day_mood'].fillna('unknown')
    df['face_emotion_hint'] = df['face_emotion_hint'].fillna('none')

    return df


# ─────────────────────────────────────────
# EXTRA FEATURES (engineered from raw data)
# ─────────────────────────────────────────

def add_features(df):
    df = df.copy()

    # text length signals — short text = harder to predict
    df['word_count']  = df['journal_text'].str.split().str.len().fillna(0).astype(int)
    df['char_count']  = df['journal_text'].str.len().fillna(0).astype(int)
    df['is_short']    = (df['word_count'] <= 5).astype(int)  # flag for uncertainty

    # conflict signal — face says happy but stress is high?
    df['signal_conflict'] = (
        (df['face_emotion_hint'] == 'happy_face') & (df['stress_level'] >= 4)
    ).astype(int)

    # mood recovery — was bad yesterday but writing today?
    negative_moods = ['overwhelmed', 'restless', 'mixed']
    df['was_negative_yesterday'] = df['previous_day_mood'].isin(negative_moods).astype(int)

    # energy-stress gap — high stress + low energy = burnout signal
    df['stress_energy_gap'] = df['stress_level'] - df['energy_level']

    return df


# ─────────────────────────────────────────
# VERIFICATION HELPER
# ─────────────────────────────────────────

def verify_cleaning(train, test):
    print("=== PREPROCESSING VERIFICATION ===\n")

    # 1. Missing values should be gone
    print("Missing values after cleaning:")
    cols_to_check = ['sleep_hours', 'previous_day_mood', 'face_emotion_hint']
    for col in cols_to_check:
        t_missing = train[col].isnull().sum()
        te_missing = test[col].isnull().sum() if col in test.columns else 'N/A'
        print(f"  {col:25s} -> train: {t_missing}  | test: {te_missing}")

    # 2. Clean text should have no empty strings
    empty_train = (train['clean_text'] == 'empty reflection').sum()
    empty_test  = (test['clean_text']  == 'empty reflection').sum()
    print(f"\n  'empty reflection' replacements -> train: {empty_train} | test: {empty_test}")

    # 3. Show before vs after for 3 samples
    print("\nBefore vs After cleaning (3 samples):")
    for i in [0, 1, 2]:
        print(f"\n  Original : {train['journal_text'].iloc[i]}")
        print(f"  Cleaned  : {train['clean_text'].iloc[i]}")

    # 4. New feature columns
    new_features = ['word_count','char_count','is_short',
                    'signal_conflict','was_negative_yesterday','stress_energy_gap']
    print("\nNew engineered features:")
    print(train[new_features].describe().round(2))

    # 5. How many short texts flagged?
    print(f"\nShort texts flagged (is_short=1): {train['is_short'].sum()}")
    print(f"Signal conflicts detected: {train['signal_conflict'].sum()}")


# ─────────────────────────────────────────
# MAIN — run everything
# ─────────────────────────────────────────

if __name__ == '__main__':

    # Load
    train = pd.read_csv(r'C:\Users\riddh\OneDrive\Coding\Intership\arvyax_solution\data\train.csv')
    test  = pd.read_csv(r'C:\Users\riddh\OneDrive\Coding\Intership\arvyax_solution\data\test.csv')

    # Fill missing
    train = fill_missing(train)
    test  = fill_missing(test)

    # Clean text
    train['clean_text'] = train['journal_text'].apply(clean_text)
    test['clean_text']  = test['journal_text'].apply(clean_text)

    # Add engineered features
    train = add_features(train)
    test  = add_features(test)

    # Verify everything looks right
    verify_cleaning(train, test)

    # Save cleaned versions
    train.to_csv('data/train_clean.csv', index=False)
    test.to_csv('data/test_clean.csv',   index=False)

    print("\n[OK] Saved -> data/train_clean.csv and data/test_clean.csv")

# ─────────────────────────────────────────
# STEP 3 — TF-IDF + METADATA + COMBINATION
# ─────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack, csr_matrix
import joblib
import os

# ── 3A. TF-IDF VECTORIZATION ──────────────

def build_tfidf(train_texts, test_texts, max_features=500):
    """
    Convert cleaned journal text → numeric vectors
    max_features=500: keep top 500 most informative words/bigrams
    ngram_range=(1,2): single words AND two-word combos
    e.g. "not calm", "deep work", "racing thoughts"
    """
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),   # unigrams + bigrams
        min_df=2,             # ignore words appearing in <2 docs (noise)
        sublinear_tf=True     # dampens effect of very frequent words
    )

    X_text_train = tfidf.fit_transform(train_texts)
    X_text_test  = tfidf.transform(test_texts)

    return tfidf, X_text_train, X_text_test


# ── 3B. METADATA ENCODING ─────────────────

CAT_COLS = ['ambience_type', 'time_of_day', 'previous_day_mood',
            'face_emotion_hint', 'reflection_quality']

NUM_COLS = ['duration_min', 'sleep_hours', 'energy_level', 'stress_level',
            'word_count', 'char_count', 'is_short',
            'signal_conflict', 'was_negative_yesterday', 'stress_energy_gap']

def encode_metadata(df, encoders=None, fit=True):
    """
    Encode categorical + numerical metadata into a numeric matrix.
    fit=True  → learn encodings from train
    fit=False → apply learned encodings to test
    """
    df = df.copy()

    if fit:
        encoders = {}

    for col in CAT_COLS:
        if fit:
            le = LabelEncoder()
            df[col + '_enc'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            # handle any unseen category in test gracefully
            df[col + '_enc'] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    enc_cols = [c + '_enc' for c in CAT_COLS] + NUM_COLS
    return df[enc_cols].values, encoders


# ── 3C. COMBINE TEXT + METADATA ───────────

def combine_features(X_text, X_meta):
    """
    Stack TF-IDF (sparse) and metadata (dense) side by side.
    Result shape: (n_samples, 500 + 15) = (n_samples, 515)
    """
    return hstack([X_text, csr_matrix(X_meta)])


# ── 3D. VERIFICATION ──────────────────────

def verify_features(tfidf, X_train_full, X_test_full,
                    X_text_train, X_meta_train):

    print("=== STEP 3: FEATURE VERIFICATION ===\n")

    print(f"TF-IDF vocabulary size      : {len(tfidf.vocabulary_)}")
    print(f"TF-IDF feature matrix train : {X_text_train.shape}")

    print(f"\nMetadata features           : {len(NUM_COLS) + len(CAT_COLS)} columns")
    print(f"Metadata matrix shape       : {X_meta_train.shape}")

    print(f"\nCombined train matrix shape : {X_train_full.shape}")
    print(f"Combined test  matrix shape : {X_test_full.shape}")

    # show top 20 TF-IDF words — these are most informative words
    feature_names = tfidf.get_feature_names_out()
    print(f"\nSample TF-IDF features (first 20):")
    print(list(feature_names[:20]))

    print(f"\nSample TF-IDF features (last 20 — likely bigrams):")
    print(list(feature_names[-20:]))

    # sparsity check — high sparsity is normal for TF-IDF
    total_elements = X_train_full.shape[0] * X_train_full.shape[1]
    nonzero = X_train_full.nnz
    sparsity = 100 * (1 - nonzero / total_elements)
    print(f"\nMatrix sparsity             : {sparsity:.1f}%  (normal for TF-IDF)")


# ── 3E. MAIN ──────────────────────────────

if __name__ == '__main__':

    # Load cleaned data from Step 2
    train = pd.read_csv('data/train_clean.csv')
    test  = pd.read_csv('data/test_clean.csv')

    # 3A — TF-IDF
    tfidf, X_text_train, X_text_test = build_tfidf(
        train['clean_text'], test['clean_text'], max_features=500
    )

    # 3B — Metadata
    X_meta_train, encoders = encode_metadata(train, fit=True)
    X_meta_test, _         = encode_metadata(test, encoders=encoders, fit=False)

    # 3C — Combine
    X_train_full = combine_features(X_text_train, X_meta_train)
    X_test_full  = combine_features(X_text_test,  X_meta_test)

    # Also keep text-only and meta-only for ablation study later
    X_train_text_only = X_text_train
    X_train_meta_only = csr_matrix(X_meta_train)

    # 3D — Verify
    verify_features(tfidf, X_train_full, X_test_full,
                    X_text_train, X_meta_train)

    # Save objects for use in train.py
    os.makedirs('models', exist_ok=True)
    joblib.dump(tfidf,    'models/tfidf.pkl')
    joblib.dump(encoders, 'models/encoders.pkl')

    # Save matrices
    import scipy.sparse as sp
    sp.save_npz('models/X_train_full.npz', X_train_full)
    sp.save_npz('models/X_test_full.npz',  X_test_full)
    sp.save_npz('models/X_train_text_only.npz', X_train_text_only)
    sp.save_npz('models/X_train_meta_only.npz', X_train_meta_only)

    print("\n✅ Saved → models/tfidf.pkl, encoders.pkl, all matrix .npz files")
