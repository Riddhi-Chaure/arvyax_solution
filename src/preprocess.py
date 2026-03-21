import re
import os
import joblib
import nltk
import pandas as pd
import numpy as np
import scipy.sparse as sp
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack, csr_matrix

nltk.download('stopwords', quiet=True)
STOPWORDS= set(stopwords.words('english'))

CAT_COLS = ['ambience_type','time_of_day','previous_day_mood','face_emotion_hint','reflection_quality']
NUM_COLS = ['duration_min','sleep_hours','energy_level','stress_level','word_count','char_count','is_short','signal_conflict','was_negative_yesterday','stress_energy_gap']


def clean_text(text):
    if pd.isna(text) or str(text).strip() == '':
        return 'empty reflection'
    text = re.sub(r'[^a-z\s]', ' ', str(text).lower())
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [t for t in text.split() if t not in STOPWORDS or t in ['not','no','never']]
    return' '.join(tokens) if tokens else 'empty reflection'


def fill_missing(df):
    df = df.copy()
    df['sleep_hours']= df['sleep_hours'].fillna(df['sleep_hours'].median())
    df['previous_day_mood']= df['previous_day_mood'].fillna('unknown')
    df['face_emotion_hint']= df['face_emotion_hint'].fillna('none')
    return df


def add_features(df):
    df = df.copy()
    df['word_count']= df['journal_text'].str.split().str.len().fillna(0).astype(int)
    df['char_count']=df['journal_text'].str.len().fillna(0).astype(int)
    df['is_short']= (df['word_count'] <= 5).astype(int)

    df['signal_conflict']= ((df['face_emotion_hint'] == 'happy_face') & (df['stress_level'] >= 4)).astype(int)

    df['was_negative_yesterday'] = df['previous_day_mood'].isin(['overwhelmed','restless','mixed']).astype(int)

    df['stress_energy_gap'] = df['stress_level'] - df['energy_level']
    return df


def build_tfidf(train_texts, test_texts, max_features=500):
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1,2),
        min_df=2,
        sublinear_tf= True
    )
    return tfidf, tfidf.fit_transform(train_texts), tfidf.transform(test_texts)


def encode_metadata(df, encoders=None, fit=True):
    df = df.copy()
    if fit:
        encoders= {}
    for col in CAT_COLS:
        if fit:
            le = LabelEncoder()
            df[col+'_enc'] = le.fit_transform(df[col].astype(str))
            encoders[col]= le
        else:
            le = encoders[col]
            df[col+'_enc'] = df[col].astype(str).apply(lambda x:le.transform([x])[0] if x in le.classes_ else -1)
    return df[[c+'_enc' for c in CAT_COLS] + NUM_COLS].values, encoders

if __name__ == '__main__':
    train=pd.read_csv(r'C:\Users\riddh\OneDrive\Coding\Intership\arvyax_solution\data\train.csv')
    test=pd.read_csv(r'C:\Users\riddh\OneDrive\Coding\Intership\arvyax_solution\data\test.csv')

    print(train.columns.tolist())  # confirm columns exist

    train=fill_missing(train)
    train['clean_text']=train['journal_text'].apply(clean_text)
    train=add_features(train)

    test=fill_missing(test)
    test['clean_text']=test['journal_text'].apply(clean_text)
    test =add_features(test)

    print(train.isnull().sum()[train.isnull().sum() > 0])
    print(f"short texts: {train['is_short'].sum()} | conflicts: {train['signal_conflict'].sum()}")
    print(f"\nbefore: {train['journal_text'].iloc[1]}")
    print(f"after :{train['clean_text'].iloc[1]}")

    train.to_csv('data/train_clean.csv',index=False)
    test.to_csv('data/test_clean.csv',index=False)

    tfidf, X_text_train, X_text_test =build_tfidf(train['clean_text'], test['clean_text'])
    X_meta_train, encoders= encode_metadata(train, fit=True)
    X_meta_test, _= encode_metadata(test, encoders=encoders, fit=False)

    X_train_full = hstack([X_text_train,csr_matrix(X_meta_train)])
    X_test_full= hstack([X_text_test,csr_matrix(X_meta_test)])

    print(f"\ntrain matrix: {X_train_full.shape} | test matrix: {X_test_full.shape}")
    print(f"sparsity:{100*(1 - X_train_full.nnz/(X_train_full.shape[0]*X_train_full.shape[1])):.1f}%")

    os.makedirs('models', exist_ok=True)
    joblib.dump(tfidf,'models/tfidf.pkl')
    joblib.dump(encoders,'models/encoders.pkl')

    sp.save_npz('models/X_train_full.npz',X_train_full)
    sp.save_npz('models/X_test_full.npz',X_test_full)
    sp.save_npz('models/X_train_text_only.npz',X_text_train)
    sp.save_npz('models/X_train_meta_only.npz',csr_matrix(X_meta_train))

    print("saved all models and matrices")