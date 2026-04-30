# features.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix, hstack
import joblib

def convert_dates(df, date_cols):
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    return df

def to_numeric_fill(df, target_col='Bad_label'):
    if target_col in df.columns:
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce').fillna(0)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

def encode_features(df, encoder_path=None, fit_encoder=True):
    object_cols = df.select_dtypes(include='object').columns.tolist()
    if object_cols:
        encoder = OneHotEncoder(sparse_output=True, drop='first', handle_unknown='ignore')
        if not fit_encoder and encoder_path:
            encoder = joblib.load(encoder_path)
            encoded = encoder.transform(df[object_cols])
        else:
            encoded = encoder.fit_transform(df[object_cols])
            if encoder_path:
                joblib.dump(encoder, encoder_path)
        df = df.drop(columns=object_cols, errors='ignore')
    else:
        encoded = csr_matrix(np.empty((df.shape[0], 0)))
        encoder = None
    return df, encoded, encoder

def build_feature_matrix(df, encoded):
    y = df['Bad_label'].values if 'Bad_label' in df.columns else None
    X_numeric_sparse = csr_matrix(df.drop(columns=['Bad_label']).values) if 'Bad_label' in df.columns else csr_matrix(df.values)
    X = hstack([X_numeric_sparse, encoded])
    return X, y
