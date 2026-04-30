# model.py
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import joblib
import numpy as np
import pandas as pd

def stratified_subsample_and_resample(X_sparse, y, test_size=0.5, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X_sparse, y, stratify=y, test_size=test_size, random_state=random_state)
    X_train_dense = X_train.toarray()
    df_train = pd.DataFrame(X_train_dense)
    df_train['Bad_label'] = y_train
    df_majority = df_train[df_train['Bad_label'] == 0]
    df_minority = df_train[df_train['Bad_label'] == 1]
    if len(df_minority) == 0:
        raise ValueError("No minority class examples found in training data.")
    df_majority_down = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=random_state)
    df_reduced = pd.concat([df_majority_down, df_minority])
    X_bal = df_reduced.drop('Bad_label', axis=1).values
    y_bal = df_reduced['Bad_label'].values
    sm = SMOTE(sampling_strategy='minority', random_state=random_state)
    X_res, y_res = sm.fit_resample(X_bal, y_bal)
    return X_res, y_res, X_test, y_test

def train_random_forest(X_res, y_res, n_estimators=100, random_state=42, n_jobs=-1):
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, verbose=0, n_jobs=n_jobs)
    clf.fit(X_res, y_res)
    return clf

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)
