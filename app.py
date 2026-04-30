# app.py
import streamlit as st
import pandas as pd
import joblib
from data import load_from_db, select_desired_columns
from features import convert_dates, to_numeric_fill, encode_features, build_feature_matrix
from model import stratified_subsample_and_resample, train_random_forest, save_model, load_model
from utils import pad_features_if_needed
from scipy.sparse import csr_matrix, hstack
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Credit Score ML (DB)", layout="wide")
st.title("Bank Credit Score - Load from Database")

# --- Database connection inputs ---
with st.form("db_form"):
    st.write("Enter database connection details")
    username = st.text_input("DB username", value="dm_team1")
    password = st.text_input("DB password", type="password", value="DM!$Team&279@20!")
    host = st.text_input("DB host", value="18.136.157.135")
    port = st.text_input("DB port", value="3306")
    database = st.text_input("DB name", value="project_banking")
    submitted = st.form_submit_button("Load data from DB")

if submitted:
    try:
        with st.spinner("Connecting and loading data..."):
            df = load_from_db(username, password, host, port, database)
        st.success("Data loaded from database.")
        st.write("Preview:", df.head())
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

    desired_cols = [
        "high_credit_amt", "cur_balance_amt", "amt_past_due", "creditlimit",
        "acct_type", "owner_indic", "paymenthistory1", "paymenthistory2",
        "paymentfrequency", "actualpaymentamount", "dt_opened_x", "last_paymt_dt",
        "enq_purpose", "enq_amt", "Bad_label"
    ]
    df_sel = select_desired_columns(df, desired_cols)
    df_sel = convert_dates(df_sel, ['dt_opened_x', 'last_paymt_dt'])
    df_sel = to_numeric_fill(df_sel, target_col='Bad_label')
    st.write("Preprocessed preview:", df_sel.head())

    # Encode and build features
    encoder_path = "encoder.pkl"
    df_encoded, encoded, encoder = encode_features(df_sel, encoder_path=encoder_path, fit_encoder=True)
    X, y = build_feature_matrix(df_encoded, encoded)
    st.write(f"Feature matrix shape: {X.shape}, Target shape: {None if y is None else y.shape}")

    # Train button
    if st.button("Train model"):
        try:
            with st.spinner("Training model (this may take a while)..."):
                X_res, y_res, X_test, y_test = stratified_subsample_and_resample(X, y)
                model = train_random_forest(X_res, y_res)
                save_model(model, "random_forest_model.pkl")
                if encoder is not None:
                    joblib.dump(encoder, encoder_path)
            st.success("Model trained and saved.")
        except Exception as e:
            st.error(f"Training failed: {e}")

    # Simple evaluation if model exists
    if os.path.exists("random_forest_model.pkl"):
        model = load_model("random_forest_model.pkl")
        try:
            X_res_tmp, y_res_tmp, X_test, y_test = stratified_subsample_and_resample(X, y)
            y_pred = model.predict(X_test.toarray())
            y_pred_proba = model.predict_proba(X_test.toarray())[:, 1]
            from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
            st.subheader("Evaluation on holdout test set")
            st.write("Confusion Matrix")
            st.write(confusion_matrix(y_test, y_pred))
            st.write("Classification Report")
            st.text(classification_report(y_test, y_pred))
            st.write("ROC AUC:", roc_auc_score(y_test, y_pred_proba))
        except Exception as e:
            st.warning(f"Could not evaluate model: {e}")

    st.markdown("---")
    st.subheader("Predict single customer from loaded data")
    row_idx = st.number_input("Row index to predict", min_value=0, max_value=max(0, df_sel.shape[0]-1), value=0)
    example_df = df_sel.iloc[[row_idx]].drop(columns=['Bad_label'], errors='ignore')
    st.write(example_df)

    if st.button("Predict row"):
        try:
            encoder = joblib.load("encoder.pkl") if os.path.exists("encoder.pkl") else None
            model = load_model("random_forest_model.pkl")
            obj_cols = example_df.select_dtypes(include='object').columns.tolist()
            if obj_cols and encoder is not None:
                encoded_example = encoder.transform(example_df[obj_cols])
                example_numeric = example_df.drop(columns=obj_cols)
            else:
                encoded_example = csr_matrix((example_df.shape[0], 0))
                example_numeric = example_df
            X_sample_sparse = hstack([csr_matrix(example_numeric.values), encoded_example])
            X_sample_sparse = pad_features_if_needed(X_sample_sparse, model)
            pred = model.predict(X_sample_sparse.toarray())
            proba = model.predict_proba(X_sample_sparse.toarray())[0]
            st.write("Prediction (0=Good,1=Bad):", int(pred[0]))
            st.write("Probabilities:", {"good(0)": float(proba[0]), "bad(1)": float(proba[1])})
        except Exception as e:
            st.error(f"Prediction failed: {e}")
