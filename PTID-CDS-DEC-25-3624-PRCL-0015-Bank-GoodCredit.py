#!/usr/bin/env python
# coding: utf-8

# # PTID-CDS-DEC-25-3624-PRCL-0015-Bank-GoodCredit

# ### Importing libraries
# Install pymysql and sqlalchemy
get_ipython().system('pip install pymysql sqlalchemy')
print("pymysql and sqlalchemy installed.")

# Import necessary libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import urllib.parse
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings# ### Initializing configuration
# Disable warnings
warnings.filterwarnings('ignore')

# Disable truncating and show all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)# ### Data loading
# Connection parameters
username = 'dm_team1'
password = 'DM!$Team&279@20!'
encoded_password = urllib.parse.quote(password)
host = '18.136.157.135'
port = '3306'
database = 'project_banking'# Create connection string and engine
connection_string = f"mysql+pymysql://{username}:{encoded_password}@{host}:{port}/{database}"
engine = create_engine(connection_string)
# Load data into DataFrames
ca_df = pd.read_sql("SELECT * FROM Cust_Account", engine)
cd_df = pd.read_sql("SELECT * FROM Cust_Demographics", engine)
ce_df = pd.read_sql("SELECT * FROM Cust_Enquiry", engine)
print("Database connection established and data loaded.")
# Merge tables
merged_df = pd.merge(cd_df, ca_df, on='customer_no', how='left')
merged_df = pd.merge(merged_df, ce_df, on='customer_no', how='left')
print("DataFrames merged.")# ### Data Analysis
# Inspect the first few rows
print(merged_df.head())

# Data Types
print("Data Types:\n", merged_df.dtypes)

# Missing values
print("Missing Values:\n", merged_df.isnull().sum())

# Convert date columns to datetime
merged_df['dt_opened'] = pd.to_datetime(merged_df['dt_opened'], errors='coerce')
merged_df['entry_time'] = pd.to_datetime(merged_df['entry_time'], errors='coerce')

# Summary Statistics
summary_stats = merged_df.describe()
print("Summary Statistics:\n", summary_stats)

# Count of unique customer numbers
unique_customers = merged_df['customer_no'].nunique()
print(f"Unique Customers: {unique_customers}")

# Count of features (e.g., Card Setup counts)
card_setup_count = merged_df['feature_5'].value_counts()
print("Card Setup Count:\n", card_setup_count)# ### Data preprocessing

# #### Feature extraction
# Define desired columns
desired_cols = [
    "high_credit_amt", "cur_balance_amt", "amt_past_due", "creditlimit",
    "acct_type", "owner_indic", "paymenthistory1", "paymenthistory2",
    "paymentfrequency", "actualpaymentamount", "dt_opened_x", "last_paymt_dt",
    "enq_purpose", "enq_amt", "Bad_label"
]

# Filter DataFrame to desired columns
current_cols = merged_df.columns.tolist()
final_desired_cols = [col for col in desired_cols if col in current_cols]
merged_df = merged_df[final_desired_cols].copy()
print(f"Filtered merged_df. Shape: {merged_df.shape}")

print(merged_df.head())

# Date Conversion
for date_col in ['dt_opened_x', 'last_paymt_dt']:
    if date_col in merged_df.columns:
        merged_df[date_col] = pd.to_datetime(merged_df[date_col], errors='coerce')

# Convert Bad_label to numeric
merged_df['Bad_label'] = pd.to_numeric(merged_df['Bad_label'], errors='coerce').fillna(0)

# Ensure all columns are numeric
for col in merged_df.columns:
    merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').fillna(0)# #### Feature encoding
# One-hot encode categorical features
object_cols_to_encode = merged_df.select_dtypes(include='object').columns.tolist()
encoder = OneHotEncoder(sparse_output=True, drop='first', handle_unknown='ignore')
print(f"Identified object columns for encoding: {object_cols_to_encode}")

#Encode features if object columns identified
if object_cols_to_encode:
    encoded_features = encoder.fit_transform(merged_df[object_cols_to_encode])
    merged_df.drop(columns=object_cols_to_encode, inplace=True, errors='ignore')
else:
    encoded_features = csr_matrix(np.empty((merged_df.shape[0], 0)))

joblib.dump(encoder, '/kaggle/working/encoder.pkl')  # Save the encoder

# Separate target variable and convert remaining numeric features to sparse matrix
y = merged_df['Bad_label'].values
X_numeric_sparse = csr_matrix(merged_df.drop(columns=['Bad_label']).values)

# Combine sparse matrices
X_sparse = hstack([X_numeric_sparse, encoded_features])
print(f"Shape of X_sparse: {X_sparse.shape}, Shape of y: {y.shape}")

# Downcast numeric types
print("Downcasting numeric columns for memory efficiency...")
for col in merged_df.columns:
    if pd.api.types.is_numeric_dtype(merged_df[col]):
        if pd.api.types.is_integer_dtype(merged_df[col]):
            merged_df[col] = pd.to_numeric(merged_df[col], downcast='integer', errors='ignore')
        elif pd.api.types.is_float_dtype(merged_df[col]):
            merged_df[col] = pd.to_numeric(merged_df[col], downcast='float', errors='ignore')

# Drop unnecessary columns
irrelevant_cols = ['customer_no']
merged_df.drop(columns=irrelevant_cols, inplace=True, errors='ignore')# ### Model training

# #### Sampling data
# Step 1: Stratified Subsampling
X_train, X_test, y_train, y_test = train_test_split(X_sparse, y, stratify=y, test_size=0.5, random_state=42)

# Step 2: Apply SMOTE to the training set
X_train_dense = X_train.toarray()  # Convert to dense array (make sure you have sufficient memory)
df_train = pd.DataFrame(X_train_dense)

# Add the target variable
df_train['Bad_label'] = y_train

# Separate majority and minority classes
df_majority = df_train[df_train['Bad_label'] == 0]
df_minority = df_train[df_train['Bad_label'] == 1]

# Downsample majority class
df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=42)

# Combine minority class with downsampled majority class
df_reduced = pd.concat([df_majority_downsampled, df_minority])

# Apply SMOTE only on the training data
X_balanced = df_reduced.drop('Bad_label', axis=1).values
y_balanced = df_reduced['Bad_label'].values
smote = SMOTE(sampling_strategy='minority', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_balanced, y_balanced)# #### Training model
# Step 3: Train the RandomForestClassifier model with the resampled data
model = RandomForestClassifier(n_estimators=100, random_state=42, verbose=2, n_jobs=-1)
model.fit(X_resampled, y_resampled)
print("RandomForestClassifier model trained on resampled data.")# #### Exporting model
# Save the trained model
joblib.dump(model, '/kaggle/working/random_forest_model.pkl')# ### Model prediction
# Step 4: Make predictions on the test set
y_pred = model.predict(X_test.toarray())  # Convert X_test to dense if necessary
y_pred_proba = model.predict_proba(X_test.toarray())[:, 1]  # Probability of the positive class (1)

# Print predictions
print(y_pred)
print(y_pred_proba)# ### Model evaluation
# Class distribution
print("\nClass Distribution:")
class_distribution = pd.Series(y_train).value_counts()
print(class_distribution)

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# AUC Score
print("\nROC AUC Score:")
auc = roc_auc_score(y_test, y_pred_proba)
print(auc)# ### Gini score
# Calculating Gini Score
gini_score = 2 * auc - 1

# Print the Gini score
print(f"Gini Score: {gini_score:.4f}")# ### Feature matrix
# One-hot encode categorical features

object_cols_to_encode = merged_df.select_dtypes(include='object').columns.tolist()

encoder = OneHotEncoder(sparse_output=True, drop='first', handle_unknown='ignore')

print(f"Identified object columns for encoding: {object_cols_to_encode}")

encoded_features = encoder.fit_transform(merged_df[object_cols_to_encode])  # Fit here

merged_df.drop(columns=object_cols_to_encode, inplace=True, errors='ignore')

# Get feature names from the fitted encoder
encoded_feature_names = encoder.get_feature_names_out()

# Get feature importances from the trained model
feature_importances = model.feature_importances_

# Combine feature names from both the numeric features and the encoded features
numeric_features = merged_df.drop(columns=['Bad_label']).columns.tolist()
all_feature_names = np.concatenate([numeric_features, encoded_feature_names])

# Create a DataFrame for feature importances
feature_importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': feature_importances
})

# Sort the DataFrame by importance
feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)

# Calculate Gain (for visualization purposes, treated as importance here)
feature_importance_df['Gain'] = feature_importance_df['Importance']

# Display the feature matrix with gain values
print("\nFeature Matrix with Gain Values:")
print(feature_importance_df)# ### Rank ordering
# Combine predictions with desired columns
pred_df = pd.DataFrame({
    'y_pred': y_pred,
    'y_pred_proba': y_pred_proba
})

# Add the features from merged_df
pred_df = pd.concat([pred_df, merged_df[final_desired_cols].reset_index(drop=True)], axis=1)

# Calculate deciles based on the predicted probabilities with duplicate handling
pred_df['decile'] = pd.qcut(pred_df['y_pred_proba'], 10, labels=False, duplicates='drop')

# Calculate average probabilities for each decile
average_probabilities = pred_df.groupby('decile')['y_pred_proba'].mean().sort_index(ascending=False)

# Create a DataFrame from the results for ranking output
ranked_output = pd.DataFrame({
    'Decile Rank': average_probabilities.index + 1,  # Adjust for 1-based ranking
    'Probability': average_probabilities.values
})# Sort the output based on Decile Rank
ranked_output.sort_values(by='Decile Rank', ascending=False, inplace=True)

# Display only the ranked output similar to the desired format
print(ranked_output.reset_index(drop=True))# ### Data visualization

# #### Feature importances
# Visualization of the feature importances (Gain Values)
plt.figure(figsize=(12, 8))
sns.barplot(data=feature_importance_df, x='Gain', y='Feature', palette='viridis')
plt.title('Feature Importances (Gain Values) from Random Forest Model')
plt.xlabel('Gain')
plt.ylabel('Feature')
plt.show()# #### Predicted probabilities
# Visualization of the predicted probabilities after resampling
plt.figure(figsize=(10, 8))
sns.histplot(y_pred_proba, bins=30, kde=True)  # Plotting the distribution of predicted probabilities
plt.title('Predicted Probabilities Distribution After SMOTE Resampling')
plt.xlabel('Predicted Probability of Positive Class')
plt.ylabel('Frequency')
plt.show()# ### Example customer prediction
# Example customer data to be processed
example_customer_data = {
    'customer_no': ['SAMPLE_123'],
    'high_credit_amt': ['10000'],
    'cur_balance_amt': ['5000'],
    'amt_past_due': ['0'],
    'acct_type': ['10'],
    'owner_indic': ['1'],
    'paymenthistory1': ['Y'],
    'paymenthistory2': ['Y'],
    'paymentfrequency': ['Monthly'],
    'actualpaymentamount': ['500'],
    'dt_opened_x': ['01-Jan-2020'],
    'last_paymt_dt': ['01-Jan-2023'],
    'enq_purpose': ['New Credit'],
    'enq_amt': ['15000'],
    'Bad_label': ['0']
}

# Convert sample data to DataFrame
example_customer_df = pd.DataFrame(example_customer_data)

# Function to preprocess example customer data
def preprocess_sample_data(sample_df_raw):
    sample_df = sample_df_raw.copy()

    # Convert date columns to datetime
    for date_col in ['dt_opened_x', 'last_paymt_dt']:
        if date_col in sample_df.columns:
            sample_df[date_col] = pd.to_datetime(sample_df[date_col], errors='coerce')

    # Drop 'Bad_label' if it exists to avoid confusion
    sample_df.drop(columns=['Bad_label'], inplace=True, errors='ignore')

    # Identify object columns for encoding
    sample_object_cols_to_encode = sample_df.select_dtypes(include='object').columns.tolist()
    print("Identified Object Columns for Encoding:", sample_object_cols_to_encode)

    # Check if there are columns to encode
    if sample_object_cols_to_encode:
        encoder = OneHotEncoder(sparse_output=True, drop='first', handle_unknown='ignore')
        print("Shape of Sample Data Before Encoding:", sample_df[sample_object_cols_to_encode].shape)
        
        encoded_features = encoder.fit_transform(sample_df[sample_object_cols_to_encode])
        sample_df.drop(columns=sample_object_cols_to_encode, inplace=True, errors='ignore')
    else:
        print("No object columns found for encoding.")
        encoded_features = csr_matrix(np.empty((sample_df.shape[0], 0)))

    # Fill NaN values and convert remaining columns to numeric
    sample_df.fillna(0, inplace=True)
    sample_df = sample_df.apply(pd.to_numeric, errors='coerce')

    # Convert the numeric part to a sparse matrix
    X_sample_numeric_sparse = csr_matrix(sample_df.values)

    # Combine numeric and encoded features into a final sparse matrix
    X_sample_sparse = hstack([X_sample_numeric_sparse, encoded_features])

    # Ensure that the sparse matrix has the right shape
    if X_sample_sparse.shape[1] != model.n_features_in_:
        padded_x_sample_sparse = csr_matrix((sample_df.shape[0], model.n_features_in_))
        padded_x_sample_sparse[:, :X_sample_sparse.shape[1]] = X_sample_sparse
        return padded_x_sample_sparse

    return X_sample_sparse
# Function to predict credit score
def predict_credit_score(processed_data, model):
    prediction = model.predict(processed_data)
    prediction_proba = model.predict_proba(processed_data)

    cred_score_status = "Good credit history (Low risk of default)" if prediction[0] == 0 else "Bad credit history (High risk of default)"

    print(f"The predicted credit score status for the example customer is: {cred_score_status}")
    print(f"Probability of Good Credit (0): {prediction_proba[0][0]:.4f}")
    print(f"Probability of Bad Credit (1): {prediction_proba[0][1]:.4f}")

    return cred_score_status, prediction_proba

# Predict the credit score of example customer using the pretrained model
model = joblib.load('/kaggle/working/random_forest_model.pkl')
processed_example_data = preprocess_sample_data(example_customer_df)
status, probabilities = predict_credit_score(processed_example_data, model)
