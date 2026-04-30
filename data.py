# data.py
import pandas as pd
import urllib.parse
from sqlalchemy import create_engine

def load_from_db(username, password, host, port, database):
    encoded_password = urllib.parse.quote(password)
    conn_str = f"mysql+pymysql://{username}:{encoded_password}@{host}:{port}/{database}"
    engine = create_engine(conn_str)
    ca_df = pd.read_sql("SELECT * FROM Cust_Account", engine)
    cd_df = pd.read_sql("SELECT * FROM Cust_Demographics", engine)
    ce_df = pd.read_sql("SELECT * FROM Cust_Enquiry", engine)
    merged = cd_df.merge(ca_df, on='customer_no', how='left').merge(ce_df, on='customer_no', how='left')
    return merged

def load_from_csv(path_or_buffer):
    return pd.read_csv(path_or_buffer)

def select_desired_columns(df, desired_cols):
    current = df.columns.tolist()
    final = [c for c in desired_cols if c in current]
    return df[final].copy()
