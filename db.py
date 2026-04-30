# save_from_db.py
import pandas as pd
import urllib.parse
from sqlalchemy import create_engine

username = "dm_team1"
password = "DM!$Team&279@20!"
host = "18.136.157.135"
port = "3306"
database = "project_banking"

encoded_password = urllib.parse.quote(password)
conn_str = f"mysql+pymysql://{username}:{encoded_password}@{host}:{port}/{database}"
engine = create_engine(conn_str)

# adjust query or table names as needed
query = """
SELECT cd.*, ca.*, ce.*
FROM Cust_Demographics cd
LEFT JOIN Cust_Account ca ON cd.customer_no = ca.customer_no
LEFT JOIN Cust_Enquiry ce ON cd.customer_no = ce.customer_no
"""

df = pd.read_sql(query, engine)

# Save to local CSV
df.to_csv("project_banking_db.csv", index=False)
print("Saved project_banking_db.csv")
