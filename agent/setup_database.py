import sqlite3
import pandas as pd

#Connect to database (creates the file if it doesn't exist)
conn = sqlite3.connect("collections.db")

# Load your CSV into a pandas DataFrame
data = pd.read_csv("invoices.csv")

# Write the DataFrame to a SQLite table called 'invoices'
data.to_sql("invoices", conn, if_exists="replace", index=False)

print("Database created successfully yo yo!")
print(f"Rows loaded yo: {len(data)}")


#to verify it worked by running a SQL query

result = pd.read_sql("SELECT * FROM invoices", conn)
print(result)

conn.close