import sqlite3
import pandas as pd

conn = sqlite3.connect("assignment.db")

# Load each CSV as a table
yogurt_1 = pd.read_csv("yogurt_production.csv")
yogurt_1.to_sql("yogurt_production", conn, if_exists="replace", index=False)

cheese_1 = pd.read_csv("cheese_production.csv")
cheese_1.to_sql("cheese_production", conn, if_exists="replace", index=False)

honey_1 = pd.read_csv("honey_production.csv")
honey_1.to_sql("honey_production", conn, if_exists="replace", index=False)

milk_1 = pd.read_csv("milk_production.csv")
milk_1.to_sql("milk_production", conn, if_exists="replace", index=False)

coffee_1 = pd.read_csv("coffee_production.csv")
coffee_1.to_sql("coffee_production", conn, if_exists="replace", index=False)

egg_1 = pd.read_csv("egg_production.csv")
egg_1.to_sql("egg_production", conn, if_exists="replace", index=False)

state_1 = pd.read_csv("state_lookup.csv")
state_1.to_sql("state_lookup", conn, if_exists="replace", index=False)

result = pd.read_sql("""
   SELECT COUNT(DISTINCT(State_ANSI)) AS number_state
           FROM cheese
           WHERE CAST(REPLACE(Value,',','') AS INTEGER) > 100000000 
                 AND Period = 'APR'  
                 AND Year = 2023
""", conn)
print(result)
