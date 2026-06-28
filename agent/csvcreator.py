import pandas as pd

credit_data = {
            "vendor": ["Siemens","Honeywell","ABB","Bosch","Schneider"],
            "credit_limit": [1500000,2000000,1000000,500000,650000],
            "current_exposure": [600000,1500000,],
            "utilisation_pct": [...]

}

df = pd.DataFrame(credit_data)
df.to_csv("agent/credit_balance.csv", index=False)
print("credit_balance.csv created")
print(df)