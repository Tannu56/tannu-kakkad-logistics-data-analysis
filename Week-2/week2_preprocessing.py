"""
Week 2 - Data Collection, Cleaning and Preprocessing
Submitted by: Tannu Kakkad
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("logistics_week2_raw_dataset.csv")

# 1. Inspect missing values and duplicates
print(df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

# 2. Remove duplicates
df = df.drop_duplicates()

# 3. Convert dates
df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
df["delivery_date"] = pd.to_datetime(df["delivery_date"], errors="coerce")

# 4. Handle missing values
df["shipping_cost"] = df["shipping_cost"].fillna(
    df["shipping_cost"].median()
)
df["shipping_mode"] = df["shipping_mode"].fillna(
    df["shipping_mode"].mode()[0]
)

# 5. Feature engineering
df["delivery_time_days"] = (
    df["delivery_date"] - df["order_date"]
).dt.days

# 6. Remove invalid negative durations
df = df[df["delivery_time_days"] >= 0]

# 7. Outlier detection using IQR
q1 = df["shipping_cost"].quantile(0.25)
q3 = df["shipping_cost"].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
outliers = df[
    (df["shipping_cost"] < lower) |
    (df["shipping_cost"] > upper)
]
print("Potential cost outliers:", len(outliers))

# 8. One-hot encoding
df = pd.get_dummies(
    df, columns=["shipping_mode"], drop_first=True
)

# 9. Normalize numerical variables
scaler = MinMaxScaler()
cols = ["distance_km", "shipping_cost",
        "quantity", "delivery_time_days"]
df[cols] = scaler.fit_transform(df[cols])

df.to_csv("logistics_week2_cleaned_dataset.csv", index=False)
print(df.head())
