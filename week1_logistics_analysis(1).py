"""
Week 1 - Strategic Planning and Data Exploration in Logistics
Submitted by: Tannu Kakkad
"""

import pandas as pd

# Load simulated reference logistics dataset
df = pd.read_csv("logistics_week1_reference_dataset.csv")

# Inspect the data
print(df.head())
print(df.info())
print(df.describe())

# Example KPIs
average_delivery_time = df["delivery_time_min"].mean()
average_transport_cost = df["transport_cost"].mean()
on_time_rate = (df["delay_min"] <= 0).mean() * 100

print("Average delivery time:", round(average_delivery_time, 2), "minutes")
print("Average transport cost:", round(average_transport_cost, 2))
print("On-time delivery rate:", round(on_time_rate, 2), "%")

average_shipment_volume = df["shipment_volume_units"].mean()
average_distance = df["distance_km"].mean()

print("Average shipment volume:", round(average_shipment_volume, 2), "units")
print("Average delivery distance:", round(average_distance, 2), "km")

# Simple relationship check
print(df[[
    "distance_km", "shipment_volume_units",
    "delivery_time_min", "transport_cost"
]].corr())
