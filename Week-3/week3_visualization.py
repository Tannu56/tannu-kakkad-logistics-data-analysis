"""Week 3 - Advanced Data Analysis and Visualization in Logistics
Submitted by: Tannu Kakkad
"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("logistics_week3_dataset.csv", parse_dates=["date"])
print(df.head())
print(df.describe())
numeric_cols = ["distance_km","shipment_volume_units","delivery_time_min","transport_cost","delay_min"]
print(df[numeric_cols].corr())

on_time_rate = (df["delay_min"] <= 0).mean() * 100
print("On-time delivery rate:", round(on_time_rate, 2), "%")

plt.figure(figsize=(8,5))
plt.hist(df["delivery_time_min"], bins=18)
plt.title("Distribution of Delivery Time")
plt.xlabel("Delivery Time (minutes)")
plt.ylabel("Number of Shipments")
plt.tight_layout()
plt.savefig("visualizations/01_delivery_time_distribution.png", dpi=180)
plt.close()

monthly = df.set_index("date")["shipment_volume_units"].resample("MS").sum()
plt.figure(figsize=(8,5))
plt.plot(monthly.index, monthly.values, marker="o")
plt.title("Monthly Shipment Volume Trend")
plt.xlabel("Month"); plt.ylabel("Shipment Volume (units)")
plt.tight_layout(); plt.savefig("visualizations/02_monthly_shipment_volume.png", dpi=180); plt.close()

plt.figure(figsize=(8,5))
for vehicle in ["Bike","Van","Truck"]:
    s=df[df["vehicle_type"]==vehicle]
    plt.scatter(s["distance_km"],s["delivery_time_min"],label=vehicle,alpha=.65)
plt.title("Distance vs Delivery Time by Vehicle Type")
plt.xlabel("Distance (km)"); plt.ylabel("Delivery Time (minutes)"); plt.legend()
plt.tight_layout(); plt.savefig("visualizations/03_distance_vs_delivery.png",dpi=180); plt.close()

cv=df.groupby("vehicle_type")["transport_cost"].mean()
plt.figure(figsize=(7,5)); plt.bar(cv.index,cv.values)
plt.title("Average Transportation Cost by Vehicle Type")
plt.xlabel("Vehicle Type"); plt.ylabel("Average Cost")
plt.tight_layout(); plt.savefig("visualizations/04_cost_by_vehicle.png",dpi=180); plt.close()

ct=df.groupby("traffic_level")["delivery_time_min"].mean().reindex(["Low","Medium","High"])
plt.figure(figsize=(7,5)); plt.bar(ct.index,ct.values)
plt.title("Average Delivery Time by Traffic Level")
plt.xlabel("Traffic Level"); plt.ylabel("Average Delivery Time (minutes)")
plt.tight_layout(); plt.savefig("visualizations/05_delivery_by_traffic.png",dpi=180); plt.close()

cor=df[numeric_cols].corr()
plt.figure(figsize=(7,6)); plt.imshow(cor,aspect="auto")
plt.xticks(range(len(cor.columns)),cor.columns,rotation=45,ha="right")
plt.yticks(range(len(cor.index)),cor.index); plt.colorbar(label="Correlation")
plt.title("Correlation Matrix of Key Logistics Variables")
plt.tight_layout(); plt.savefig("visualizations/06_correlation_matrix.png",dpi=180); plt.close()
print("All visualizations generated.")
