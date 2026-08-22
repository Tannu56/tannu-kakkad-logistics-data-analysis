import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("logistics_week4_dataset.csv")
X = df.drop(columns=["shipment_id", "delivery_time_days"])
y = df["delivery_time_days"]

categorical = ["traffic_level", "vehicle_type", "weather_condition"]
numerical = ["distance_km", "shipment_volume", "transportation_cost"]
preprocessor = ColumnTransformer([
    ("num", "passthrough", numerical),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=10, min_samples_leaf=2, random_state=42)
}

results = []
fitted = {}
for name, model in models.items():
    pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": mean_squared_error(y_test, pred) ** 0.5,
        "R2": r2_score(y_test, pred)
    })
    fitted[name] = pipe

results_df = pd.DataFrame(results).sort_values("RMSE")
print(results_df)

best_name = results_df.iloc[0]["Model"]
best_model = fitted[best_name]
cv = KFold(n_splits=5, shuffle=True, random_state=42)
rmse = np.sqrt(-cross_val_score(best_model, X, y, cv=cv, scoring="neg_mean_squared_error"))
mae = -cross_val_score(best_model, X, y, cv=cv, scoring="neg_mean_absolute_error")
print("Best model:", best_name)
print("Mean CV RMSE:", rmse.mean())
print("Mean CV MAE:", mae.mean())

scenario = X_test.copy()
scenario["traffic_level"] = scenario["traffic_level"].replace({"High":"Medium","Medium":"Low"})
print("Scenario average predicted improvement (days):",
      np.mean(best_model.predict(X_test) - best_model.predict(scenario)))
