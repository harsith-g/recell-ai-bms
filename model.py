import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 1. Load simulated data
print("Loading dataset...")
with open("simulated_battery_data.json") as f:
    data = json.load(f)
df = pd.DataFrame(data)

# 2. Features and Target split
X = df[["voltage_V", "current_A", "temperature_C", "internal_resistance_mOhm", "cycle_count"]]
y = df["actual_soh"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Model
print("Training Random Forest Regressor model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Model Training Complete! Mean Squared Error: {mse:.4f}\n")

# 5. Simulate Live Edge Inference
print("--- Simulating Live Sensor Input from Edge Node ---")
# Example input: Low voltage, high temp, high internal resistance at cycle 450
live_sensor_reading = [[3.5, 1.8, 42.1, 55.4, 450]] 
predicted_soh = model.predict(live_sensor_reading)

print(f"Live Sensor Telemetry: {live_sensor_reading[0]}")
print(f"Predicted Battery State of Health (SoH): {predicted_soh[0]:.2f}%")
