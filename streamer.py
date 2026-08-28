import time
import random
import json

def generate_sensor_data(cycle_count):
    # As cycle count increases, battery health drops
    degradation_factor = min(1.0, cycle_count / 1000.0) 
    
    voltage = round(random.uniform(3.6, 4.2) - (0.5 * degradation_factor), 2)
    current = round(random.uniform(1.0, 2.5), 2)
    temperature = round(random.uniform(25.0, 35.0) + (10.0 * degradation_factor), 1)
    internal_resistance = round(random.uniform(15.0, 25.0) + (50.0 * degradation_factor), 1)
    
    # Target Ground Truth SoH (100% down to 70%)
    actual_soh = round(max(70.0, 100.0 - (30.0 * degradation_factor)), 2)
    
    return {
        "voltage_V": voltage,
        "current_A": current,
        "temperature_C": temperature,
        "internal_resistance_mOhm": internal_resistance,
        "cycle_count": cycle_count,
        "actual_soh": actual_soh
    }

# Simulating 500 cycles of data generation
print("Generating simulated battery dataset...")
dataset = [generate_sensor_data(c) for c in range(1, 501)]

with open("simulated_battery_data.json", "w") as f:
    json.dump(dataset, f, indent=4)
print("Data simulation complete! Saved to 'simulated_battery_data.json'")
