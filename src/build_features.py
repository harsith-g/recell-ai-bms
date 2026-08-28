"""
build_features.py
------------------
Turns raw per-timestep NASA PCoE Li-ion discharge telemetry into
per-cycle engineered features for State-of-Health (SoH) regression.

Source data: NASA Ames Prognostics Center of Excellence (PCoE)
Li-ion Battery Aging Dataset (batteries B0005, B0006, B0007, B0018),
18650 cells, rated capacity 2.0 Ah, end-of-life defined at 1.4 Ah
(70% SoH) per NASA's published dataset documentation.
"""

import pandas as pd
import numpy as np

RATED_CAPACITY_AH = 2.0  # NASA PCoE documented rated capacity for these cells

RAW_PATH = "data/nasa_discharge_raw.csv"
OUT_PATH = "data/cycle_features.csv"


def build_cycle_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (battery, cycle), g in df.groupby(["Battery", "id_cycle"]):
        g = g.sort_values("Time")
        duration = g["Time"].max() - g["Time"].min()
        if duration <= 0:
            continue

        voltage = g["Voltage_measured"]
        current = g["Current_measured"]
        temp = g["Temperature_measured"]

        row = {
            "battery": battery,
            "cycle": cycle,
            "mean_voltage": voltage.mean(),
            "min_voltage": voltage.min(),
            "max_voltage": voltage.max(),
            "voltage_range": voltage.max() - voltage.min(),
            "mean_current": current.mean(),
            "min_current": current.min(),
            "mean_temperature": temp.mean(),
            "max_temperature": temp.max(),
            "discharge_duration_s": duration,
            "voltage_drop_rate": (voltage.max() - voltage.min()) / duration,
            "capacity_ah": g["Capacity"].iloc[0],
        }
        rows.append(row)

    feat = pd.DataFrame(rows).sort_values(["battery", "cycle"]).reset_index(drop=True)
    feat["soh_pct"] = (feat["capacity_ah"] / RATED_CAPACITY_AH) * 100.0
    return feat


if __name__ == "__main__":
    raw = pd.read_csv(RAW_PATH)
    raw = raw[raw["type"] == "discharge"].copy()
    features = build_cycle_features(raw)
    features.to_csv(OUT_PATH, index=False)

    print(f"Built {len(features)} cycle-level records from {len(raw)} raw sensor rows")
    print(features.groupby("battery")["soh_pct"].agg(["min", "max", "count"]))
