"""
train.py
--------
Trains a RandomForestRegressor to estimate battery State-of-Health (SoH)
from cycle-level telemetry features, and evaluates it two ways:

1. CROSS-BATTERY HOLDOUT (the honest, hard test):
   Train on B0005 + B0006 + B0007, test on B0018 (a battery the model
   has never seen). This is the realistic deployment scenario -- a BMS
   sees a battery unit it wasn't trained on.

2. POOLED RANDOM SPLIT (the easy, optimistic test):
   80/20 random split across all four batteries pooled together.
   Reported for comparison only -- this leaks information because
   nearby cycles from the same battery end up in both train and test.

Both numbers get written to results/metrics.json and results/report.md
so there's no cherry-picking after the fact.
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

FEATURES = [
    "cycle",
    "mean_voltage",
    "min_voltage",
    "max_voltage",
    "voltage_range",
    "mean_current",
    "min_current",
    "mean_temperature",
    "max_temperature",
    "discharge_duration_s",
    "voltage_drop_rate",
]
TARGET = "soh_pct"


def evaluate(y_true, y_pred):
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "mse": round(float(mean_squared_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "n_test": int(len(y_true)),
    }


def main():
    df = pd.read_csv("data/cycle_features.csv")

    # --- Scenario 1: cross-battery holdout (honest, hard) ---
    holdout_battery = "B0018"
    train_df = df[df["battery"] != holdout_battery]
    test_df = df[df["battery"] == holdout_battery]

    model_cross = RandomForestRegressor(
        n_estimators=300, max_depth=8, random_state=42, n_jobs=-1
    )
    model_cross.fit(train_df[FEATURES], train_df[TARGET])
    pred_cross = model_cross.predict(test_df[FEATURES])
    metrics_cross = evaluate(test_df[TARGET], pred_cross)
    metrics_cross["setup"] = f"train=B0005+B0006+B0007, test={holdout_battery} (unseen battery)"

    # --- Scenario 2: pooled random split (optimistic, for comparison) ---
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=42
    )
    model_pooled = RandomForestRegressor(
        n_estimators=300, max_depth=8, random_state=42, n_jobs=-1
    )
    model_pooled.fit(X_train, y_train)
    pred_pooled = model_pooled.predict(X_test)
    metrics_pooled = evaluate(y_test, pred_pooled)
    metrics_pooled["setup"] = "80/20 random split, all 4 batteries pooled (optimistic)"

    # --- Scenario 3: real-time-friendly feature set (honest stress test) ---
    # discharge_duration_s is almost a direct restatement of capacity
    # (duration to voltage cutoff ~ capacity / current), so a model that
    # leans on it is really just recovering capacity from a near-proxy,
    # not doing genuine prognostics. Drop it to see how the model does
    # on features you'd actually have mid-cycle, before a discharge
    # finishes -- the realistic constraint for a live BMS.
    rt_features = [f for f in FEATURES if f != "discharge_duration_s"]
    train_rt = df[df["battery"] != holdout_battery]
    test_rt = df[df["battery"] == holdout_battery]
    model_rt = RandomForestRegressor(
        n_estimators=300, max_depth=8, random_state=42, n_jobs=-1
    )
    model_rt.fit(train_rt[rt_features], train_rt[TARGET])
    pred_rt = model_rt.predict(test_rt[rt_features])
    metrics_rt = evaluate(test_rt[TARGET], pred_rt)
    metrics_rt["setup"] = (
        f"train=B0005+B0006+B0007, test={holdout_battery}, "
        "discharge_duration_s excluded (mid-cycle-realistic feature set)"
    )

    results = {
        "cross_battery_holdout": metrics_cross,
        "cross_battery_no_duration_feature": metrics_rt,
        "pooled_random_split": metrics_pooled,
        "rated_capacity_ah": 2.0,
        "batteries_used": sorted(df["battery"].unique().tolist()),
        "n_cycles_total": int(len(df)),
    }

    with open("results/metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Feature importance
    importance = sorted(
        zip(FEATURES, model_cross.feature_importances_), key=lambda x: -x[1]
    )

    joblib.dump(model_cross, "models/soh_model_cross_battery.joblib")
    joblib.dump(model_pooled, "models/soh_model_pooled.joblib")

    with open("results/report.md", "w") as f:
        f.write("# ReCell AI - SoH Model Results\n\n")
        f.write("## Cross-battery holdout (primary, honest metric)\n")
        f.write(f"Setup: {metrics_cross['setup']}\n\n")
        f.write(f"- R²: {metrics_cross['r2']}\n")
        f.write(f"- RMSE: {metrics_cross['rmse']} (SoH %)\n")
        f.write(f"- MAE: {metrics_cross['mae']} (SoH %)\n")
        f.write(f"- Test samples: {metrics_cross['n_test']}\n\n")
        f.write("## Mid-cycle-realistic feature set (honest stress test)\n")
        f.write(f"Setup: {metrics_rt['setup']}\n\n")
        f.write(f"- R²: {metrics_rt['r2']}\n")
        f.write(f"- RMSE: {metrics_rt['rmse']} (SoH %)\n")
        f.write(f"- MAE: {metrics_rt['mae']} (SoH %)\n\n")
        f.write("## Pooled random split (optimistic, for comparison)\n")
        f.write(f"Setup: {metrics_pooled['setup']}\n\n")
        f.write(f"- R²: {metrics_pooled['r2']}\n")
        f.write(f"- RMSE: {metrics_pooled['rmse']} (SoH %)\n")
        f.write(f"- MAE: {metrics_pooled['mae']} (SoH %)\n\n")
        f.write("## Feature importance (cross-battery model)\n")
        for feat, imp in importance:
            f.write(f"- {feat}: {imp:.3f}\n")

    print(json.dumps(results, indent=2))
    print("\nFeature importance:")
    for feat, imp in importance:
        print(f"  {feat}: {imp:.3f}")


if __name__ == "__main__":
    main()
