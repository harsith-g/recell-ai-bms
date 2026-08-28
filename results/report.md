# ReCell AI - SoH Model Results

## Cross-battery holdout (primary, honest metric)
Setup: train=B0005+B0006+B0007, test=B0018 (unseen battery)

- R²: 0.998
- RMSE: 0.3465 (SoH %)
- MAE: 0.272 (SoH %)
- Test samples: 132

## Mid-cycle-realistic feature set (honest stress test)
Setup: train=B0005+B0006+B0007, test=B0018, discharge_duration_s excluded (mid-cycle-realistic feature set)

- R²: 0.5693
- RMSE: 5.0631 (SoH %)
- MAE: 4.3948 (SoH %)

## Pooled random split (optimistic, for comparison)
Setup: 80/20 random split, all 4 batteries pooled (optimistic)

- R²: 0.999
- RMSE: 0.299 (SoH %)
- MAE: 0.2373 (SoH %)

## Feature importance (cross-battery model)
- discharge_duration_s: 0.987
- mean_voltage: 0.007
- cycle: 0.004
- voltage_drop_rate: 0.001
- max_voltage: 0.000
- mean_current: 0.000
- max_temperature: 0.000
- min_current: 0.000
- min_voltage: 0.000
- voltage_range: 0.000
- mean_temperature: 0.000
