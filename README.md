# ReCell AI — Battery State-of-Health Estimation

A machine learning pipeline that estimates the State of Health (SoH) of
lithium-ion 18650 cells from charge/discharge telemetry, using the
NASA Ames Prognostics Center of Excellence (PCoE) Li-ion Battery Aging
dataset (batteries B0005, B0006, B0007, B0018).

Built as part of my prep for the Mitacs Globalink Research Internship
application, in the battery/EV prognostics space. AI-assisted
development (Claude) — I directed the pipeline design and evaluation
methodology; the code was written with AI tool assistance, which I
reviewed and understood before including it here.

## What this actually does

1. **`src/build_features.py`** — takes raw per-timestep discharge
   telemetry (voltage, current, temperature) from the NASA dataset and
   aggregates it into per-cycle features (636 cycles across 4 batteries).
2. **`src/train.py`** — trains a `RandomForestRegressor` to predict SoH
   (%) from those features, and evaluates it three different ways so
   the result can't be cherry-picked:

| Evaluation setup | R² | RMSE (SoH %) | What it means |
|---|---|---|---|
| **Cross-battery holdout** (train on B0005/6/7, test on unseen B0018) | 0.998 | 0.35 | Realistic generalization to a battery the model never saw |
| **Mid-cycle-realistic features** (same holdout, `discharge_duration_s` removed) | 0.569 | 5.06 | The honest, harder number — see caveat below |
| Pooled 80/20 random split (all batteries mixed) | 0.999 | 0.30 | Optimistic upper bound; nearby cycles leak across train/test |

See `results/report.md` and `results/soh_results.png` for the full
breakdown and plots.

## An honest caveat (the part most people skip)

The headline 0.998 R² is *mostly* driven by one feature:
`discharge_duration_s` — how long the cell took to hit the voltage
cutoff. That's almost a direct restatement of capacity (duration ≈
capacity ÷ current), so a model leaning on it is closer to recovering
capacity from a near-proxy than doing genuine prognostics.

Drop that one feature and R² falls to **0.57** — that's the real
difficulty of estimating SoH from features you'd actually have
*before* a discharge cycle finishes, which is the realistic constraint
for a live battery management system doing continuous monitoring
rather than a post-hoc health check after a full discharge test.

I'm reporting both numbers on purpose. A single suspiciously clean
metric is a red flag in prognostics research; showing where the easy
number comes from — and how hard the problem actually is once you
remove it — is the more honest and more useful result.

## Why this, why now

Targeting the Mitacs Globalink Research Internship 2027 (AICTE-Mitacs
partnership quota) in EV / intelligent transportation / predictive
maintenance, as a step toward a self-funded M.Sc. in Germany and a
long-term goal of working on embedded ML for industrial/automotive
hardware.

## Data source

NASA Ames Prognostics Center of Excellence, Li-ion Battery Aging
Dataset. Rated capacity 2.0 Ah, end-of-life defined at 1.4 Ah (70%
SoH) per NASA's dataset documentation. CC0 licensed.
https://data.nasa.gov/dataset/li-ion-battery-aging-datasets

## Running it

```bash
pip install -r requirements.txt
python src/build_features.py
python src/train.py
```

## Next steps

- Try a sliding-window mid-cycle feature set (voltage/current samples
  from just the first N seconds of discharge) to close the gap on the
  harder 0.57 number without needing a full discharge to complete.
- Test on a Raspberry Pi to get real inference-latency numbers before
  claiming "edge-deployable" anywhere.
