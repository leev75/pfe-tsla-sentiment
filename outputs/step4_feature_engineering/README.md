# Step 4 — Feature Engineering

## Purpose
Convert per-document sentiment scores and raw price data into a daily feature matrix suitable for market prediction models, and define BUY/SELL/HOLD prediction targets.

## Dataset (hosted on Kaggle)

| Content | Dataset | Link |
|---|---|---|
| Final feature/target matrix | `features_targets_final_clean` | [Kaggle](https://www.kaggle.com/datasets/leev75/test999) |

> **Naming note:** consider renaming this dataset on Kaggle to something like `tsla-features-targets` for clarity before final submission/publication — `test999` won't read well in a thesis or paper appendix.

## Feature matrix
**Shape:** 995 rows (trading days) × 237 columns

| Feature group | Count |
|---|---|
| Same-day sentiment/price features | 35 |
| Lag features | 75 |
| Rolling window features | 90 |
| Momentum features | 40 |
| EMA (exponential moving average) features | 10 |
| Meta / target columns | 7 |

## Target definition
BUY/SELL/HOLD labels defined at a ±1% price-move threshold. Class distribution:
- **Train:** BUY 308 / SELL 306 / HOLD 132
- **Test:** BUY 110 / SELL 87 / HOLD 52

## Intermediate artifact
`daily_sentiment.parquet` (5,030 rows = 5 sources × 1,006 days) is produced as an intermediate step before the final feature matrix — see the daily aggregation notebook below.

## Notebooks
- `notebooks/08-daily-aggregation.ipynb` — aggregates per-document sentiment into daily scores per source
- `notebooks/09-features-targets-split.ipynb` — builds the full feature matrix and train/test split → produces `test999`

## Next step
Feeds into `step5_prediction/`.
