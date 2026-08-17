# Step 5 — Market Prediction

## Purpose
Train and evaluate market prediction models on the Step 4 feature matrix, comparing price-only vs. full-feature (price + sentiment) variants, using trading-relevant metrics (Sharpe ratio, directional accuracy) rather than accuracy alone.

## Dataset
No separate Kaggle dataset uploaded yet for this step — models are trained directly from `step4_feature_engineering`'s `test999` dataset. If you export model predictions/artifacts (e.g. saved model weights, prediction CSVs) to Kaggle, add the link here.

## Results (2023 test set)

| Model | Sharpe Ratio | Directional Accuracy | F1 |
|---|---|---|---|
| **XGBoost (price-only)** | **5.71** | 0.530 | 0.400 |
| Random Forest (price-only) | 4.65 | — | — |
| LSTM + Bahdanau Attention (price-only) | 3.65 | — | — |
| LSTM (price-only) | 3.49 | — | — |
| Buy-and-hold baseline | 2.47 | — | — |

TSLA 2023 total return (buy-and-hold): **+113.7%**

## Key finding
**All full-feature (sentiment + price) models underperform their price-only counterparts** in the 2023 test period. This is attributed to a market regime shift and a Twitter data coverage gap in 2023, which let sentiment features add noise rather than signal.

The attention-based full-feature model (`Attn_full`) collapses to predicting 0 BUY signals entirely — a curse-of-dimensionality effect, since it has 237 features but only ~750 training rows.

Despite this, sequential models (LSTM/Attention) show evidence of extracting temporal sentiment signal that tree-based models (XGBoost, RF) cannot — this nuance is discussed alongside the headline Sharpe numbers rather than treated as a contradiction.

## Evaluation methodology
- Walk-forward cross-validation (not a single train/test split) to avoid lookahead bias
- Sharpe ratio computed on the resulting trading signal sequence, not just classification accuracy

## Figures
- `figures/fig_attention.png` — attention weight visualization (LSTM+Attention)
- `figures/fig_confusion_lstm.png` — LSTM confusion matrix
- `figures/fig_volume.png` — trading volume context for the test period

## Notebooks
- `notebooks/10-tree-based-baselines.ipynb` — XGBoost, Random Forest (price-only and full-feature variants)
- `notebooks/11-lstm-attention.ipynb` — LSTM and LSTM+Bahdanau Attention models

## Downstream use
Results feed the **Trading Signals** and **Model Comparison** pages of the Streamlit dashboard.
