# Step 3 — Sentiment Analysis

## Purpose
Score every document in the cleaned corpus for sentiment, and benchmark four approaches (FinBERT, TF-IDF+LR, TF-IDF+SVM, VADER) on a manually labeled TSLA test set — including a domain-shift analysis using the Financial PhraseBank.

## Dataset (hosted on Kaggle)

| Content | Dataset | Link |
|---|---|---|
| Baseline inference + FinBERT sentiment outputs | `layer2-multimodel` | [Kaggle](https://www.kaggle.com/datasets/leev75/layer2-multimodel) |

## Models compared
| Model | Macro-F1 (200-doc TSLA manual test set) |
|---|---|
| **FinBERT** | **0.403** [0.33, 0.47] |
| TF-IDF + Logistic Regression | 0.380 |
| TF-IDF + SVM | 0.364 |
| VADER | 0.355 |

## Domain-shift analysis
Classical models (LR, SVM) were also evaluated in-domain on the Financial PhraseBank (LR = 0.807, SVM = 0.808 accuracy) to quantify how much performance is lost moving from a clean in-domain benchmark to real-world TSLA text:
- LR domain-shift cost: **−0.427pp**
- SVM domain-shift cost: **−0.443pp**

This motivates using FinBERT (pretrained on financial text) despite its lower absolute F1 — it degrades far less out-of-domain.

## Inter-model agreement
- Cohen's kappa, LR vs SVM: 0.872 (high agreement — both rely on similar surface features)
- Cohen's kappa, LR/SVM vs FinBERT: ≈0.32 (low agreement — FinBERT captures different signal)

## Figures
- `figures/07_confusion_matrices.png` — confusion matrices per model
- `figures/07_per_class_f1.png` — per-class F1 (positive/neutral/negative)
- `figures/07_macro_f1_with_ci.png` — macro-F1 with confidence intervals
- `figures/07_kappa_heatmap.png` — inter-model agreement heatmap

## Notebooks
- `notebooks/05-baseline.ipynb` — TF-IDF + LR / SVM / VADER baselines
- `notebooks/06-finbert-sentiment.ipynb` — FinBERT inference
- `notebooks/07-sentiment-comparison.ipynb` — cross-model comparison, domain-shift analysis, produces the figures above

## Next step
Feeds into `step4_feature_engineering/`.
