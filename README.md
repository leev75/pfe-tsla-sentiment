# A Machine Learning Approach to Financial News Sentiment and Market Trend Prediction

End-to-end ML pipeline that combines financial sentiment analysis with market prediction, applied to Tesla (TSLA) stock, January 2020 – December 2023.

**Author:** Khelil Dhiaeddine — 4th-year Data Science Engineering, University of Blida 1
**Supervisor:** Nesrine Lahiani
**Teacher:** Hadjer Ykhlef

🔗 **Live dashboard:** [Streamlit App](https://pfe-tsla-sentiment-khpifztqkqztwuosxsmtc4.streamlit.app)

---

## Overview

This project builds a full pipeline — from raw text collection to trading-signal generation — to test whether financial sentiment (news, Reddit, Twitter) adds predictive value on top of price-based features for TSLA stock, using rigorous trading-focused evaluation (Sharpe ratio, walk-forward CV) instead of accuracy alone.

### Key novelties

1. **FinBERT vs. TF-IDF, with domain-shift quantified** — FinBERT reaches macro-F1 = 0.403 on a 200-doc manual TSLA test set (vs. TF-IDF+LR = 0.380, TF-IDF+SVM = 0.364, VADER = 0.355), and we explicitly measure the in-domain → out-of-domain performance drop for classical models (LR: −0.427pp, SVM: −0.443pp).
2. **LSTM + Bahdanau Attention evaluated with trading metrics** — Sharpe ratio and walk-forward cross-validation, not just classification accuracy.
3. **Four-source corpus** (news, Reddit, general Twitter, Musk-specific Twitter) — 85,649 documents total, with the Musk stream architecturally separated as an independent signal.

### Research gaps addressed

| Gap | Description |
|---|---|
| Gap 1 | Prior work relies on single-source datasets |
| Gap 2 | Prior evaluations lack trading-relevant metrics |
| Gap 3 | Domain-shift cost of sentiment models is rarely quantified |

---

## Pipeline

```
1. Data Collection      → news, Reddit, Twitter (general + Musk), stock prices
2. NLP Preprocessing    → cleaning, tokenization, lemmatization, NER
3. Sentiment Analysis   → FinBERT / TF-IDF+LR / TF-IDF+SVM / VADER
4. Feature Engineering  → 237 features (same-day, lag, rolling, momentum, EMA)
5. Market Prediction    → XGBoost, Random Forest, LSTM, LSTM+Attention
```

### Repo structure

```
├── app/                     # Streamlit dashboard source
├── docs/                    # Thesis (LaTeX + PDF)
├── notebooks/               # Kaggle notebooks, one per pipeline stage (01–11)
├── outputs/                 # Per-step figures + README with dataset links (raw data lives on Kaggle)
│   ├── step1_data_collection/
│   ├── step2_nlp_preprocessing/
│   ├── step3_sentiment_analysis/
│   ├── step4_feature_engineering/
│   └── step5_prediction/
├── src/                      # Reusable modules (data, nlp, sentiment, prediction)
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

> **Note on data:** raw and intermediate datasets are hosted on Kaggle rather than committed to this repo (large parquet files). See the **Datasets** section below for links, or each step's `outputs/stepX/README.md` for the ones specific to that stage.

---

## Results summary

**Sentiment classification** (200-doc manual TSLA test set):

| Model | Macro-F1 |
|---|---|
| FinBERT | 0.403 [0.33, 0.47] |
| TF-IDF + LR | 0.380 |
| TF-IDF + SVM | 0.364 |
| VADER | 0.355 |

**Market prediction** (2023 test set, Sharpe ratio):

| Model | Sharpe | Directional Accuracy | F1 |
|---|---|---|---|
| **XGBoost (price-only)** | **5.71** | 0.530 | 0.400 |
| Random Forest (price-only) | 4.65 | — | — |
| LSTM + Attention (price-only) | 3.65 | — | — |
| LSTM (price-only) | 3.49 | — | — |
| Buy-and-hold | 2.47 | — | — |

Full-feature models (sentiment + price) underperform price-only models across the board in 2023 — attributed to a market regime shift and a Twitter coverage gap that let sentiment features add noise rather than signal. The attention-based full-feature model collapses to zero BUY predictions, a curse-of-dimensionality effect (237 features vs. ~750 training rows).

Full methodology, dataset statistics, and discussion are in [`docs/thesis.pdf`](docs/thesis.pdf).

---

## Datasets

All raw and intermediate datasets are hosted on Kaggle. Full context for each is in the corresponding `outputs/stepX/README.md`.

| Step | Content | Dataset | Link |
|---|---|---|---|
| 1 | Reddit (part 1) | `reddit-s1-parquet` | [Kaggle](https://www.kaggle.com/datasets/leev75/reddit-s1-parquet) |
| 1 | Reddit (part 2, 2022) | `reddit-s2-2022-parquet` | [Kaggle](https://www.kaggle.com/datasets/leev75/reddit-s2-2022-parquet) |
| 1 | Reddit (part 3) | `reddit-s3-parquet` | [Kaggle](https://www.kaggle.com/datasets/leev75/reddit-s3-parquet) |
| 1 | Twitter (general) | `layer1` | [Kaggle](https://www.kaggle.com/datasets/leev75/layer1) |
| 1 | News | `layer1-newsupdated` | [Kaggle](https://www.kaggle.com/datasets/leev75/layer1-newsupdated) |
| 2 | Cleaned layer | `layer2-cleaned` | [Kaggle](https://www.kaggle.com/datasets/leev75/layer2-cleaned) |
| 2 | Preprocessed layer | `processd` | [Kaggle](https://www.kaggle.com/datasets/leev75/processd) |
| 3 | Baseline + FinBERT sentiment | `layer2-multimodel` | [Kaggle](https://www.kaggle.com/datasets/leev75/layer2-multimodel) |
| 4 | Final feature/target matrix | `test999` | [Kaggle](https://www.kaggle.com/datasets/leev75/test999) |
| 5 | Prediction outputs | — | not yet published |

**Known gaps:**
- The Musk Twitter stream (Step 1) doesn't have a public Kaggle dataset yet, despite being architecturally separated as a core novelty of this project.
- Step 5 has no standalone dataset — models train directly from the Step 4 `test999` matrix.

---

## Dashboard

A 4-page Streamlit app for exploring the results interactively:

- **Overview** — corpus and dataset summary
- **Sentiment Analysis** — model comparison on the manual test set
- **Model Comparison** — prediction model performance (Sharpe, DA, F1)
- **Trading Signals** — BUY/SELL/HOLD signals over the test period

Reads from `features_targets_final_clean.parquet`.

Run locally:
```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

---

## Setup

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

Notebooks were developed on Kaggle (T4 GPU) with hardcoded `/kaggle/input/` and `/kaggle/working/` paths — adjust paths if running elsewhere.

**Core dependencies:** Python 3.12, PyTorch 2.10 (cu128), Transformers 5.0, scikit-learn 1.6, XGBoost 3.2, spaCy 3.8, pandas 2.3, numpy 2.0.

---

## Known issues

- **`doc_id` collision (0.002%, 1 out of ~85,650 rows):** tweets with missing `tweet_id` share a placeholder URL, collapsing the upstream hash. Currently handled by deduplicating on `doc_id`. Planned fix: switch to a content-based hash (`md5(text + published_at)`).

---

## Future work

- Patch the `doc_id` hashing to content-based keys
- Extend beyond TSLA to other high-volatility tickers
- Re-evaluate sentiment contribution outside regime-shift periods
- Convert this work into a conference/journal paper (ACM ICAIF, IEEE ICMLA, FinNLP, *Expert Systems with Applications*) and submit an arXiv preprint

---

## Citation

If you use this work, please cite the accompanying thesis (citation details to be added upon publication).

## License

Specify a license here (e.g. MIT) if you intend the repo to be reused.
