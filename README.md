# TSLA Sentiment & Market Prediction

A Machine Learning Approach to Financial News Sentiment and Market Trend Prediction (TSLA, Jan 2020 – Dec 2023).
Project for the *Projet Multidisciplinaire* course.

End-to-end pipeline: raw financial text (news, Twitter/Musk stream, Reddit, StockTwits) → NLP preprocessing → FinBERT-based sentiment scoring → feature engineering → LSTM+Attention / Random Forest / XGBoost market prediction, deployed as a live Streamlit dashboard.

**Author:** Khelil Dhiaeddine — 4th-year Data Science Engineering, University of Blida 1
**Supervisor:** Nesrine Lahiani

**Live demo:** [TSLA sentiment & market prediction Streamlit dashboard](https://pfe-tsla-sentiment-khpifztqkqztwuosxsmtc4.streamlit.app)

---

## Project Highlights

- **85,649 documents** collected across 4 sources (financial news, Twitter/Musk stream, Reddit, StockTwits)
- **FinBERT** sentiment scoring vs. 3 classical baselines (TF-IDF+LR, TF-IDF+SVM, VADER), with domain-shift quantified
- **995 × 237 feature matrix**: same-day, lag, rolling, momentum, and EMA sentiment/price features
- **LSTM + Bahdanau Attention**, Random Forest, and XGBoost predictors evaluated with walk-forward cross-validation and Sharpe Ratio, not just accuracy
- **Best result:** XGBoost (price-only) — Sharpe = 5.71, Directional Accuracy = 0.530 on the 2023 held-out test year
- Deployed as an interactive **Streamlit dashboard** for exploring sentiment, features, and predictions

---

## Repository Structure

```
tsla-sentiment-market-prediction/
├── app/                              # Streamlit dashboard (deployed app)
│   ├── app.py                        # Dashboard entry point
│   ├── features_targets_final_clean.parquet  # Final feature/target matrix used by the app
│   └── requirements.txt              # App-specific dependencies (Streamlit Cloud)
│
├── docs/                             # Written deliverables
│   └── financial_sentimnet.pdf       # Project report
│
├── notebooks/                        # End-to-end pipeline, one notebook per stage
│   ├── 01_data_collection.ipynb
│   ├── 02_nlp_preprocessing.ipynb
│   ├── 03_sentiment_analysis.ipynb
│   ├── 04_feature_engineering.ipynb
│   └── 05_market_prediction.ipynb
│
├── outputs/                           # Versioned artifacts produced by each notebook
│   ├── step1_data_collection/         # Raw + cleaned corpus (4 sources, 85,649 docs)
│   ├── step2_nlp_preprocessing/       # Two-track preprocessing outputs (raw text + lemmatized tokens)
│   ├── step3_sentiment_analysis/      # FinBERT + baseline sentiment scores, evaluation metrics
│   ├── step4_feature_engineering/     # daily_sentiment.parquet, features_targets.parquet
│   └── step5_prediction/              # Trained model outputs, predictions, evaluation results
│
├── .gitignore
├── README.md                          # You are here
└── requirements.txt                   # Root/pipeline dependencies (Kaggle notebooks)
```

---

## Pipeline Overview

| Step | Notebook | Description | Key Output |
|------|----------|-------------|------------|
| 1 | `01_data_collection.ipynb` | Collects financial news, Twitter/Musk stream, Reddit, and StockTwits data for TSLA | `outputs/step1_data_collection/` — 85,649 documents |
| 2 | `02_nlp_preprocessing.ipynb` | Two-track NLP cleaning: raw text preserved for FinBERT, lemmatized tokens for classical baselines | `outputs/step2_nlp_preprocessing/` |
| 3 | `03_sentiment_analysis.ipynb` | FinBERT sentiment scoring + TF-IDF/VADER baselines, domain-shift analysis | `outputs/step3_sentiment_analysis/` |
| 4 | `04_feature_engineering.ipynb` | Builds daily sentiment indices and joins with TSLA price data (NYSE 16:00 EST cutoff) | `outputs/step4_feature_engineering/` — 995×237 feature matrix |
| 5 | `05_market_prediction.ipynb` | Trains LSTM+Attention, Random Forest, XGBoost; walk-forward CV; Sharpe Ratio evaluation | `outputs/step5_prediction/` |

The `app/` folder consumes the final cleaned feature/target matrix (`features_targets_final_clean.parquet`) produced at the end of Step 4/5 to power the Streamlit dashboard, independent of the Kaggle notebook environment.

---

## Running the Dashboard Locally

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

## Running the Pipeline

The notebooks in `notebooks/` are Kaggle-compatible (hardcoded `/kaggle/input/` and `/kaggle/working/` paths) and are intended to be run in order, 01 → 05, on a GPU-enabled Kaggle kernel (developed on a T4 GPU). Each notebook writes its artifacts to the corresponding `outputs/stepN_*/` folder for the next step to consume.

```bash
pip install -r requirements.txt
```

---

## Documentation

Full methodology, literature review, system design, implementation details, and results are documented in the project report: [`docs/financial_sentimnet.pdf`](docs/financial_sentimnet.pdf).

---

## Known Limitations

- A small `doc_id` collision (1 out of ~85,650 rows, ~0.002%) occurs for tweets missing a `tweet_id`, which collapse to a shared placeholder URL upstream. Current handling: deduplicate on `doc_id`, keeping the first occurrence (see Step 3 preprocessing notebook).
- Sentiment features add noise rather than signal during the 2023 test period, likely due to a market regime shift and a gap in Twitter/Musk-stream coverage; price-only models outperform full-feature (sentiment + price) models in this window.

---

## Citation

If you use this work, please cite the accompanying report (Khelil Dhiaeddine, University of Blida 1, supervised by Nesrine Lahiani).
