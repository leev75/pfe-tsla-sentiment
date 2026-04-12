# PFE — A Machine Learning Approach to Financial News Sentiment and Market Trend Prediction

**Student:** Khelil Dhiaeddine  
**Supervisor:** Nesrine Lahiani  
**Teacher:** Hadjer Ykhlef  
**Focus Asset:** Tesla (TSLA) | **Time Window:** January 2020 – December 2023

---

## Project Overview

This project builds a financial sentiment analysis and market prediction system targeting Tesla (TSLA), anchored by the academically defensible **"Musk Effect"** — documented price movements driven by Elon Musk's public statements.

### Core Novelties
- **Novelty 1:** FinBERT replacing TF-IDF for sentiment classification, with sentiment momentum and lag feature engineering
- **Novelty 2:** LSTM with Bahdanau Attention mechanism for trading signal generation with backtesting evaluation (Sharpe Ratio, Directional Accuracy)

---

## Pipeline

```
Data Collection → NLP Preprocessing → Sentiment Analysis → Feature Engineering → Market Prediction
```

| Step | Status | Notebook |
|------|--------|----------|
| 1. Data Collection & Layer 2 Merge | ✅ Done | `01_data_collection_layer2_merge.ipynb` |
| 2. EDA — FinBERT Pilot (200-row sample) | ✅ Done | `02_eda_finbert_pilot.ipynb` |
| 3. NLP Cleaning by Source | ✅ Done | `03_nlp_cleaning_by_source.ipynb` |
| 4. NLP Preprocessing (full pipeline) | 🔜 Next | `04_nlp_preprocessing.ipynb` |
| 5. Sentiment Analysis (FinBERT — 87K rows) | 🔜 Pending | `05_sentiment_analysis.ipynb` |
| 6. Feature Engineering | 🔜 Pending | `06_feature_engineering.ipynb` |
| 7. Market Prediction | 🔜 Pending | `07_market_prediction.ipynb` |

---

## Data Architecture

### Layer 1 — Source-Specific Raw Schemas
Preserves engagement metadata per source for later feature engineering.

| Source | Key Fields |
|--------|-----------|
| `reddit_s1` | title, selftext, score, upvote_ratio, num_comments, subreddit |
| `reddit_s2_2022` | title, text, score, num_comments, created_utc |
| `news` | article_id, headline, content, source_name, author |
| `twitter` | tweet_id, text, retweet_count, like_count, username |

### Layer 2 — Unified NLP-Ready Schema
Six-field Parquet schema feeding the NLP pipeline.

| Field | Type | Description |
|-------|------|-------------|
| `doc_id` | string | Unique document identifier |
| `published_at` | datetime (UTC) | Normalized timestamp |
| `source` | string | Origin source tag |
| `text` | string | Cleaned concatenated text |
| `ticker` | string | Stock ticker (TSLA) |
| `url` | string | Source URL (nullable) |

**Final Layer 2 volume: ~87,196 documents** across ~1,000 trading days

---

## Repository Structure

```
pfe-tsla-sentiment/
├── README.md
├── .gitignore
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── layer1/          # Source-specific raw Parquet files
│   │   └── layer2/          # Unified layer2_unified_final.parquet
│   └── processed/           # Post-NLP outputs (future)
├── notebooks/
│   ├── 01_data_collection_layer2_merge.ipynb
│   ├── 02_eda_finbert_pilot.ipynb
│   ├── 03_nlp_cleaning_by_source.ipynb
│   ├── 04_nlp_preprocessing.ipynb
│   ├── 05_sentiment_analysis.ipynb
│   ├── 06_feature_engineering.ipynb
│   └── 07_market_prediction.ipynb
├── src/
│   ├── data/
│   │   └── layer2_merge.py
│   ├── nlp/
│   ├── sentiment/
│   └── prediction/
├── outputs/
│   └── figures/
└── docs/
    └── methodology.md
```

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/pfe-tsla-sentiment.git
cd pfe-tsla-sentiment
pip install -r requirements.txt
```

Place your raw data files in `data/raw/layer1/` then run notebooks in order.
