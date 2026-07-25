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
├── notebooks/              # Kaggle notebooks, one per pipeline step
├── figures/                # Generated plots (confusion matrices, F1, kappa, attention, etc.)
├── docs/                   # Thesis (LaTeX + PDF)
├── dashboard/              # Streamlit app source
├── requirements.txt
└── README.md
```

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
