# Project Methodology

## Title
A Machine Learning Approach to Financial News Sentiment and Market Trend Prediction

## Focus Asset
Tesla (TSLA) — January 2020 to December 2023

## Core Novelties

### Novelty 1 — FinBERT for Sentiment Classification
FinBERT replaces TF-IDF as the primary sentiment classifier.
Sentiment momentum and lag features are engineered from raw scores.

### Novelty 2 — LSTM with Bahdanau Attention
LSTM with Bahdanau Attention mechanism generates trading signals.
Evaluated via backtesting using Sharpe Ratio and Directional Accuracy.

## Pipeline Stages

1. **Data Collection** — Multi-source text data (news, Reddit, Twitter) + OHLCV stock prices
2. **NLP Preprocessing** — Source-specific cleaning, tokenization, FinBERT-compatible input preparation
3. **Sentiment Analysis** — FinBERT inference, classical baselines (LR, SVM), deep learning (LSTM, GRU)
4. **Feature Engineering** — Daily sentiment aggregation, momentum indicators, lag features
5. **Market Prediction** — Random Forest, XGBoost, LSTM, Temporal Transformers → BUY/SELL/HOLD signals

## Evaluation Metrics
- Accuracy, F1-Score (sentiment classification)
- Directional Accuracy (market prediction)
- Sharpe Ratio (backtesting)
- Confusion Matrix (trading signals)
