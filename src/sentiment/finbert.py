"""
finbert.py
----------
FinBERT inference utilities for sentiment classification.

Model: ProsusAI/finbert
Labels: positive=0, negative=1, neutral=2
Sentiment score: p_pos - p_neg  (range: -1 to +1)

Usage:
    from src.sentiment.finbert import load_finbert, run_finbert_batch

    tokenizer, model = load_finbert(device='cpu')
    results = run_finbert_batch(texts, tokenizer, model, device='cpu')
"""

import numpy as np
import torch
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

FINBERT_MODEL = "ProsusAI/finbert"
LABEL_MAP = {0: "positive", 1: "negative", 2: "neutral"}


def load_finbert(device: str = None):
    """
    Load the FinBERT tokenizer and model.

    Parameters
    ----------
    device : 'cuda', 'cpu', or None (auto-detects GPU if available)

    Returns
    -------
    (tokenizer, model) tuple — model is in eval mode on the target device.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    model.to(device)
    model.eval()

    print(f"FinBERT loaded on {device}")
    print(f"Label map: {model.config.id2label}")
    return tokenizer, model


def run_finbert_batch(texts, tokenizer, model, device: str, batch_size: int = 16):
    """
    Run FinBERT inference on a list of texts using first-chunk truncation.

    Truncation strategy: texts exceeding 512 tokens are truncated to the first
    512 tokens. For news, financial signal concentrates in the headline and
    opening paragraph — validated in 02_eda_finbert_pilot.ipynb.

    Parameters
    ----------
    texts      : list of str — cleaned text strings
    tokenizer  : HuggingFace tokenizer (from load_finbert)
    model      : HuggingFace model (from load_finbert)
    device     : 'cuda' or 'cpu'
    batch_size : number of texts per forward pass (default 16)

    Returns
    -------
    List of dicts with keys:
        p_pos, p_neg, p_neu        — class probabilities (float)
        sentiment_label            — 'positive' | 'negative' | 'neutral'
        sentiment_score            — p_pos - p_neg (float, range -1 to +1)
    """
    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**encoded).logits

        probs = softmax(logits, dim=1).cpu().numpy()

        for p in probs:
            label_idx = int(np.argmax(p))
            results.append(
                {
                    "p_pos": round(float(p[0]), 4),
                    "p_neg": round(float(p[1]), 4),
                    "p_neu": round(float(p[2]), 4),
                    "sentiment_label": LABEL_MAP[label_idx],
                    "sentiment_score": round(float(p[0]) - float(p[1]), 4),
                }
            )

    return results
