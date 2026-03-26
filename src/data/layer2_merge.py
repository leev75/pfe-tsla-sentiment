"""
layer2_merge.py
---------------
Reusable module for normalizing all Layer 1 sources
into the unified Layer 2 Parquet schema.

Layer 2 Schema:
    doc_id        | string       | Unique document identifier
    published_at  | datetime UTC | Normalized timestamp
    source        | string       | Source tag
    text          | string       | Full text content
    ticker        | string       | Stock ticker (e.g. TSLA)
    url           | string       | Source URL (nullable)
"""

import hashlib
import pandas as pd


TICKER = "TSLA"
DATE_START = "2020-01-01"
DATE_END   = "2023-12-31"


# ── Utilities ─────────────────────────────────────────────────────────────────

def generate_doc_id(row, fields=("published_at", "text")):
    """Deterministic 16-char MD5 doc_id from timestamp + text prefix."""
    raw = "_".join(str(row.get(f, ""))[:50] for f in fields)
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def extract_subreddit_from_url(url):
    """Extract subreddit from Reddit URL."""
    try:
        parts = str(url).split("/")
        r_idx = parts.index("r")
        return parts[r_idx + 1]
    except (ValueError, IndexError):
        return None


def build_combined_text(*parts, sep=". "):
    """Join non-empty text parts with separator."""
    filled = [str(p).strip() for p in parts if pd.notna(p) and str(p).strip()]
    return sep.join(filled) if filled else None


# ── Per-source normalizers ─────────────────────────────────────────────────────

def normalize_reddit_s1(df):
    df = df.copy()
    df["text"] = df.apply(
        lambda r: build_combined_text(r.get("title"), r.get("selftext")), axis=1
    )
    df = df[df["text"].notna() & (df["text"].str.strip() != "")]
    return _to_layer2(df, doc_id_col="id", ts_col="created_utc",
                      source="reddit", url_col="url")


def normalize_reddit_s2(df):
    df = df.copy()
    df = df.rename(columns={"body": "selftext", "comms_num": "num_comments"})

    # Resolve timestamp
    if "created_utc" in df.columns:
        df["_ts"] = df["created_utc"]
    elif "created" in df.columns:
        df["_ts"] = pd.to_datetime(df["created"], unit="s", utc=True)

    # Extract subreddit from URL if missing
    if "subreddit" not in df.columns:
        df["subreddit"] = df["url"].apply(extract_subreddit_from_url)

    df["text"] = df.apply(
        lambda r: build_combined_text(r.get("title"), r.get("selftext")), axis=1
    )
    df = df[df["text"].notna() & (df["text"].str.strip() != "")]
    return _to_layer2(df, ts_col="_ts", source="reddit", url_col="url")


def normalize_news(df):
    df = df.copy()
    # Drop rows where both headline and content are null
    df = df[df["headline"].notna() | df["content"].notna()]
    df["text"] = df.apply(
        lambda r: build_combined_text(r.get("headline"), r.get("content")), axis=1
    )
    return _to_layer2(df, doc_id_col="article_id", ts_col="published_at",
                      source="news", url_col="url")


def normalize_twitter(df):
    df = df.copy()
    # Generate doc_id from tweet_id if available
    if "tweet_id" in df.columns:
        df["_doc_id"] = df["tweet_id"].astype(str)
    else:
        df["_doc_id"] = df.apply(generate_doc_id, axis=1)

    ts_col = "created_at" if "created_at" in df.columns else "published_at"
    df = df[df["text"].notna() & (df["text"].str.strip() != "")]
    return _to_layer2(df, doc_id_col="_doc_id", ts_col=ts_col, source="twitter")


# ── Core normalizer ───────────────────────────────────────────────────────────

def _to_layer2(df, doc_id_col=None, ts_col="published_at",
               source=None, url_col=None):
    out = pd.DataFrame()
    out["doc_id"]       = df[doc_id_col].astype(str) if doc_id_col else df.apply(generate_doc_id, axis=1)
    out["published_at"] = df[ts_col]
    out["source"]       = source if source else df["source"]
    out["text"]         = df["text"]
    out["ticker"]       = TICKER
    out["url"]          = df[url_col] if url_col and url_col in df.columns else None
    return out


# ── Final merge ───────────────────────────────────────────────────────────────

def build_layer2(frames):
    """
    Concatenate, clean, and return the final Layer 2 DataFrame.

    Parameters
    ----------
    frames : list of pd.DataFrame
        Pre-normalized Layer 2 frames from each source normalizer.
    """
    df = pd.concat(frames, ignore_index=True)

    # Normalize timestamp to UTC
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")

    # Drop null text
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip() != ""]

    # Deduplicate
    df = df.drop_duplicates(subset=["doc_id"])

    # Filter to target date range
    df = df[
        (df["published_at"] >= DATE_START) &
        (df["published_at"] <= DATE_END)
    ]

    df = df.reset_index(drop=True)
    return df
