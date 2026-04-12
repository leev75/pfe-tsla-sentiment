"""
cleaning.py
-----------
Source-specific NLP cleaning functions for the Layer 2 dataset.

Design decisions (derived from 02_eda_finbert_pilot.ipynb diagnostics):
    - Cashtags ($TSLA): normalized to TSLA — token kept, $ removed.
      Cashtag presence signals the text is explicitly about that stock.
    - Mentions (@user): stripped entirely — carry no sentiment signal.
    - Newlines: normalized to single space — FinBERT was trained on prose.
    - Short texts (<5 words after cleaning): flagged with is_too_short=True
      but NOT dropped here — that decision belongs in feature engineering.
    - Sarcasm (Reddit): cannot be resolved by regex. Documented as limitation.
    - One cleaner per source: noise profiles differ enough that a universal
      cleaner would either under-clean Twitter or over-clean news.

Usage:
    from src.nlp.cleaning import clean_text, flag_short_text

    df['text_clean']  = df.apply(lambda r: clean_text(r['text'], r['source']), axis=1)
    df['is_too_short'] = df['text_clean'].apply(flag_short_text)
"""

import re
import unicodedata

import pandas as pd


# ── Shared utility functions ───────────────────────────────────────────────────

def remove_urls(text: str) -> str:
    """Strip http/https URLs and bare www. links."""
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    return text


def remove_mentions(text: str) -> str:
    """Strip @username mentions — carry no sentiment signal."""
    return re.sub(r'@\w+', '', text)


def normalize_cashtags(text: str) -> str:
    """
    Normalize $TICKER → TICKER.
    Keeps the token; removes only the $ symbol.
    Cashtag presence signals the text is explicitly about that stock.
    """
    return re.sub(r'\$([A-Z]{1,5})', r'\1', text)


def remove_emojis(text: str) -> str:
    """Remove Unicode emoji blocks."""
    emoji_pattern = re.compile(
        '[\U00010000-\U0010ffff'
        '\U0001F300-\U0001F9FF'
        '\U00002702-\U000027B0'
        '\U000024C2-\U0001F251]+',
        flags=re.UNICODE
    )
    return emoji_pattern.sub('', text)


def normalize_whitespace(text: str) -> str:
    """Collapse newlines, tabs, and multiple spaces into a single space."""
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def remove_repeated_chars(text: str) -> str:
    """Reduce elongated characters: 'looooong' → 'long'. Keeps max 2 repeats."""
    return re.sub(r'(.)\1{2,}', r'\1\1', text)


def normalize_unicode(text: str) -> str:
    """Normalize unicode to NFC — handles accented chars and special quotes."""
    return unicodedata.normalize('NFC', text)


def flag_short_text(text: str, min_words: int = 5) -> bool:
    """Return True if text has fewer than min_words tokens after cleaning."""
    return len(text.split()) < min_words


# ── Source-specific cleaning functions ────────────────────────────────────────

def clean_news(text: str) -> str:
    """
    News cleaning pipeline.

    Day 1 findings (02_eda_finbert_pilot.ipynb):
        - Newlines: 35/50 HIGH  → normalize_whitespace (primary fix)
        - Short texts: 11/50   → flagged separately via flag_short_text()
        - URLs: 1/50 LOW       → stripped defensively
        - Mentions: 2/50 LOW   → stripped defensively

    Not applied to news:
        - Cashtag normalization (rare in news prose)
        - Emoji removal (0/50 in Day 1)
        - Markdown stripping (0/50 in Day 1)
    """
    text = normalize_unicode(text)
    text = remove_urls(text)
    text = remove_mentions(text)
    text = normalize_whitespace(text)
    return text


def clean_reddit(text: str) -> str:
    """
    Reddit cleaning pipeline.

    Day 1 findings:
        - Newlines: 25/50 HIGH  → normalize_whitespace
        - URLs: 8/50 HIGH       → remove_urls
        - Markdown: 5/50        → strip bold/italic/code/headers/blockquotes
        - Repeated chars: 4/50  → remove_repeated_chars
        - Short texts: 4/50     → flagged separately
        - Emojis: 1/50 LOW      → remove_emojis (defensive)

    Known limitation: Reddit sarcasm cannot be resolved by rule-based cleaning.
    Low-confidence FinBERT predictions on Reddit are expected to persist.
    """
    text = normalize_unicode(text)
    text = remove_urls(text)
    # Strip markdown syntax
    text = re.sub(r'\*{1,3}|_{1,3}|`{1,3}', '', text)          # bold, italic, code
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)  # headings
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)        # [anchor](url) → anchor
    text = re.sub(r'>\s?', '', text)                             # blockquotes
    text = remove_emojis(text)
    text = remove_repeated_chars(text)
    text = normalize_whitespace(text)
    return text


def clean_twitter_general(text: str) -> str:
    """
    twitter_general cleaning pipeline.

    Day 1 findings (most noise of any source):
        - Cashtags: 36/50 HIGH  → normalize_cashtags ($TSLA → TSLA)
        - URLs: 24/50 HIGH      → remove_urls
        - Emojis: 11/50 HIGH    → remove_emojis
        - Mentions: 10/50 HIGH  → remove_mentions
        - Hashtags: 6/50        → normalize (#Tesla → Tesla, token kept)
        - Newlines: 19/50       → normalize_whitespace
        - RT prefix: rare       → stripped

    Hashtag decision: same logic as cashtags — #Tesla is a positive
    signal, so the word is kept and only the # symbol is removed.
    """
    text = normalize_unicode(text)
    text = re.sub(r'^RT\s+', '', text.strip())  # remove RT prefix
    text = remove_urls(text)
    text = remove_mentions(text)
    text = normalize_cashtags(text)              # $TSLA → TSLA
    text = re.sub(r'#(\w+)', r'\1', text)        # #Tesla → Tesla
    text = remove_emojis(text)
    text = remove_repeated_chars(text)
    text = normalize_whitespace(text)
    return text


def clean_twitter_musk(text: str) -> str:
    """
    twitter_musk cleaning pipeline.

    Day 1 findings:
        - Mentions: 42/50 HIGH  → remove_mentions (primary fix)
        - Short texts: 9/50     → flagged separately
        - URLs: 7/50 HIGH       → remove_urls
        - Newlines: 9/50        → normalize_whitespace
        - Emojis: 2/50 LOW      → remove_emojis (defensive)
        - RT prefix: 3/50       → stripped

    Hypothesis validated in 03_nlp_cleaning_by_source.ipynb:
    removing 42/50 mentions redistributed labels away from neutral.
    """
    text = normalize_unicode(text)
    text = re.sub(r'^RT\s+', '', text.strip())  # remove RT prefix
    text = remove_urls(text)
    text = remove_mentions(text)                 # primary fix
    text = normalize_cashtags(text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = remove_emojis(text)
    text = remove_repeated_chars(text)
    text = normalize_whitespace(text)
    return text


# ── Dispatch router ───────────────────────────────────────────────────────────

CLEANER_MAP = {
    'news':             clean_news,
    'reddit':           clean_reddit,
    'twitter_general':  clean_twitter_general,
    'twitter_musk':     clean_twitter_musk,
}


def clean_text(text: str, source: str) -> str:
    """
    Route text to the correct source-specific cleaner.

    Parameters
    ----------
    text   : raw text string
    source : one of 'news', 'reddit', 'twitter_general', 'twitter_musk'

    Returns
    -------
    Cleaned text string. Returns '' for null/empty input.
    """
    if pd.isna(text) or str(text).strip() == '':
        return ''
    cleaner = CLEANER_MAP.get(source, lambda x: x)  # identity fallback for unknown sources
    return cleaner(str(text))
