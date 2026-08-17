# Step 2 — NLP Preprocessing

## Purpose
Clean and normalize the raw 4-source corpus so it can be consistently fed into the sentiment models in Step 3. Uses a two-track architecture: raw `text` preserved for FinBERT, `tokens_lemma` (cleaned/lemmatized) preserved for classical baselines.

## Datasets (hosted on Kaggle)

| Content | Dataset | Link |
|---|---|---|
| Cleaned layer (post source-specific cleaning) | `layer2-cleaned` | [Kaggle](https://www.kaggle.com/datasets/leev75/layer2-cleaned) |
| Fully preprocessed layer (`layer2_preprocessed`) | `processd` | [Kaggle](https://www.kaggle.com/datasets/leev75/processd) |

## Processing applied
- Text cleaning (HTML/URL stripping, encoding fixes), done per-source before merging
- Deduplication on `doc_id` (interim fix for the Step 1 collision bug — see Step 1 README)
- Tokenization, stopword removal, lemmatization (classical-model track only)
- Named entity recognition
- Token length filtering — see `figures/06_token_lengths.png` for the distribution used to set truncation limits (FinBERT `max_len=512`)

## Figures
- `figures/06_token_lengths.png` — token length distribution across the four sources

## Notebooks
- `notebooks/03_nlp_cleaning_by_source.ipynb` — source-specific cleaning → produces `layer2-cleaned`
- `notebooks/04_nlp_preprocessing.ipynb` — tokenization/lemmatization/NER → produces `processd` (`layer2_preprocessed`)
- `notebooks/04.1_nlp_preprocessing_validation.ipynb` — validation checks on the preprocessed output

## Next step
Feeds into `step3_sentiment_analysis/`.
