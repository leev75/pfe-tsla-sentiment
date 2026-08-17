# Step 1 — Data Collection

## Purpose
Collect raw financial text and market data for TSLA (Jan 2020 – Dec 2023) across four independent sources, forming the basis of the sentiment corpus used in later steps.

## Datasets (hosted on Kaggle)

| Source | Dataset | Link |
|---|---|---|
| Reddit (part 1) | `reddit-s1-parquet` | [Kaggle](https://www.kaggle.com/datasets/leev75/reddit-s1-parquet) |
| Reddit (part 2, 2022) | `reddit-s2-2022-parquet` | [Kaggle](https://www.kaggle.com/datasets/leev75/reddit-s2-2022-parquet) |
| Reddit (part 3) | `reddit-s3-parquet` | [Kaggle](https://www.kaggle.com/datasets/leev75/reddit-s3-parquet) |
| Twitter (general) | `tsla-twitter-general` | [Kaggle](https://www.kaggle.com/datasets/leev75/layer1) |
| News | `layer1-newsupdated` | [Kaggle](https://www.kaggle.com/datasets/leev75/layer1-newsupdated) |

> **Note:** Reddit is split across 3 datasets due to Kaggle upload size limits — concatenate them before use.

## Known gap
The **Elon Musk Twitter stream** (kept architecturally separate from general Twitter, per the project's design — see thesis §6 System Design) does not currently have a corresponding public Kaggle dataset. If you locate/re-export it, add the link here and update the total document count below accordingly.

## Total corpus
**85,649 documents** across all four sources combined (news + Reddit + Twitter general + Twitter Musk), as documented in the thesis (§4 Background / §7 Implementation).

## Notebooks
- `notebooks/01_data_collection_layer2_merge.ipynb` — merges the raw sources into a unified layer
- `notebooks/02_eda_finbert_pilot.ipynb` — exploratory analysis / early FinBERT pilot on the collected corpus

## Next step
Feeds into `step2_nlp_preprocessing/`.
