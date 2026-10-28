# recipe-recommender

Recommends recipes from what people have cooked and reviewed on Food.com, with nutrition filters.
Recipe suggestions only, not dietary or medical advice.

Work in progress (sprint 28 to 30 Oct 2026). Results table to follow.

## Run

```bash
uv sync
uv run python -m reciperec.evaluate --dataset ml-1m      # downloads MovieLens 1M
uv run python -m reciperec.evaluate --dataset foodcom    # needs the Kaggle data, below
uv run pytest
```

## Data

Neither dataset is committed: both licences forbid redistribution.

- **Food.com** (Majumder et al., 2019): download
  `shuyangli94/food-com-recipes-and-user-interactions` from Kaggle and unzip into `data/foodcom/`.
- **MovieLens 1M** (Harper and Konstan, 2015): downloaded automatically. Used only to check the
  evaluation code against a well-known benchmark.

## Evaluation

Each user's most recent 20% of interactions are held out for testing (a time split, so the model
never trains on the future). Every model ranks the full catalogue, items already seen in training
are removed, and the top 10 are scored with precision, recall, NDCG and catalogue coverage.
