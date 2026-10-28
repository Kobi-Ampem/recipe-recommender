# recipe-recommender

Recommends recipes from what people have cooked and reviewed on Food.com, with nutrition filters.
Recipe suggestions only, not dietary or medical advice.

Work in progress (sprint 28 to 30 Oct 2026). Results table to follow.

## Run

```bash
uv sync
uv run python -m reciperec.evaluate --dataset ml-1m      # downloads MovieLens 1M
uv run python -m reciperec.profile                       # Food.com profile -> docs/data.md
uv run python -m reciperec.evaluate --dataset foodcom    # needs the Kaggle data, below
uv run pytest
```

## Data

Neither dataset is committed: both licences forbid redistribution.

- **Food.com** (Majumder et al., 2019): only `RAW_interactions.csv` and `RAW_recipes.csv` from
  Kaggle's `shuyangli94/food-com-recipes-and-user-interactions` are needed, in `data/foodcom/`:

  ```bash
  mkdir -p data/foodcom && cd data/foodcom
  for f in RAW_interactions.csv RAW_recipes.csv; do
    kaggle datasets download shuyangli94/food-com-recipes-and-user-interactions -f "$f" --unzip
  done
  ```

  The profile (counts, rating skew, sparsity before and after the 5-core) is in
  [docs/data.md](docs/data.md).
- **MovieLens 1M** (Harper and Konstan, 2015): downloaded automatically. Used only to check the
  evaluation code against a well-known benchmark.

## Evaluation

Each user's most recent 20% of interactions are held out for testing (a time split, so the model
never trains on the future). Every model ranks the full catalogue, items already seen in training
are removed, and the top 10 are scored with precision, recall, NDCG and catalogue coverage.
