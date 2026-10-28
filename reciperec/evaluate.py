"""Train and score models on one dataset.

uv run python -m reciperec.evaluate --dataset ml-1m --model popularity
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from reciperec.data import LOADERS
from reciperec.metrics import evaluate
from reciperec.models import MODELS
from reciperec.split import temporal_user_split, to_csr

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(LOADERS), default="ml-1m")
    parser.add_argument("--model", choices=sorted(MODELS), nargs="+", default=sorted(MODELS))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--test-frac", type=float, default=0.2)
    args = parser.parse_args()

    data = LOADERS[args.dataset]()
    train_df, test_df = temporal_user_split(data.df, args.test_frac)
    train = to_csr(train_df, data.n_users, data.n_items)
    test = to_csr(test_df, data.n_users, data.n_items)
    density = data.df.shape[0] / (data.n_users * data.n_items)
    print(
        f"{data.name}: {data.n_users:,} users, {data.n_items:,} items, "
        f"{len(data.df):,} interactions (density {density:.4%}); "
        f"train {len(train_df):,}, test {len(test_df):,}"
    )

    results = {}
    for name in args.model:
        start = time.perf_counter()
        model = MODELS[name]().fit(train)
        fit_s = time.perf_counter() - start
        metrics = evaluate(model, train, test, k=args.k)
        metrics["fit_seconds"] = round(fit_s, 2)
        results[name] = metrics
        print(
            f"  {name:<12} "
            + "  ".join(f"{m} {v:.4f}" for m, v in metrics.items() if isinstance(v, float))
        )

    RUNS_DIR.mkdir(exist_ok=True)
    out = RUNS_DIR / f"{data.name}.json"
    out.write_text(
        json.dumps(
            {"dataset": data.name, "k": args.k, "test_frac": args.test_frac, "results": results},
            indent=2,
        )
    )
    print(f"Saved {out.relative_to(RUNS_DIR.parent)}")


if __name__ == "__main__":
    main()
