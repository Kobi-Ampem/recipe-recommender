"""Grid search on the validation slice. Test is never touched here.

uv run python -m reciperec.tune --dataset foodcom --model itemknn

Every setting is fitted on train and scored on validation. All results go
to runs/tuning-<dataset>-<model>.json, saved after each run so a long grid
can be stopped and still keep what it found. The best by NDCG@10 is
recorded for `evaluate --tuned`.
"""

from __future__ import annotations

import argparse
import itertools
import json

from reciperec.data import LOADERS
from reciperec.evaluate import RUNS_DIR, fmt, run_one, stage_matrices

GRIDS = {
    "popularity": {},
    "itemknn": {"neighbors": [20, 50, 100, 200, 500, 1000], "shrink": [0, 50, 200, 500]},
    "puresvd": {"factors": [4, 8, 16, 32, 64, 128, 256]},
    "bpr": {
        "factors": [32, 128, 256],
        "lr": [0.003, 0.01],
        "reg": [0.003, 0.01, 0.03],
        "epochs": [40],
    },
}


def settings(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*grid.values())]


def tuning_path(dataset: str, model: str):
    return RUNS_DIR / f"tuning-{dataset}-{model}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(LOADERS), default="ml-1m")
    parser.add_argument("--model", choices=sorted(GRIDS), nargs="+", default=sorted(GRIDS))
    args = parser.parse_args()

    data = LOADERS[args.dataset]()
    fit, val = stage_matrices(data, "val")
    RUNS_DIR.mkdir(exist_ok=True)
    for name in args.model:
        grid = settings(GRIDS[name])
        print(f"{data.name} {name}: {len(grid)} settings")
        out = tuning_path(data.name, name)
        runs = []
        for params in grid:
            metrics = run_one(name, params, fit, val, early_stop=True)
            metrics.pop("history", None)
            runs.append(metrics)
            print(f"  {json.dumps(params):<60} {fmt(metrics)}  fit {metrics['fit_seconds']}s")
            best = max(runs, key=lambda m: m["ndcg@10"])
            out.write_text(
                json.dumps(
                    {"dataset": data.name, "model": name, "best": best, "runs": runs}, indent=2
                )
            )
        print(f"  best {json.dumps(best['params'])}  ndcg@10 {best['ndcg@10']:.4f}")


if __name__ == "__main__":
    main()
