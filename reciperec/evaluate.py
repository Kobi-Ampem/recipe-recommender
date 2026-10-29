"""Train and score models on one dataset.

uv run python -m reciperec.evaluate --dataset ml-1m --model popularity
uv run python -m reciperec.evaluate --dataset foodcom --model puresvd --params '{"factors": 64}'

--stage val fits on train and scores on validation (for choosing settings).
--stage test, the default, fits on train plus validation and scores on test.
Test is the final exam: score it once per model, with settings chosen on val.
"""

from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path

import pandas as pd

from reciperec.data import LOADERS, Interactions
from reciperec.metrics import evaluate
from reciperec.models import MODELS
from reciperec.split import to_csr, train_val_test_split

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def stage_matrices(data: Interactions, stage: str, test_frac: float = 0.2, val_frac: float = 0.2):
    """(fit matrix, score matrix) for the stage."""
    train, val, test = train_val_test_split(data.df, test_frac, val_frac)
    if stage == "val":
        fit_df, score_df = train, val
    else:
        fit_df, score_df = pd.concat([train, val]), test
    return (
        to_csr(fit_df, data.n_users, data.n_items),
        to_csr(score_df, data.n_users, data.n_items),
    )


def run_one(name: str, params: dict, fit, score, k: int = 10, early_stop: bool = False) -> dict:
    """Fit one model and return its metrics, settings and timings.

    With early_stop (validation stage only), models that support it watch
    the score matrix while training. Never pass it for test.
    """
    model = MODELS[name](**params)
    start = time.perf_counter()
    if early_stop and "val" in inspect.signature(model.fit).parameters:
        model.fit(fit, val=score)
    else:
        model.fit(fit)
    fit_s = time.perf_counter() - start
    start = time.perf_counter()
    metrics = evaluate(model, fit, score, k=k)
    metrics["fit_seconds"] = round(fit_s, 2)
    metrics["score_seconds"] = round(time.perf_counter() - start, 2)
    metrics["params"] = params
    metrics.update(getattr(model, "fit_info", {}))
    return metrics


def fmt(metrics: dict) -> str:
    return "  ".join(f"{m} {v:.4f}" for m, v in metrics.items() if "@" in m)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(LOADERS), default="ml-1m")
    parser.add_argument("--model", choices=sorted(MODELS), nargs="+", default=["popularity"])
    parser.add_argument("--params", type=json.loads, default={}, help="JSON, for one model")
    parser.add_argument("--stage", choices=["val", "test"], default="test")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    if args.params and len(args.model) > 1:
        parser.error("--params applies to one model at a time")

    data = LOADERS[args.dataset]()
    fit, score = stage_matrices(data, args.stage)
    print(
        f"{data.name} [{args.stage}]: {data.n_users:,} users, {data.n_items:,} items; "
        f"fit {fit.nnz:,}, score {score.nnz:,}"
    )

    RUNS_DIR.mkdir(exist_ok=True)
    out = RUNS_DIR / f"{data.name}.json" if args.stage == "test" else None
    saved = json.loads(out.read_text()) if out and out.exists() else {"results": {}}
    for name in args.model:
        metrics = run_one(name, args.params, fit, score, args.k, args.stage == "val")
        print(f"  {name:<12} {fmt(metrics)}  fit {metrics['fit_seconds']}s")
        saved["results"][name] = metrics

    if out:
        saved.update(dataset=data.name, k=args.k, test_frac=0.2, val_frac=0.2)
        out.write_text(json.dumps(saved, indent=2))
        print(f"Saved {out.relative_to(RUNS_DIR.parent)}")


if __name__ == "__main__":
    main()
