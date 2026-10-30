"""Build docs/results.md and the README results table from the saved runs.

uv run python -m reciperec.report

Reads runs/<dataset>.json (test results) and runs/tuning-<dataset>-<model>.json
(validation grids), so every number in the README traces back to a run.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR, DOCS_DIR = ROOT / "runs", ROOT / "docs"
DATASETS = [("foodcom-5core", "Food.com (5-core)"), ("ml-1m", "MovieLens 1M (harness check)")]
ORDER = ["popularity", "itemknn", "puresvd", "bpr"]
NAMES = {"popularity": "Popularity", "itemknn": "Item-kNN", "puresvd": "PureSVD", "bpr": "BPR-MF"}


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def settings(model: str, dataset: str, tuned: dict | None) -> str:
    if model == "popularity":
        return "none"
    if not tuned:
        return "not tuned"
    best = tuned["best"]
    params = {k: v for k, v in best["params"].items() if k != "epochs"}
    if "best_epoch" in best:
        params["epochs"] = best["best_epoch"]
    return ", ".join(f"{k} {v}" for k, v in params.items())


def slices_section(results: dict) -> list[str]:
    models = [m for m in ORDER if "slices" in results.get(m, {})]
    if not models:
        return []
    bins = list(results[models[0]]["slices"]["by_activity"])
    head = "| Training interactions per user | Users | " + " | ".join(NAMES[m] for m in models)
    lines = [
        "",
        "**Test NDCG@10 by user activity**",
        "",
        head + " |",
        "|---|---:|" + "---:|" * len(models),
    ]
    for b in bins:
        cells = [results[m]["slices"]["by_activity"].get(b, {}).get("ndcg@10") for m in models]
        users = results[models[0]]["slices"]["by_activity"][b]["users"]
        lines.append(
            f"| {b} | {users:,} | "
            + " | ".join("" if c is None else f"{c:.4f}" for c in cells)
            + " |"
        )
    lines += [
        "",
        "**Popularity bias**: share of top-10 recommendations from the most popular 1% of "
        "recipes in training: "
        + ", ".join(f"{NAMES[m]} {results[m]['slices']['top1pct_share']:.0%}" for m in models)
        + ".",
    ]
    return lines


def dataset_section(key: str, title: str) -> list[str]:
    run = load(RUNS_DIR / f"{key}.json")
    if not run:
        return [f"## {title}", "", "No test run saved yet.", ""]
    results = run["results"]
    lines = [
        f"## {title}",
        "",
        (
            "| Model | P@10 | R@10 | NDCG@10 | vs popularity [95% CI] | Coverage@10 | Val NDCG@10 "
            "| Settings (chosen on validation) | Fit |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for model in [m for m in ORDER if m in results]:
        r = results[model]
        tuned = load(RUNS_DIR / f"tuning-{key}-{model}.json")
        lift = ""
        if "vs_popularity" in r:
            v = r["vs_popularity"]
            lo, hi = v["ci95"]
            lift = f"{v['gain']:+.0%} [{lo:+.0%}, {hi:+.0%}]" + ("" if v["significant"] else " ns")
        val = f"{tuned['best']['ndcg@10']:.4f}" if tuned else ""
        lines.append(
            f"| {NAMES[model]} | {r['precision@10']:.4f} | {r['recall@10']:.4f} "
            f"| **{r['ndcg@10']:.4f}** | {lift} | {r['coverage@10']:.1%} | {val} "
            f"| {settings(model, key, tuned)} | {r['fit_seconds']:.0f} s |"
        )
    lines += ["", f"{results[ORDER[0]]['users']:,} test users." if "popularity" in results else ""]
    compared = [m for m in ORDER[1:] if "vs_popularity" in results.get(m, {})]
    if compared:
        sig = [NAMES[m] for m in compared if results[m]["vs_popularity"]["ci95"][0] > 0]
        rest = [NAMES[m] for m in compared if results[m]["vs_popularity"]["ci95"][0] <= 0]
        lines += [
            "",
            (
                '"vs popularity" is the relative NDCG@10 gain with a 95% paired bootstrap '
                "interval over test users (2,000 resamples); ns means the interval includes 0."
            ),
            "",
            f"Beat popularity beyond chance: {', '.join(sig) or 'none'}.",
        ]
        if rest:
            lines += [f"Not distinguishable from popularity: {', '.join(rest)}."]
        vals = {
            m: load(RUNS_DIR / f"tuning-{key}-{m}.json")["best"]["ndcg@10"]
            for m in results
            if (RUNS_DIR / f"tuning-{key}-{m}.json").exists()
        }
        best_val = max(vals, key=vals.get) if vals else None
        best_test = max(results, key=lambda m: results[m]["ndcg@10"])
        if best_val and best_val != best_test:
            lines += [
                "",
                (
                    f"Best on validation: {NAMES[best_val]}; best on test: {NAMES[best_test]}. "
                    "Settings were fixed on validation and not changed after the test run."
                ),
            ]
    lines += slices_section(results)
    if key == "ml-1m":
        lines += [
            "",
            (
                "MovieLens 1M is here only to check the harness. Published numbers use other "
                "splits and metrics, so the check is the ordering: every personalised model "
                "should beat popularity, and a tuned item-kNN matching or beating matrix "
                "factorisation is a known result (Dacrema et al., 2019)."
            ),
        ]
    return [*lines, ""]


README_START, README_END = "<!-- results:start -->", "<!-- results:end -->"


def readme_table() -> list[str]:
    """Compact test table for the README: NDCG@10, gain over popularity, coverage."""
    lines = [
        "| Dataset | Model | NDCG@10 | vs popularity [95% CI] | Coverage@10 |",
        "|---|---|---:|---:|---:|",
    ]
    for key, title in DATASETS:
        run = load(RUNS_DIR / f"{key}.json")
        if not run:
            continue
        for model in [m for m in ORDER if m in run["results"]]:
            r = run["results"][model]
            gain = ""
            if "vs_popularity" in r:
                v = r["vs_popularity"]
                lo, hi = v["ci95"]
                gain = f"{v['gain']:+.0%} [{lo:+.0%}, {hi:+.0%}]"
            name = title.split(" (")[0]
            lines.append(
                f"| {name} | {NAMES[model]} | {r['ndcg@10']:.4f} | {gain} | {r['coverage@10']:.1%} |"
            )
    return lines


def update_readme() -> None:
    path = ROOT / "README.md"
    text = path.read_text()
    if README_START not in text:
        return
    head, rest = text.split(README_START, 1)
    _, tail = rest.split(README_END, 1)
    body = "\n".join(readme_table())
    path.write_text(f"{head}{README_START}\n{body}\n{README_END}{tail}")


def main() -> None:
    out = [
        "# Results",
        "",
        "Generated by `uv run python -m reciperec.report` from `runs/`. Do not edit by hand.",
        "",
        (
            "Protocol: each user's timeline is split oldest ~64% train, next ~16% validation, newest "
            "20% test. Settings are chosen by validation NDCG@10; the test run then fits on train "
            "plus validation with those settings, once. Every model ranks the full catalogue with "
            "the user's own training items removed."
        ),
        "",
    ]
    for key, title in DATASETS:
        out += dataset_section(key, title)
    DOCS_DIR.mkdir(exist_ok=True)
    path = DOCS_DIR / "results.md"
    path.write_text("\n".join(out).rstrip() + "\n")
    update_readme()
    print(path.read_text())


if __name__ == "__main__":
    main()
