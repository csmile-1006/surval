"""Join DROID checkpoint proxies to custom outcomes using explicit row keys."""

from collections import defaultdict
import csv
from pathlib import Path

import numpy as np

from .metrics import compute_seed_metrics, nanmean, nanstd
from .seqcache_metrics import LOSS_TAG, OMN_TAG, MSE_VARIANT_TAGS
from .tb_io import save_cache_file


ROW_KEY = ("run", "run_timestamp", "dataset", "epoch")
PROXY_DIRECTIONS = {"surval_score": 1, **{
    tag.removeprefix("Cache/Valid/"): -1
    for tag in [LOSS_TAG, OMN_TAG, *MSE_VARIANT_TAGS]
}}


OPTIONAL_PROXY_DIRECTIONS = {"Off_Manifold_Norm_DINO": -1}


def _read_rows(path, required, optional=()):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = set(ROW_KEY + tuple(required)) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns in {path}: {sorted(missing)}")
        rows = {}
        for row in reader:
            key = tuple(row[name] for name in ROW_KEY[:-1]) + (int(row["epoch"]),)
            if not all(key[:-1]) or key[-1] < 0:
                raise ValueError(f"Invalid checkpoint key: {key}")
            if key in rows:
                raise ValueError(f"Duplicate checkpoint key in {path}: {key}")
            for name in (*required, *(name for name in optional if name in row)):
                value = row[name]
                row[name] = float(value) if value is not None and value.strip() else None
                if row[name] is not None and not np.isfinite(row[name]):
                    raise ValueError(f"Non-finite {name} at {key}; leave missing values blank")
            rows[key] = row
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def aggregate_droid_metrics(metrics_csv, outcomes_csv, output_json, *, k_list=(1, 3, 5)):
    """Evaluate SURVAL and available baselines; custom outcome is higher-is-better.

    Each run/timestamp/dataset is a checkpoint-selection problem. Missing values
    are omitted per proxy, with exact epoch coverage reported. Dataset summaries
    weight runs equally, never treating checkpoints as independent seeds.
    """
    if not k_list or any(k < 1 or int(k) != k for k in k_list):
        raise ValueError("Top-k values must be positive integers")
    if Path(output_json).resolve() in {Path(metrics_csv).resolve(), Path(outcomes_csv).resolve()}:
        raise ValueError("Output must not overwrite either input CSV")
    proxies = _read_rows(metrics_csv, PROXY_DIRECTIONS, OPTIONAL_PROXY_DIRECTIONS)
    directions = {**PROXY_DIRECTIONS, **{name: direction for name, direction in OPTIONAL_PROXY_DIRECTIONS.items()
                                       if name in next(iter(proxies.values()))}}
    outcomes = _read_rows(outcomes_csv, ("outcome",))
    unknown = outcomes.keys() - proxies.keys()
    if unknown:
        raise ValueError(f"Outcome keys absent from metrics CSV: {sorted(unknown)[:5]}")
    groups = defaultdict(list)
    for key in sorted(proxies):
        groups[key[:-1]].append(key)
    per_group, skipped = [], []
    for group, keys in groups.items():
        identity = dict(zip(ROW_KEY[:-1], group))
        for method, direction in directions.items():
            paired = [key for key in keys if outcomes.get(key, {}).get("outcome") is not None
                      and proxies[key][method] is not None]
            coverage = {**identity, "method": method, "num_checkpoints": len(paired),
                        "epochs": [key[-1] for key in paired],
                        "omitted_epochs": [key[-1] for key in keys if key not in paired]}
            if len(paired) < 2:
                skipped.append({**coverage, "reason": "Need at least two paired checkpoints"})
                continue
            actual = np.array([outcomes[key]["outcome"] for key in paired])
            proxy = direction * np.array([proxies[key][method] for key in paired])
            metrics = compute_seed_metrics(actual, proxy, k_list=k_list, eps=1e-12,
                                           tie_tol=0.0, nan_policy="omit")
            selected, oracle = int(np.argmax(proxy)), int(np.argmax(actual))
            per_group.append({**coverage, "higher_is_better": direction > 0,
                              "selected_epoch": paired[selected][-1],
                              "oracle_epoch": paired[oracle][-1],
                              "selected_outcome": float(actual[selected]),
                              "oracle_outcome": float(actual[oracle]),
                              "metrics": {name: float(value) if np.isfinite(value) else None
                                          for name, value in metrics.items()}})
    by_dataset = defaultdict(list)
    for row in per_group:
        by_dataset[(row["dataset"], row["method"])].append(row)
    aggregate = []
    for (dataset, method), rows in sorted(by_dataset.items()):
        statistics = {}
        for name in rows[0]["metrics"]:
            values = [row["metrics"][name] for row in rows if row["metrics"][name] is not None]
            statistics[name] = {"mean": nanmean(values) if values else None,
                                "std": nanstd(values) if values else None,
                                "n_groups": len(values)}
        aggregate.append({"dataset": dataset, "method": method, "n_groups": len(rows),
                          "metrics": statistics})
    result = {
        "metrics_csv": str(Path(metrics_csv).resolve()),
        "outcomes_csv": str(Path(outcomes_csv).resolve()),
        "outcome_direction": "higher_is_better", "k_list": list(k_list),
        "aggregation": "equal weight per run/timestamp within each dataset; population std",
        "alignment": "explicit row key; per-proxy finite pairs, see epochs/omitted_epochs",
        "ties": "earliest epoch wins, matching existing surval tie_tol=0 convention",
        "delta_correlations": "differences between adjacent available epochs, not per-epoch slopes",
        "num_unlabelled_checkpoints": sum(outcomes.get(key, {}).get("outcome") is None for key in proxies),
        "per_group": per_group, "aggregate": aggregate, "skipped": skipped,
    }
    save_cache_file(result, str(output_json))
    return result
