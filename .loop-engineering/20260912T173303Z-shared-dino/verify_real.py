"""Run from surval/: PYTHONPATH=src .venv-droid/bin/python <this file>."""

import hashlib
import json
from pathlib import Path

import numpy as np

from surval.droid import load_policy_cache
from surval.droid_dino import align_dino_cache, compute_dino_omn, dino_neighbors, load_shared_dino_db
from surval.tb_aggregate.seqcache_metrics import extract_seqcache_metrics_for_group


root = Path("outputs/droid_dino_apple_val5_20260912")
rows = json.loads((root / "checkpoint_metrics.json").read_text())["results"]
assert len(rows) == 10 and len({r["state_db_dir"] for r in rows}) == 1
db, chunks, meta = load_shared_dino_db(rows[0]["dino_omn"]["db_path"])
neighbors = dino_neighbors(db, k=5, temporal_radius=5)
assert all(len(n) == 5 for n in neighbors)
baselines = extract_seqcache_metrics_for_group(str(Path(rows[0]["cache_file"]).parent), "epoch")["tags"]
for row in rows:
    assert row["num_rows"] == 488 and row["num_samples"] == 8
    for tag, series in baselines.items():
        assert row[tag.removeprefix("Cache/Valid/")] == dict(zip(series["steps"], series["values"]))[row["epoch"]]
    assert np.isfinite(row["Off_Manifold_Norm_DINO"])
    assert row["dino_omn"]["rows_without_neighbors"] == 0
cache = load_policy_cache(rows[0]["cache_file"])
order = align_dino_cache(db, chunks, meta, cache)
for i in [0, 17, 100, 487]:
    similarities = db.embeddings @ db.embeddings[i]
    eligible = ((db.records.demo_id_int != db.records.demo_id_int[i])
                | (abs(db.records.t - db.records.t[i]) > 5))
    expected = np.flatnonzero(eligible)[np.argsort(-similarities[eligible])[:5]]
    assert set(neighbors[i]) == set(expected)
values = []
for indices in np.array_split(np.arange(488), 7):
    part = {**cache, "pred_actions": cache["pred_actions"][:, indices]}
    value, _ = compute_dino_omn(part, chunks, order[indices], neighbors)
    values.append(value * len(indices))
np.testing.assert_allclose(sum(values) / 488, rows[0]["Off_Manifold_Norm_DINO"], rtol=1e-12)

# Reproduce the pre-build `sort | xargs sha256sum | sha256sum` manifest.
paths = sorted(Path("outputs/droid_cache_val5_10_20_30_20260910/conditions").rglob("seqcache_epoch_*.hdf5"))
assert len(paths) == 120
manifest = hashlib.sha256()
for path in paths:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    manifest.update(f"{digest.hexdigest()}  {path}\n".encode())
assert manifest.hexdigest() == "64e8e49941e2e7e1aa5671b12b01c31300c84914e1ef71d8ff27e1cf9c1639bc"
print(json.dumps({"checkpoints": 10, "rows": 488, "neighbors_per_row": 5,
                  "baseline_values_unchanged": True, "input_caches_unchanged": 120,
                  "brute_force_knn_verified": True, "seven_way_batch_partition_invariant": True,
                  "shared_threshold_dirs": 1, "db_sha256": meta["content_sha256"]}, indent=2))
