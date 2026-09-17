"""Checkpoint-independent validation DINO database and full-split OMN."""

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .cache_io import project_neighbor_actions
from .local_threshold.config import LocalThresholdConfig
from .local_threshold.database import StateDatabase, StateRecords
from .local_threshold.state_embed import build_window_indices, compose_state_vectors_batch
from .local_threshold.threshold import compute_global_thresholds_from_db, compute_local_thresholds
from .rlds_cache import row_key, row_keys_digest, validate_row_coverage


DINO_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINO_OMN = "Off_Manifold_Norm_DINO"


def dataset_db_path(root, dataset_name):
    path = Path(dataset_name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("dataset_name must be a non-empty relative path without '..'")
    target = (Path(root) / path).resolve()
    if not target.is_relative_to(Path(root).resolve()):
        raise ValueError("Dataset DB path escapes its root")
    return target


def _keys(db):
    r = db.records
    return [(r.demo_id_str_by_int[int(d)], int(t)) for d, t in zip(r.demo_id_int, r.t)]


def _content_hash(db, chunks):
    digest = hashlib.sha256(row_keys_digest(_keys(db)).encode())
    # Include order as well as coverage: the arrays must match the same ordered rows.
    digest.update(json.dumps(_keys(db), separators=(",", ":")).encode())
    for array in (db.embeddings, db.records.actions, chunks):
        digest.update(str((array.shape, str(array.dtype))).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def save_shared_dino_db(target, db, chunks, metadata):
    """Publish a complete DB atomically. Never overwrite an existing artifact."""
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    if (chunks.ndim != 3 or chunks.shape[0] != db.n_states
            or not np.isfinite(chunks).all()
            or not np.array_equal(chunks[:, 0], db.records.actions)):
        raise ValueError("Expert chunks must be finite and aligned with DB first actions")
    target.parent.mkdir(parents=True, exist_ok=True)
    meta = {**metadata, "version": 1, "complete": True,
            "content_sha256": _content_hash(db, chunks), "num_rows": db.n_states}
    with tempfile.TemporaryDirectory(prefix=".dino-", dir=target.parent) as temporary:
        staging = Path(temporary) / "db"
        db.save(str(staging))
        np.save(staging / "expert_action_chunks.npy", chunks)
        (staging / "shared_dino_meta.json").write_text(json.dumps(meta, indent=2))
        os.rename(staging, target)
    return target


def align_dino_cache(db, chunks, metadata, cache):
    """Return DB row indices in cache order; reject partial or incompatible splits."""
    p = cache["provenance"]
    for name, value in (("dataset_name", p["dataset_name"]),
                        ("observation_horizon", p["observation_horizon"]),
                        ("action_start_offset", cache["action_start_offset"]),
                        ("droid_action_space", cache["droid_action_space"])):
        if metadata.get(name) != value:
            raise ValueError(f"Shared DINO DB {name} differs from cache")
    if metadata.get("dataset_files") != p.get("dataset_files"):
        raise ValueError("Shared DINO DB dataset provenance differs from cache")
    keys = _keys(db)
    cache_keys = list(zip(cache["demo_ids"].tolist(), cache["index_in_demo"].tolist()))
    validate_row_coverage(keys, cache_keys)
    lookup = {key: i for i, key in enumerate(keys)}
    order = np.array([lookup[key] for key in cache_keys], dtype=np.int64)
    if (chunks.shape != cache["actions"].shape
            or not np.allclose(chunks[order], cache["actions"], rtol=1e-5, atol=1e-6)):
        raise ValueError("Shared DINO expert action chunks differ from cache GT/offset/normalization")
    return order


def load_shared_dino_db(path):
    path = Path(path)
    meta = json.loads((path / "shared_dino_meta.json").read_text())
    if meta.get("version") != 1 or meta.get("complete") is not True or meta.get("split") != "val":
        raise ValueError("Expected a complete, version-1 full-validation DINO DB")
    cfg = LocalThresholdConfig(**json.loads((path / "config.json").read_text()))
    if not cfg.encoder_name.startswith("dinov2_") or cfg.use_proprio or cfg.index_type != "flat_ip":
        raise ValueError("Shared DB must use image-only DINO with exact cosine retrieval")
    if (cfg.dataset_name != meta.get("dataset_name")
            or cfg.state_window_size != meta.get("observation_horizon")
            or cfg.droid_action_space != meta.get("droid_action_space")):
        raise ValueError("DINO DB config differs from its metadata")
    db = StateDatabase.load(str(path), cfg)
    chunks = np.load(path / "expert_action_chunks.npy", allow_pickle=False)
    if (chunks.ndim != 3 or chunks.shape[0] != db.n_states
            or not np.isfinite(chunks).all() or not np.array_equal(chunks[:, 0], db.records.actions)
            or meta.get("num_rows") != db.n_states or _content_hash(db, chunks) != meta["content_sha256"]):
        raise ValueError("Shared DINO DB content/row alignment is corrupt")
    validate_row_coverage(_keys(db), _keys(db))
    # Validate unit norms and rebuild the cheap exact index from verified arrays.
    db.attach(db.embeddings, db.records)
    return db, chunks, meta


def build_dino_thresholds(db, metadata, output_dir, *, quantile, k_neighbors, min_neighbors,
                          temporal_radius, ta, exact_neighbors=False, scale_multiplier=1.0):
    if not np.isfinite(scale_multiplier) or scale_multiplier <= 0:
        raise ValueError("scale_multiplier must be finite and positive")
    cfg = replace(db.cfg, quantiles=(quantile,), k_neighbors=k_neighbors,
                  min_neighbors_for_local=min_neighbors, temporal_exclusion_radius=temporal_radius, ta=ta)
    source = {"shared_dino": metadata, "config": cfg.to_dict()}
    if exact_neighbors or scale_multiplier != 1.0:
        source.update(exact_neighbors=bool(exact_neighbors), scale_multiplier=float(scale_multiplier))
    source = json.loads(json.dumps(source, sort_keys=True))
    signature = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()[:16]
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"dino_{signature}"
    if target.exists():
        if json.loads((target / "source.json").read_text()) != source:
            raise ValueError(f"DINO threshold provenance mismatch: {target}")
        return str(target)
    global_scales = compute_global_thresholds_from_db(db, cfg, exact_neighbors=exact_neighbors)
    thresholds = compute_local_thresholds(db, cfg, global_scales, exact_neighbors=exact_neighbors)
    thresholds.thresholds *= np.float32(scale_multiplier)
    thresholds.global_thresholds *= np.float32(scale_multiplier)
    if not np.isfinite(thresholds.thresholds).all():
        raise ValueError("Non-finite DINO thresholds")
    with tempfile.TemporaryDirectory(prefix=".threshold-", dir=root) as temporary:
        staging = Path(temporary) / "db"
        db.save(str(staging))
        (staging / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
        thresholds.save(str(staging / "thresholds"))
        (staging / "source.json").write_text(json.dumps(source, indent=2))
        os.rename(staging, target)
    return str(target)


def dino_neighbors(db, *, k=5, temporal_radius=5):
    """One exact full-split cosine query per observation, never per chunk element."""
    if k < 1 or temporal_radius < 0:
        raise ValueError("k must be positive and temporal_radius nonnegative")
    # Unique timesteps bound excluded same-demo rows by 2*radius+1.
    _, indices = db.query(db.embeddings, k=k + 2 * temporal_radius + 1)
    cfg = replace(db.cfg, same_demo_allowed=True, temporal_exclusion_radius=temporal_radius)
    filtered = db.filter_neighbors(indices, db.records.demo_id_int, db.records.t, cfg)
    return [row[:k] for row in filtered]


def compute_dino_omn(cache, chunks, order, neighbors):
    """Same-offset expert projection, averaged over all saved rows/horizon/samples.

    Retrieval is checkpoint-independent. Projection uses checkpoint-normalized
    action coordinates (physical coordinates for a physical-only producer).
    No-neighbor rows make the metric unavailable, not spuriously perfect.
    """
    counts = np.array([len(neighbors[i]) for i in order])
    info = {"reference": "full_validation_split", "distance": "cosine",
            "action_alignment": "same_chunk_offset", "self_excluded": True,
            "num_rows": len(order), "rows_without_neighbors": int((counts == 0).sum()),
            "min_neighbors": int(counts.min()), "max_neighbors": int(counts.max()),
            "action_units": "checkpoint_normalized" if cache.get("action_scale") is not None else "physical",
            "num_samples": cache["pred_actions"].shape[0], "horizon": chunks.shape[1]}
    if not counts.all():
        return None, {**info, "per_sample": None, "reason": "No eligible neighbors for some rows"}
    predictions = cache["pred_actions"]
    scale, offset = cache.get("action_scale"), cache.get("action_offset")
    if scale is not None:
        chunks, predictions = (chunks - offset) / scale, (predictions - offset) / scale
    errors = np.empty(predictions.shape[:3], dtype=np.float64)
    for count in np.unique(counts):
        selected = np.flatnonzero(counts == count)
        idx = np.stack([neighbors[order[i]] for i in selected])
        for h in range(chunks.shape[1]):
            experts = chunks[idx, h, :]
            for sample in range(len(predictions)):
                errors[sample, selected, h] = project_neighbor_actions(predictions[sample, selected, h], experts)
    per_sample = errors.mean(axis=(1, 2))
    return float(per_sample.mean()), {**info, "per_sample": per_sample.tolist()}


def encode_droid_rows(rows, cache, cfg, encoder):
    """Encode current images once, then reuse the library's causal history gather."""
    keys = [row_key(row) for row in rows]
    cache_keys = list(zip(cache["demo_ids"].tolist(), cache["index_in_demo"].tolist()))
    validate_row_coverage(cache_keys, keys, require_order=True)
    names, inverse = np.unique(cache["demo_ids"], return_inverse=True)
    times = cache["index_in_demo"]
    for demo in np.unique(inverse):
        if not np.array_equal(times[inverse == demo], np.arange(sum(inverse == demo))):
            raise ValueError("DINO history requires contiguous complete demo timesteps")
    start = cfg.state_window_size - 1
    gt = np.stack([row["actions"][start:start + cache["actions"].shape[1]] for row in rows])
    gt = gt * cache["action_scale"] + cache["action_offset"]
    if gt.shape != cache["actions"].shape or not np.allclose(gt, cache["actions"], rtol=1e-5, atol=1e-6):
        raise ValueError("RLDS expert chunks do not match cache time/action axes")
    features = {}
    for view in cfg.image_views:
        batches = []
        for begin in range(0, len(rows), cfg.encoder_batch_size):
            images = np.stack([r["obs"][view][-1] for r in rows[begin:begin + cfg.encoder_batch_size]])
            batches.append(np.asarray(encoder.encode(images), dtype=np.float32))
        features[view] = np.concatenate(batches)
    windows = build_window_indices(inverse, times, cfg.state_window_size)
    embeddings = compose_state_vectors_batch(features, np.empty((len(rows), 0)), cfg, windows)
    records = StateRecords(cache["actions"][:, 0], inverse.astype(np.int32), times.astype(np.int64),
                           {i: str(name) for i, name in enumerate(names)})
    db = StateDatabase(cfg)
    db.attach(embeddings, records)
    return db
