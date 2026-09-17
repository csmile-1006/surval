"""Validation of cached DROID policies with local or shared-DINO features.

Heavy policy/RLDS dependencies belong to the cache producer. This module only
needs surval's CPU dependencies and FAISS for the existing threshold pipeline.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np

from .local_threshold.config import LocalThresholdConfig
from .local_threshold.database import StateDatabase, StateRecords
from .local_threshold.threshold import compute_global_thresholds_from_db, compute_local_thresholds
from .sequential_validate import _compute_from_cache, _discover_cache_groups, _load_cache_hdf5
from .tb_aggregate.seqcache_metrics import extract_seqcache_metrics_for_group
from .rlds_cache import row_keys_digest


DROID_ACTION_SPACE = {
    "action_dim": 10,
    "block_names": ["pos", "rot_6d"],
    "block_slices": {"pos": slice(0, 3), "rot_6d": slice(3, 9)},
    "block_dims": {"pos": 3, "rot_6d": 6},
    "arm_pairs": [],
    "scale_groups": [{"blocks": ["pos"], "summary_key": "s_pos"},
                     {"blocks": ["rot_6d"], "summary_key": "s_rot"}],
    "summary_scale_fields": [("ActionBlockScale_pos", "s_pos"), ("ActionBlockScale_rot", "s_rot")],
    "block_types": {"rot_6d": "rot6d"},
}

# 7 revolute joints plus a gripper dim that is excluded from scoring blocks,
# matching the pos_rot6d layout's treatment of gripper.
_JOINT_ACTION_SPACE = {
    "action_dim": 8,
    "block_names": [f"joint_{i}" for i in range(7)],
    "block_slices": {f"joint_{i}": slice(i, i + 1) for i in range(7)},
    "block_dims": {f"joint_{i}": 1 for i in range(7)},
    "arm_pairs": [],
    "scale_groups": [{"blocks": [f"joint_{i}" for i in range(7)], "summary_key": "s_joint"}],
    "summary_scale_fields": [("ActionBlockScale_joint", "s_joint")],
    "block_types": {},
}

_ACTION_SPACES = {
    "pos_rot6d": DROID_ACTION_SPACE,
    "cartesian_position_and_rot6d": DROID_ACTION_SPACE,
    "joint_velocity": _JOINT_ACTION_SPACE,
    "joint_position": _JOINT_ACTION_SPACE,
}


def action_space_config(name):
    """Consumer-side block layout for a DROID action space.

    Must stay consistent with ``LocalThresholdConfig.for_droid_action_space``:
    the threshold pass and the score pass have to agree on block names, slices
    and types or the scaled errors are meaningless.
    """
    spec = _ACTION_SPACES.get(str(name).lower())
    if spec is None:
        raise ValueError(
            f"Unknown droid_action_space {name!r}; choose one of {sorted(_ACTION_SPACES)}"
        )
    return spec


def denormalize_actions(actions, scale, offset):
    """Convert checkpoint-normalized actions to physical units."""
    scale, offset = np.asarray(scale), np.asarray(offset)
    action_dim = actions.shape[-1]
    if scale.shape != (action_dim,) or offset.shape != (action_dim,):
        raise ValueError(
            f"Normalization vectors must match the {action_dim}-D actions, "
            f"got scale {scale.shape} and offset {offset.shape}"
        )
    if not np.isfinite(scale).all() or not np.isfinite(offset).all() or np.any(scale <= 0):
        raise ValueError("Normalization must be finite with positive scales")
    return (actions * scale + offset).astype(np.float32)


def load_policy_cache(path, *, mode="epoch"):
    """Validate the producer contract before computing physical action errors.

    Two producers write this cache and their conventions differ, so both are
    recorded explicitly rather than assumed:

    ``action_space``
        ``"normalized_checkpoint"`` (robomimic/Octo) stores checkpoint-normalized
        actions plus ``action_scale``/``action_offset`` and is denormalized here.
        ``"physical"`` (openpi) already unnormalized before writing.
    ``action_start_offset``
        ``1`` for Octo's ``action[1:]`` convention (observation t -> action t+1),
        ``0`` for openpi's RLDS chunking (observation t -> action t).
    ``mode``
        ``"epoch"`` or ``"step"`` -- openpi indexes checkpoints by training step.
        The returned ``epoch`` key carries whichever index the cache uses.
    """
    data = _load_cache_hdf5(path, load_obs_features=True, mode=mode)
    ids, times, actions, predictions, features, checkpoint, index, loss, omn = data
    with h5py.File(path, "r") as f:
        if int(f.attrs.get("complete", 0)) != 1 or int(f.attrs.get("droid_cache_version", 0)) != 1:
            raise ValueError(f"Regenerate legacy/incomplete DROID cache: {path}")
        action_space = str(f.attrs.get("action_space", ""))
        if action_space not in ("normalized_checkpoint", "physical"):
            raise ValueError(
                f"Unknown cache action_space {action_space!r}; "
                "expected 'normalized_checkpoint' or 'physical'"
            )
        start_offset = int(f.attrs.get("action_start_offset", -1))
        if start_offset not in (0, 1):
            raise ValueError(
                "Cache must record action_start_offset (0 = observation t -> action t, "
                "1 = observation t -> action t+1)"
            )
        provenance = json.loads(f.attrs["provenance"])
        feature_source = str(f.attrs["feature_source"])
        if action_space == "normalized_checkpoint":
            scale = np.asarray(f.attrs["action_scale"])
            offset = np.asarray(f.attrs["action_offset"])
        else:
            scale = offset = None
    droid_action_space = str(provenance.get("droid_action_space", "pos_rot6d"))
    if (provenance.get("producer") == "robomimic_diffusion_policy"
            and (action_space != "normalized_checkpoint" or start_offset != 1
                 or droid_action_space != "pos_rot6d")):
        raise ValueError("Robomimic DROID requires normalized pos_rot6d actions starting at t+1")
    expected_dim = action_space_config(droid_action_space)["action_dim"]
    if index < 0 or features is None or features.shape[0] != actions.shape[0] or not len(actions):
        raise ValueError("A non-empty cache with aligned encoder features is required")
    if actions.shape[-1] != expected_dim or predictions.shape[1:] != actions.shape:
        raise ValueError(
            f"Cache actions must be {expected_dim}-D for {droid_action_space!r} and "
            "predictions must match GT dimensions exactly"
        )
    if len(set(zip(ids.tolist(), times.tolist()))) != len(ids):
        raise ValueError("Duplicate (demo_id, index_in_demo) rows in cache")
    coverage = provenance.get("row_coverage")
    if (provenance.get("producer") == "robomimic_diffusion_policy"
            and provenance.get("producer_version", 1) >= 2 and coverage is None):
        raise ValueError("Producer v2 cache is missing its independent row coverage manifest")
    if coverage is not None and (coverage["num_rows"] != len(ids)
            or coverage["num_demos"] != len(set(ids))
            or coverage["row_keys_sha256"] != row_keys_digest(list(zip(ids.tolist(), times.tolist())))):
        raise ValueError("Cached rows differ from their independent coverage manifest")
    if not all(np.isfinite(a).all() for a in (actions, predictions, features)):
        raise ValueError("Cache contains non-finite actions or encoder features")
    if provenance.get("feature_source") != feature_source:
        raise ValueError("Encoder provenance differs between cache attributes and manifest")
    if scale is None:
        gt = np.asarray(actions, dtype=np.float32)
        pred = np.asarray(predictions, dtype=np.float32)
    else:
        gt = denormalize_actions(actions, scale, offset)
        pred = denormalize_actions(predictions, scale, offset)
    return {
        "demo_ids": ids, "index_in_demo": times,
        "actions": gt, "pred_actions": pred,
        "obs_features": features, "checkpoint": checkpoint, "epoch": index,
        "valid_loss": loss, "valid_omn": omn, "provenance": provenance,
        "droid_action_space": droid_action_space, "action_start_offset": start_offset,
        "action_scale": scale, "action_offset": offset,
    }


def _source_signature(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def build_policy_thresholds(cache_path, cache, output_dir, cfg):
    """Build/reuse thresholds for this exact checkpoint's cached encoder rows."""
    source = {"cache": _source_signature(cache_path), "config": cfg.to_dict()}
    # JSON canonicalization also makes tuple-valued configs compare after reload.
    source = json.loads(json.dumps(source, sort_keys=True))
    signature = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()[:16]
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"epoch_{cache['epoch']:06d}_{signature}"
    if target.exists():
        if json.loads((target / "source.json").read_text()) != source:
            raise ValueError(f"Threshold source metadata mismatch: {target}")
        return str(target)

    features = cache["obs_features"].astype(np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
        raise ValueError("Policy encoder features must have nonzero finite norms")
    ids = cache["demo_ids"]
    names, inverse = np.unique(ids, return_inverse=True)
    records = StateRecords(
        actions=cache["actions"][:, 0, :], demo_id_int=inverse.astype(np.int32),
        t=cache["index_in_demo"].astype(np.int64),
        demo_id_str_by_int={i: str(name) for i, name in enumerate(names)},
    )
    db = StateDatabase(cfg)
    db.attach(features / norms, records)
    global_scales = compute_global_thresholds_from_db(db, cfg)
    thresholds = compute_local_thresholds(db, cfg, global_scales)
    if not np.isfinite(thresholds.thresholds).all():
        raise ValueError("Non-finite local thresholds")
    with tempfile.TemporaryDirectory(prefix=".threshold-", dir=root) as temporary:
        staging = Path(temporary) / "state_db"
        db.save(str(staging))
        thresholds.save(str(staging / "thresholds"))
        (staging / "source.json").write_text(json.dumps(source, indent=2))
        os.replace(staging, target)
    return str(target)


def score_policy_cache(cache, state_db_dir, *, quantile=0.95, ta=None, num_samples=None,
                       chunk_top_frac=0.5, lse_tau=1.0, every_step=False):
    """Delegate to the library's continuous score, with no mode selection."""
    summary, episodes = _compute_from_cache(
        cache["demo_ids"], cache["index_in_demo"], cache["actions"], cache["pred_actions"],
        ta=ta, num_diffusion_samples=num_samples or cache["pred_actions"].shape[0],
        prefix_epsilon=0.0, prefix_tau=1.0, prefix_epsilon_soft=1.0,
        prefix_mode="soft", prefix_scoring_mode="cumprod",
        action_space_config=action_space_config(cache["droid_action_space"]),
        block_scale_method="local", state_db_dir=state_db_dir, threshold_quantile=quantile,
        block_share_scales_across_arms=False, distance_mode="L2",
        chunk_time_agg="top_k_mean", chunk_top_frac=chunk_top_frac,
        skip_intermediate_on_pass=not every_step,
        prefix_soft_aggregator="logsumexp", prefix_soft_lse_tau=lse_tau,
        prefix_time_reduction="product", valid_loss=cache["valid_loss"], valid_off=cache["valid_omn"],
    )
    if summary.get("LocalThreshold_MissingRows") != 0:
        raise ValueError("Scored rows are missing from their checkpoint's state database")
    return summary, episodes


def _json_value(value):
    if isinstance(value, dict):
        return {k: _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def evaluate_policy_caches(cache_dir, output_dir, *, quantile=0.95, k_neighbors=50,
                           min_neighbors=10, temporal_radius=5, ta=None, num_samples=None,
                           chunk_top_frac=0.5, lse_tau=1.0, every_step=False, mode="epoch",
                           dino_db_root=None, dino_omn_k=5, exact_neighbors=False, scale_multiplier=1.0):
    """Write checkpoint-level JSON/CSV for scoring and custom-outcome aggregation."""
    if not 0 < quantile <= 1 or not 0 < chunk_top_frac <= 1 or lse_tau <= 0:
        raise ValueError("Invalid quantile, chunk fraction or LSE temperature")
    if k_neighbors < 1 or min_neighbors < 2 or temporal_radius < 0:
        raise ValueError("k must be positive, min_neighbors >= 2 and temporal_radius >= 0")
    if (ta is not None and ta < 1) or (num_samples is not None and num_samples < 1):
        raise ValueError("ta and num_samples must be positive")
    if mode not in ("epoch", "step"):
        raise ValueError(f"Unknown cache mode {mode!r}; choose 'epoch' or 'step'")
    if not np.isfinite(scale_multiplier) or scale_multiplier <= 0:
        raise ValueError("scale_multiplier must be finite and positive")
    if dino_db_root is None and (exact_neighbors or scale_multiplier != 1.0):
        raise ValueError("Exact-neighbor/scale controls currently require --dino-db-root")
    if dino_omn_k < 1:
        raise ValueError("dino_omn_k must be positive")
    groups = _discover_cache_groups(cache_dir, mode=mode)
    if not groups:
        raise ValueError(f"No {mode} caches under {cache_dir}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results, rows, shared = [], [], {}
    if dino_db_root is not None:
        from .droid_dino import (DINO_OMN, align_dino_cache, build_dino_thresholds,
                                compute_dino_omn, dataset_db_path, dino_neighbors, load_shared_dino_db)
    for group in groups:
        baselines = extract_seqcache_metrics_for_group(group["group_dir"], mode)["tags"]
        for path in group["cache_files"]:
            cache = load_policy_cache(path, mode=mode)
            p = cache["provenance"]
            cfg = LocalThresholdConfig.for_droid_action_space(
                cache["droid_action_space"],
                encoder_name=f"{cache['checkpoint']}:{p['feature_source']}",
                encoder_pool="flatten_observation_horizon", encoder_device="cached",
                state_window_size=p["observation_horizon"], use_proprio=False,
                data_dir=str(Path(path).parent), dataset_name=p["dataset_name"],
                actions_from_cache=True, quantiles=(quantile,), k_neighbors=k_neighbors,
                min_neighbors_for_local=min_neighbors, temporal_exclusion_radius=temporal_radius,
                ta=ta or cache["actions"].shape[1],
            )
            dino_info = None
            if dino_db_root is None:
                state_dir = build_policy_thresholds(path, cache, output / "thresholds", cfg)
            else:
                dataset = p["dataset_name"]
                if dataset not in shared:
                    db_path = dataset_db_path(dino_db_root, dataset)
                    db, chunks, metadata = load_shared_dino_db(db_path)
                    state_dir = build_dino_thresholds(
                        db, metadata, output / "thresholds", quantile=quantile,
                        k_neighbors=k_neighbors, min_neighbors=min_neighbors,
                        temporal_radius=temporal_radius, ta=ta or cache["actions"].shape[1],
                        exact_neighbors=exact_neighbors, scale_multiplier=scale_multiplier,
                    )
                    neighbors = dino_neighbors(db, k=dino_omn_k, temporal_radius=temporal_radius)
                    shared[dataset] = db, chunks, metadata, neighbors, state_dir, str(db_path)
                db, chunks, metadata, neighbors, state_dir, db_path = shared[dataset]
                order = align_dino_cache(db, chunks, metadata, cache)
                dino_value, dino_info = compute_dino_omn(cache, chunks, order, neighbors)
                dino_info.update({"k_requested": dino_omn_k, "temporal_radius": temporal_radius,
                                  "db_path": db_path, "db_content_sha256": metadata["content_sha256"]})
            summary, episodes = score_policy_cache(
                cache, state_dir, quantile=quantile, ta=ta, num_samples=num_samples,
                chunk_top_frac=chunk_top_frac, lse_tau=lse_tau, every_step=every_step,
            )
            epoch = cache["epoch"]
            row = {
                "run": p["run"], "run_timestamp": p["run_timestamp"], "dataset": p["dataset_name"],
                # ``epoch`` stays the canonical join key for the aggregator; in
                # step mode it holds the training step, mirrored below so the
                # CSV never hides which index it actually is.
                "epoch": epoch, "checkpoint_index_kind": mode, mode: epoch,
                "droid_action_space": cache["droid_action_space"],
                "checkpoint": cache["checkpoint"], "cache_file": str(Path(path).resolve()),
                "surval_score": summary["PrefixSurvival_Score"], "quantile": quantile,
                "num_rows": len(cache["actions"]), "num_demos": len(set(cache["demo_ids"])),
                "num_samples": num_samples or cache["pred_actions"].shape[0],
                "ta": ta or cache["actions"].shape[1],
            }
            for tag, series in baselines.items():
                values = dict(zip(series["steps"], series["values"]))
                row[tag.removeprefix("Cache/Valid/")] = values.get(epoch)
            if dino_db_root is not None:
                row[DINO_OMN] = dino_value
            row["neighbor_representation"] = "shared_dino" if dino_db_root is not None else "policy_encoder"
            rows.append(_json_value(row))
            results.append(_json_value({**row, "state_db_dir": state_dir,
                                       "summary": summary, "per_episode": episodes,
                                       "dino_omn": dino_info, "provenance": p}))
            print(f"epoch={epoch} SURVAL={row['surval_score']:.6f} dataset={row['dataset']}", flush=True)
    payload = {
        "score_formula": "surval continuous / logsumexp blocks / cumulative product",
        "checkpoint_index_kind": mode,
        "action_units": "score and thresholds: physical; baselines: original cache coordinates",
        "settings": {"quantile": quantile, "k_neighbors": k_neighbors, "min_neighbors": min_neighbors,
                     "temporal_radius": temporal_radius, "chunk_top_frac": chunk_top_frac,
                     "lse_tau": lse_tau, "every_step": every_step,
                     "dino_db_root": str(dino_db_root) if dino_db_root is not None else None,
                     "dino_omn_k": dino_omn_k, "exact_neighbors": exact_neighbors,
                     "scale_multiplier": scale_multiplier},
        "results": results,
    }
    for name in ("checkpoint_metrics.json", "checkpoint_metrics.csv"):
        fd, temporary = tempfile.mkstemp(prefix=".metrics-", dir=output)
        try:
            with os.fdopen(fd, "w", newline="") as f:
                if name.endswith(".json"):
                    json.dump(payload, f, indent=2, allow_nan=False)
                else:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            os.replace(temporary, output / name)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return payload
