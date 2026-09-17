"""Finite RLDS validation loading with independently checked row coverage.

TensorFlow and Octo are imported only by the loader, not by the coverage checks.
"""

from collections import Counter
import hashlib
import json
import time

import numpy as np


def row_key(row):
    source = row["demo_id"]
    source = source.decode("utf-8") if isinstance(source, bytes) else str(source)
    index = int(row["index_in_demo"])
    if not source or index < 0:
        raise ValueError("Expected a non-empty source ID and nonnegative timestep")
    return source, index


def validate_row_coverage(expected, actual, *, require_order=False):
    """Counts alone cannot detect a missing episode, tail or substituted row."""
    expected, actual = list(expected), list(actual)
    for label, keys in (("expected", expected), ("actual", actual)):
        counts = Counter(keys)
        duplicates = [key for key, count in counts.items() if count != 1]
        if duplicates:
            raise ValueError(f"Duplicate {label} row keys: {duplicates[:5]}")
    missing, unexpected = set(expected) - set(actual), set(actual) - set(expected)
    if missing or unexpected:
        raise ValueError(f"Row coverage mismatch: missing={sorted(missing)[:5]}, "
                         f"unexpected={sorted(unexpected)[:5]}")
    if require_order and actual != expected:
        raise ValueError("Rows must be in canonical source/timestep order")


def row_keys_digest(keys):
    return hashlib.sha256(json.dumps(sorted(keys), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def materialize_validation_rows(dataset, frame_transform_kwargs, row_transform, *,
                                parallelism=4, max_demos=None, max_rows=None):
    """Inspect filtered/windowed trajectories, then parallel-decode a fixed subset.

    The caller must require stable source IDs in Octo. This deliberately retains
    the existing in-memory decoded dataset reused across checkpoints. The cheap
    metadata pass does not decode images; no repeated/sampled training input is
    accepted. Subsets are selected lexicographically, independently of arrival.
    """
    import tensorflow as tf
    from octo.data.dataset import apply_frame_transforms

    if parallelism < 1 or any(x is not None and x < 1 for x in (max_demos, max_rows)):
        raise ValueError("Parallelism and optional row/demo limits must be positive")
    started = time.perf_counter()
    options = tf.data.Options()
    options.deterministic = False
    options.threading.private_threadpool_size = parallelism
    options.threading.max_intra_op_parallelism = 1
    dataset = dataset.with_options(options)
    lengths = {}
    metadata = dataset.traj_map(lambda traj: {k: traj[k] for k in ("demo_id", "index_in_demo")},
                                num_parallel_calls=parallelism)
    for traj in metadata.as_numpy_iterator():
        sources, times = traj["demo_id"], traj["index_in_demo"]
        if not len(times) or len(sources) != len(times):
            raise ValueError("Empty or misaligned trajectory metadata")
        source, _ = row_key({"demo_id": sources[0], "index_in_demo": times[0]})
        if source in lengths or not np.all(sources == sources[0]):
            raise ValueError(f"Duplicate or changing source episode ID: {source}")
        if not np.array_equal(times, np.arange(len(times))):
            raise ValueError(f"Expected every original timestep before subsetting: {source}")
        lengths[source] = len(times)
    selected = {}
    remaining = max_rows
    for source in sorted(lengths)[:max_demos]:
        count = lengths[source] if remaining is None else min(lengths[source], remaining)
        if count:
            selected[source] = count
        if remaining is not None:
            remaining -= count
            if remaining == 0:
                break
    expected = [(source, t) for source, count in selected.items() for t in range(count)]
    if not expected:
        raise ValueError("Validation selection is empty")

    limits = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(
        tf.constant(list(selected)), tf.constant(list(selected.values()), dtype=tf.int64)), default_value=0)
    # Windowing already happened: slicing here preserves future GT for capped rows.
    dataset = dataset.filter(lambda traj: limits.lookup(traj["demo_id"][0]) > 0)
    dataset = dataset.traj_map(lambda traj: tf.nest.map_structure(
        lambda value: value[:limits.lookup(traj["demo_id"][0])], traj), num_parallel_calls=parallelism)
    dataset = dataset.flatten(num_parallel_calls=parallelism)
    dataset = apply_frame_transforms(dataset, **{**frame_transform_kwargs, "num_parallel_calls": parallelism},
                                     train=False)
    dataset = dataset.map(row_transform, num_parallel_calls=parallelism)
    # The finite source/subset has stable IDs; completion order may now vary.
    dataset = dataset.with_options(options).prefetch(1)
    samples = list(dataset.as_numpy_iterator())
    validate_row_coverage(expected, [row_key(row) for row in samples])
    samples.sort(key=row_key)
    validate_row_coverage(expected, [row_key(row) for row in samples], require_order=True)
    digest = row_keys_digest(expected)
    return {"dataset": samples, "row_keys": expected,
            "coverage": {"version": 1, "num_rows": len(expected), "num_demos": len(selected),
                         "source_num_rows": selected, "row_keys_sha256": digest,
                         "row_order": "source_id_lexicographic_then_timestep",
                         "source_id": "episode_metadata.file_path"},
            "load_seconds": time.perf_counter() - started}
