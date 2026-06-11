"""Generic seqcache ingestion: read any robot-learning dataset, build a cache.

Pair a dataset *reader* with your policy callbacks and let
:func:`build_seqcache` produce a canonical seqcache HDF5:

    from surval.ingest import RobomimicHDF5Reader, build_seqcache

    reader = RobomimicHDF5Reader("demos.hdf5", split="valid",
                                 obs_keys=["robot0_eef_pos", "robot0_eef_quat"],
                                 state_keys=["robot0_eef_pos", "robot0_eef_quat"])
    build_seqcache(
        "out/seqcache_step_000600.hdf5", reader,
        predict_fn=my_policy_predict,   # obs_batch -> [S, B, T_h, A]
        feature_fn=my_encoder,          # obs_batch -> [B, F]  (optional; else state)
        horizon=15, num_samples=8, step=600,
    )

Readers provided: :class:`RobomimicHDF5Reader` (robomimic / robocasa /
dexmimicgen HDF5) and :class:`LeRobotReader` (LeRobot v2.x parquet). Add a reader
for any other format by subclassing :class:`EpisodeReader`.
"""

from __future__ import annotations

from .base import Episode, EpisodeReader
from .builder import build_seqcache
from .chunking import chunk_actions
from .lerobot import LeRobotReader
from .robomimic import RobomimicHDF5Reader

__all__ = [
    "Episode",
    "EpisodeReader",
    "build_seqcache",
    "chunk_actions",
    "LeRobotReader",
    "RobomimicHDF5Reader",
]
