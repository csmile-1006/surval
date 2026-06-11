"""Common episode abstraction shared by every dataset reader.

A *reader* turns a robot-learning dataset (robomimic/robocasa HDF5, a LeRobot
dataset, ...) into a stream of :class:`Episode` objects. The downstream
:func:`surval.ingest.builder.build_seqcache` orchestrator chunks each episode's
per-step actions into fixed-horizon rows, runs the user's policy callbacks, and
writes a canonical seqcache via :func:`surval.cache_io.write_seqcache_hdf5`.

Keeping the per-format parsing behind this one interface is what lets the same
cache-building loop work for any library: add a reader, reuse everything else.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Episode:
    """One trajectory/demo, with per-step (time-major) arrays.

    Attributes
    ----------
    demo_id
        Original identifier from the dataset (e.g. ``"demo_3"`` or
        ``"episode_000007"``). Used as the seqcache demo id.
    actions
        ``[T_ep, A]`` ground-truth per-step actions.
    obs
        Mapping of observation key -> ``[T_ep, ...]`` array. Whatever the reader
        chooses to surface (proprioceptive state, images, ...). Passed straight
        through to the user's ``predict_fn`` / ``feature_fn`` at the row's start
        frame, so the keys are a contract between the reader and those callbacks.
    state
        Optional ``[T_ep, D]`` low-dimensional proprioceptive state. Used as the
        ``obs_features`` fallback when no ``feature_fn`` is supplied. ``None`` if
        the reader cannot produce a single state vector.
    """

    demo_id: str
    actions: np.ndarray
    obs: Mapping[str, np.ndarray] = field(default_factory=dict)
    state: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.actions = np.asarray(self.actions)
        if self.actions.ndim != 2:
            raise ValueError(
                f"Episode {self.demo_id!r}: actions must be [T_ep, A]; got {self.actions.shape}"
            )
        t_ep = self.actions.shape[0]
        for k, v in self.obs.items():
            v = np.asarray(v)
            if v.shape[0] != t_ep:
                raise ValueError(
                    f"Episode {self.demo_id!r}: obs[{k!r}] has {v.shape[0]} steps "
                    f"but actions has {t_ep}"
                )
        if self.state is not None:
            self.state = np.asarray(self.state)
            if self.state.ndim != 2 or self.state.shape[0] != t_ep:
                raise ValueError(
                    f"Episode {self.demo_id!r}: state must be [T_ep, D] aligned with "
                    f"actions ({t_ep} steps); got {self.state.shape}"
                )

    @property
    def num_steps(self) -> int:
        return int(self.actions.shape[0])


class EpisodeReader:
    """Iterable of :class:`Episode`. Subclass per dataset format.

    Implementations override :meth:`__iter__` (and ideally :meth:`__len__`). The
    builder only relies on iteration, so a generator-backed reader is fine for
    datasets too large to index up front.
    """

    def __iter__(self) -> Iterator[Episode]:  # pragma: no cover - interface
        raise NotImplementedError

    def __len__(self) -> int:  # pragma: no cover - optional
        raise NotImplementedError
