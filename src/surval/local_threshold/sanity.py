"""Sanity check helpers for Phase 1 (DB build) and Phase 2 (threshold compute).

Each function writes figures/text under ``output_dir/figures/`` and returns a
short markdown snippet that the calling script appends to its sanity report.
A failed check raises (per spec §3.5: do not silently warn).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
import os

import numpy as np

from .config import LocalThresholdConfig
from .database import StateDatabase
from .threshold import LocalThresholdMap


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _import_plt():
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    return plt


# -------- Phase 1 sanity --------


def check_encoder_determinism(encoder, n_images: int = 2) -> str:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(n_images, encoder.cfg.image_size, encoder.cfg.image_size, 3), dtype=np.uint8)
    f1 = encoder.encode(img).numpy()
    f2 = encoder.encode(img).numpy()
    diff = float(np.max(np.abs(f1 - f2)))
    if diff >= 1e-5:
        raise AssertionError(f"encoder non-deterministic: max abs diff = {diff}")
    return f"- **Encoder determinism**: PASS (max abs diff = {diff:.2e})\n"


def check_modality_balance(
    image_features_per_view: dict[str, np.ndarray],
    proprio_array: np.ndarray,
    cfg: LocalThresholdConfig,
    *,
    window_indices: np.ndarray | None = None,
    sample_size: int = 1000,
) -> str:
    from .state_embed import modality_block_norms

    # Compute per-row norms on the full array (windowed) so the gather indices
    # remain valid, then sub-sample the resulting norms for the report stats.
    norms = modality_block_norms(
        image_features_per_view, proprio_array, cfg, window_indices=window_indices
    )
    n = next(iter(norms.values())).shape[0]
    sample_size = min(sample_size, n)
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=sample_size, replace=False)
    norms = {k: v[idx] for k, v in norms.items()}

    lines = ["- **Modality balance** (per-block norms; should be ~1.0):\n"]
    for name, n_arr in norms.items():
        lines.append(f"  - {name}: mean={n_arr.mean():.4f}, std={n_arr.std():.4f}\n")
        if abs(n_arr.mean() - 1.0) > 1e-3:
            raise AssertionError(f"{name} norm mean {n_arr.mean()} not ~1.0")
    return "".join(lines)


def check_index_round_trip(db: StateDatabase, n_queries: int = 100) -> str:
    rng = np.random.default_rng(0)
    n = db.n_states
    if n == 0:
        raise AssertionError("DB is empty")
    n_queries = min(n_queries, n)
    idx = rng.choice(n, size=n_queries, replace=False)
    q = db.embeddings[idx]
    distances, indices = db.query(q, k=1)
    if not np.array_equal(indices[:, 0], idx):
        raise AssertionError("Top-1 of DB members must be self")
    max_d = float(np.max(distances[:, 0]))
    if max_d > 1e-4:
        raise AssertionError(f"Self-query distance {max_d} too large")
    return f"- **Index round-trip**: PASS (max self-distance = {max_d:.2e})\n"


def visual_neighbor_inspection(
    db: StateDatabase,
    get_image: Callable[[int, str], np.ndarray],  # (state_idx, view) -> uint8 (H, W, 3)
    output_dir: str,
    n_queries: int = 5,
    k: int = 10,
) -> str:
    """Save top-k neighbor image grids for a few random queries.

    ``get_image`` is a callable that returns the raw image for any
    (state_idx, view) pair — the function uses it for both the queries it
    samples and the FAISS neighbors it discovers, so the caller does not need
    to pre-build a full lookup dict.
    """
    plt = _import_plt()
    _ensure_dir(output_dir)
    rng = np.random.default_rng(0)
    n = db.n_states
    n_queries = min(n_queries, n)
    query_idx = rng.choice(n, size=n_queries, replace=False)
    q_emb = db.embeddings[query_idx]
    _, indices = db.query(q_emb, k=k + 1)

    views = list(db.cfg.image_views)

    paths = []
    for q_pos, q in enumerate(query_idx):
        nbs = [int(x) for x in indices[q_pos] if int(x) != int(q)][:k]
        n_views = len(views)
        fig, axes = plt.subplots(n_views, k + 1, figsize=(2 * (k + 1), 2 * n_views))
        if n_views == 1:
            axes = axes[None, :]
        for vi, view in enumerate(views):
            axes[vi, 0].imshow(get_image(int(q), view))
            axes[vi, 0].set_title(f"Q (idx={int(q)}, t={int(db.records.t[int(q)])})", fontsize=7)
            axes[vi, 0].axis("off")
            for j, nb in enumerate(nbs):
                axes[vi, j + 1].imshow(get_image(nb, view))
                axes[vi, j + 1].set_title(
                    f"nb (d={int(db.records.demo_id_int[nb])}, t={int(db.records.t[nb])})",
                    fontsize=7,
                )
                axes[vi, j + 1].axis("off")
        out_path = os.path.join(output_dir, f"phase1_neighbors_q{q_pos:02d}.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=80)
        plt.close(fig)
        paths.append(out_path)

    md = "- **Visual neighbor inspection**: figures saved\n"
    for p in paths:
        md += f"  - `{p}`\n"
    return md


# -------- Phase 2 sanity --------


def check_fallback_rate(m: LocalThresholdMap, threshold: float = 0.30) -> str:
    rate = float(m.fallback_used.mean())
    extra = ""
    if rate > threshold:
        extra = (
            f" — WARNING: > {threshold:.0%}. Consider lowering min_neighbors_for_local or temporal_exclusion_radius."
        )
    return f"- **Fallback rate**: {rate:.2%}{extra}\n"


def plot_local_vs_global(m: LocalThresholdMap, output_dir: str) -> str:
    """For each (block, quantile) pair, plot histogram of S_local / S_global."""
    plt = _import_plt()
    _ensure_dir(output_dir)
    n_blocks = len(m.block_names)
    n_q = len(m.quantiles)
    fig, axes = plt.subplots(n_blocks, n_q, figsize=(2.5 * n_q, 2.0 * n_blocks), squeeze=False)
    eps = 1e-12
    stats_lines = ["- **Local / global ratio statistics** (excluding fallbacks):\n"]
    for bi, b in enumerate(m.block_names):
        for qi, q in enumerate(m.quantiles):
            local = m.thresholds[~m.fallback_used, bi, qi]
            g = max(float(m.global_thresholds[bi, qi]), eps)
            ratio = local / g
            ax = axes[bi, qi]
            if ratio.size > 0:
                ax.hist(ratio, bins=40, range=(0.0, min(3.0, float(ratio.max()) * 1.05 if ratio.size else 3.0)))
            ax.axvline(1.0, color="red", linestyle="--", linewidth=0.8)
            ax.set_title(f"{b} q={q}", fontsize=7)
            ax.tick_params(labelsize=6)
            if ratio.size > 0:
                stats_lines.append(
                    f"  - {b} q={q}: mean={ratio.mean():.3f}, std={ratio.std():.3f}, "
                    f"p5={np.percentile(ratio, 5):.3f}, p95={np.percentile(ratio, 95):.3f}\n"
                )
    fig.tight_layout()
    out = os.path.join(output_dir, "phase2_local_vs_global.png")
    fig.savefig(out, dpi=80)
    plt.close(fig)
    return f"- **Local vs global histogram**: `{out}`\n" + "".join(stats_lines)


def plot_threshold_over_time(
    db: StateDatabase,
    m: LocalThresholdMap,
    output_dir: str,
    quantile_to_plot: float | None = None,
    n_demos: int = 5,
) -> str:
    """For a few random demos, plot S_g(s_t) over t. Should be piecewise smooth.

    Plots at the median quantile by default.
    """
    plt = _import_plt()
    _ensure_dir(output_dir)
    if db.records is None:
        raise RuntimeError("DB not built")

    qi = len(m.quantiles) // 2 if quantile_to_plot is None else m.quantile_index(quantile_to_plot)
    q_value = m.quantiles[qi]

    rng = np.random.default_rng(0)
    unique_demos = np.unique(db.records.demo_id_int)
    pick = rng.choice(unique_demos, size=min(n_demos, unique_demos.size), replace=False)

    n_blocks = len(m.block_names)
    fig, axes = plt.subplots(len(pick), 1, figsize=(7, 2.0 * len(pick)), squeeze=False)
    for di, demo in enumerate(pick):
        mask = db.records.demo_id_int == demo
        order = np.argsort(db.records.t[mask])
        ts = db.records.t[mask][order]
        ax = axes[di, 0]
        for bi, b in enumerate(m.block_names):
            vals = m.thresholds[mask, bi, qi][order]
            ax.plot(ts, vals, label=b, linewidth=0.8)
        ax.set_title(f"demo {int(demo)} (q={q_value})", fontsize=8)
        ax.set_xlabel("t", fontsize=7)
        ax.set_ylabel("S_g(s_t)", fontsize=7)
        ax.tick_params(labelsize=6)
        if di == 0:
            ax.legend(fontsize=6, ncol=n_blocks)
    fig.tight_layout()
    out = os.path.join(output_dir, "phase2_threshold_over_time.png")
    fig.savefig(out, dpi=80)
    plt.close(fig)
    return f"- **Spatial smoothness over time**: `{out}` (q={q_value})\n"


def write_markdown_report(path: str, title: str, sections: Iterable[str]) -> None:
    with open(path, "w") as f:
        f.write(f"# {title}\n\n")
        for s in sections:
            f.write(s)
            if not s.endswith("\n"):
                f.write("\n")
