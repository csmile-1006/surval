"""Extract ONLY the ground-truth signals from a training TensorBoard run.

The only things the metric actually needs from TensorBoard are the **success
rate** (ground truth) and **valid loss** (a baseline proxy). This pulls just
those two scalar series and saves them as a tiny ``.npz`` — no full-scalar JSON
cache, no aggregation pipeline. The proxy ("B") side comes from
``score_seqcache.py``, not from TensorBoard.

    python scripts/extract_tb_metrics.py --tb-dir <train_tb> --out gt.npz \
        --success-tag "Rollout/Success_Rate/TwoArmBoxCleanup-mean" \
        --valid-loss-tag "Valid/Loss"

Pass ``--list`` to print the available scalar tags and exit (handy to find the
exact success-rate tag, which includes the task name).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, os.pardir, "src"))
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from surval.tb_aggregate.tb_io import _load_event_accumulator, _latest_event_file, extract_tb_scalars  # noqa: E402


def _resolve_tag(available, tag):
    """Exact match, else unique substring match (so the task-name suffix on the
    success-rate tag doesn't have to be typed exactly)."""
    if tag in available:
        return tag
    hits = [t for t in available if tag in t]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise SystemExit(f"tag {tag!r} not found. Available:\n  " + "\n  ".join(available))
    raise SystemExit(f"tag {tag!r} is ambiguous, matches:\n  " + "\n  ".join(hits))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tb-dir", required=True, help="Training TensorBoard event dir.")
    ap.add_argument("--out", help="Output .npz path (required unless --list).")
    ap.add_argument("--success-tag", default="Success_Rate", help="Success-rate scalar tag (exact or unique substring).")
    ap.add_argument("--valid-loss-tag", default="Valid/Loss", help="Validation-loss scalar tag.")
    ap.add_argument("--all-event-files", action="store_true", help="Read all event files, not just the latest.")
    ap.add_argument("--list", action="store_true", help="Print available scalar tags and exit.")
    args = ap.parse_args()

    if args.list:
        load_path = args.tb_dir if args.all_event_files else (_latest_event_file(args.tb_dir) or args.tb_dir)
        acc = _load_event_accumulator(load_path)
        print("\n".join(acc.Tags().get("scalars", [])))
        return

    if not args.out:
        raise SystemExit("--out is required (or pass --list).")

    extracted = extract_tb_scalars(args.tb_dir, use_only_latest=not args.all_event_files)["tags"]
    available = list(extracted.keys())
    success_tag = _resolve_tag(available, args.success_tag)
    valid_tag = _resolve_tag(available, args.valid_loss_tag)

    np.savez(
        args.out,
        success_steps=np.asarray(extracted[success_tag]["steps"], dtype=np.int64),
        success_rate=np.asarray(extracted[success_tag]["values"], dtype=np.float64),
        valid_loss_steps=np.asarray(extracted[valid_tag]["steps"], dtype=np.int64),
        valid_loss=np.asarray(extracted[valid_tag]["values"], dtype=np.float64),
        success_tag=success_tag,
        valid_loss_tag=valid_tag,
    )
    n_s = len(extracted[success_tag]["steps"])
    n_v = len(extracted[valid_tag]["steps"])
    print(f"[extract] success_rate <- {success_tag!r} ({n_s} pts), valid_loss <- {valid_tag!r} ({n_v} pts)")
    print(f"[extract] saved -> {args.out}")


if __name__ == "__main__":
    main()
