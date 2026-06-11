"""
Filesystem layout helpers for local_threshold seqval runs.

Train side:
    <train_root>/<seed_name>/<timestamp>/logs/tb/events.out.tfevents.*

Eval side:
    <eval_root>/<seed_name>/<hp_dir>/<cache_group_dir>/tb/events.out.tfevents.*

Where:
    seed_name        e.g. flow_policy_image_ds_two_arm_drawer_cleanup_D0_n_demo_500_seed_8
    hp_dir           tq{tq}_lsetau{lsetau}_ctf{ctf}_ta{ta}_nds{nds}_{share|noshare}_{skip|noskip}_{L2|maha}
                     optional trailing _ab{baseline|nogrp|fixthr|fixthr_intra|tmean
                     |globaldb|q05|q99} for leave-one-out ablation +
                     sanity-baseline runs.
    cache_group_dir  seqcache_ds<dataset>_mode<mode>_split<split>_vkey<vkey>_efd<efd>
"""

from __future__ import annotations

import os
import re
from typing import Iterator


SEED_DIR_PATTERN = re.compile(r".*_seed_(\d+)$")

LOCAL_THRESHOLD_HP_DIR_PATTERN = re.compile(
    r"^(?:k(?P<k>\d+)_)?"
    r"tq(?P<tq>[0-9._]+)"
    r"_lsetau(?P<lsetau>[0-9._]+)"
    r"_ctf(?P<ctf>[0-9._]+)"
    r"_ta(?P<ta>\d+)"
    r"_nds(?P<nds>\d+)"
    r"_(?P<share>share|noshare)"
    r"_(?P<skip>skip|noskip)"
    r"_(?P<dist>L2|maha)"
    r"(?:_ab(?P<ab>baseline|nogrp|fixthr_intra|fixthr|tmean|globaldb|q05|q99))?$"
)

CACHE_GROUP_DIR_PATTERN = re.compile(
    r"^seqcache_ds(?P<dataset>.+?)_mode"
    r"(?P<mode>.+?)_split"
    r"(?P<split>.+?)_vkey"
    r"(?P<vkey>.+?)_efd"
    r"(?P<efd>.+)$"
)


def _unsanitize_float(s: str) -> str:
    """Convert sanitized float string back (e.g. '0_75' -> '0.75'). Already-dot form unchanged."""
    if "." in s:
        return s
    return re.sub(r"(\d)_(\d)", r"\1.\2", s)


def _extract_seed(name: str):
    m = SEED_DIR_PATTERN.match(name)
    if m is None:
        return None
    return int(m.group(1))


def collect_seed_dirs_by_seed(root_dir: str, seed_glob_prefix: str = "") -> dict:
    """Return {seed: abs_seed_dir} for every *_seed_<n> child of root_dir."""
    seed_dirs: dict = {}
    for name in sorted(os.listdir(root_dir)):
        path = os.path.join(root_dir, name)
        if not os.path.isdir(path):
            continue
        if SEED_DIR_PATTERN.match(name) is None:
            continue
        if seed_glob_prefix and not name.startswith(seed_glob_prefix):
            continue
        seed = _extract_seed(name)
        if seed is None:
            continue
        if seed in seed_dirs:
            prev = seed_dirs[seed]
            if os.path.getmtime(path) > os.path.getmtime(prev):
                seed_dirs[seed] = path
        else:
            seed_dirs[seed] = path
    return seed_dirs


def find_train_tb_dir(seed_dir: str):
    """Find latest <seed_dir>/<timestamp>/logs/tb. Returns None if not found."""
    candidates = []
    for child in os.listdir(seed_dir):
        child_path = os.path.join(seed_dir, child)
        if not os.path.isdir(child_path):
            continue
        tb_path = os.path.join(child_path, "logs", "tb")
        if os.path.isdir(tb_path):
            candidates.append(tb_path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def find_tb_event_dir(seqval_dir: str):
    """Locate the directory holding events.out.tfevents.* for a seqval cache group dir."""
    candidates = [
        os.path.join(seqval_dir, "tb"),
        os.path.join(seqval_dir, "logs", "tb"),
        seqval_dir,
    ]
    for path in candidates:
        if not os.path.isdir(path):
            continue
        try:
            names = os.listdir(path)
        except OSError:
            continue
        if any("tfevents" in n for n in names):
            return path
    return None


def parse_hparam(hp_dir_name: str, cache_group_dir_name: str):
    """
    Parse local_threshold HP dir + cache group dir to (hp_key, hp_dict).
    Returns None if either pattern fails.
    """
    hp_match = LOCAL_THRESHOLD_HP_DIR_PATTERN.match(hp_dir_name)
    if hp_match is None:
        return None
    group_match = CACHE_GROUP_DIR_PATTERN.match(cache_group_dir_name)
    if group_match is None:
        return None

    hp = {
        "dataset": group_match.group("dataset"),
        "cache_mode": group_match.group("mode"),
        "cache_split": group_match.group("split"),
        "cache_vkey": group_match.group("vkey"),
        "eval_first_n_demos": group_match.group("efd"),
        "k": hp_match.group("k"),
        "tq": _unsanitize_float(hp_match.group("tq")),
        "lsetau": _unsanitize_float(hp_match.group("lsetau")),
        "ctf": _unsanitize_float(hp_match.group("ctf")),
        "ta": hp_match.group("ta"),
        "nds": hp_match.group("nds"),
        "share": hp_match.group("share"),
        "skip": hp_match.group("skip"),
        "dist": hp_match.group("dist"),
        "ab": hp_match.group("ab"),
    }
    hp_key = (
        "dataset={dataset}|mode={cache_mode}|split={cache_split}|vkey={cache_vkey}|efd={eval_first_n_demos}"
        "|k={k}|tq={tq}|lsetau={lsetau}|ctf={ctf}|ta={ta}|nds={nds}|share={share}|skip={skip}|dist={dist}"
        "|ab={ab}"
    ).format(**hp)
    return hp_key, hp


def iter_seqval_eval_dirs(eval_seed_dir: str, *, verbose: bool = False) -> Iterator[tuple]:
    """
    Yield (group_path, hp_dir_name, cache_group_dir_name) for every valid
    HP dir x cache-group dir pair under one seed's eval root.
    """
    for hp_dir in sorted(os.listdir(eval_seed_dir)):
        hp_path = os.path.join(eval_seed_dir, hp_dir)
        if not os.path.isdir(hp_path):
            continue
        if LOCAL_THRESHOLD_HP_DIR_PATTERN.match(hp_dir) is None:
            if verbose:
                print(f"[layouts] skip non-HP dir: {hp_dir}")
            continue
        for group_dir in sorted(os.listdir(hp_path)):
            group_path = os.path.join(hp_path, group_dir)
            if not os.path.isdir(group_path):
                continue
            if CACHE_GROUP_DIR_PATTERN.match(group_dir) is None:
                if verbose:
                    print(f"[layouts] skip non-group dir under {hp_dir}: {group_dir}")
                continue
            yield group_path, hp_dir, group_dir
