"""Built-in action-space configs for ``surval.sequential_validate``.

These are the per-robot block layouts the metric needs, bundled so a cache can be
scored with surval alone (no per-consumer wrapper script). Each value is the
``ACTION_SPACE`` dict that ``sequential_validate.run`` expects.

- ``droid``    — DROID single arm, 8-D ``[joint (7), gripper (1)]``; 7 joint
  blocks (gripper unscored), L2.
- ``gripper``  — 14-D bimanual ``[L_pos,L_rot,grip | R_pos,R_rot,grip]``, L2.
- ``dex``      — 24-D dexterous bimanual (6-D fingers per arm), L2.
- ``humanoid`` — 30-D bimanual with 6-D continuous rotations (geodesic / rot6d).

The block layout (names/slices/types) matches ``build_state_db_from_cache.py``'s
``--block-layout`` so state-conditional thresholds line up with the consumer.
"""

from __future__ import annotations

DROID = {
    "action_dim": 8,
    "block_names": [f"joint_{i}" for i in range(7)],
    "block_slices": {f"joint_{i}": slice(i, i + 1) for i in range(7)},
    "block_dims": {f"joint_{i}": 1 for i in range(7)},
    "arm_pairs": [],
    "scale_groups": [{"blocks": [f"joint_{i}" for i in range(7)], "summary_key": "s_joint"}],
    "summary_scale_fields": [("ActionBlockScale_joint", "s_joint")],
    "block_types": {},
}

GRIPPER = {
    "action_dim": 14,
    "block_names": ["L_pos", "L_rot", "R_pos", "R_rot"],
    "block_slices": {"L_pos": slice(0, 3), "L_rot": slice(3, 6), "R_pos": slice(7, 10), "R_rot": slice(10, 13)},
    "block_dims": {"L_pos": 3, "L_rot": 3, "R_pos": 3, "R_rot": 3},
    "arm_pairs": [("L_pos", "R_pos"), ("L_rot", "R_rot")],
    "scale_groups": [
        {"blocks": ["L_pos", "R_pos"], "summary_key": "s_pos"},
        {"blocks": ["L_rot", "R_rot"], "summary_key": "s_rot"},
    ],
    "summary_scale_fields": [("ActionBlockScale_pos", "s_pos"), ("ActionBlockScale_rot", "s_rot")],
    "block_types": {},
}

DEX = {
    "action_dim": 24,
    "block_names": ["L_pos", "L_rot", "L_finger", "R_pos", "R_rot", "R_finger"],
    "block_slices": {
        "L_pos": slice(0, 3), "L_rot": slice(3, 6), "L_finger": slice(6, 12),
        "R_pos": slice(12, 15), "R_rot": slice(15, 18), "R_finger": slice(18, 24),
    },
    "block_dims": {"L_pos": 3, "L_rot": 3, "L_finger": 6, "R_pos": 3, "R_rot": 3, "R_finger": 6},
    "arm_pairs": [("L_pos", "R_pos"), ("L_rot", "R_rot"), ("L_finger", "R_finger")],
    "scale_groups": [
        {"blocks": ["L_pos", "R_pos"], "summary_key": "s_pos"},
        {"blocks": ["L_rot", "R_rot"], "summary_key": "s_rot"},
        {"blocks": ["L_finger", "R_finger"], "summary_key": "s_finger"},
    ],
    "summary_scale_fields": [
        ("ActionBlockScale_pos", "s_pos"), ("ActionBlockScale_rot", "s_rot"),
        ("ActionBlockScale_finger", "s_finger"),
    ],
    "block_types": {},
}

HUMANOID = {
    "action_dim": 30,
    "block_names": ["R_pos", "R_rot", "L_pos", "L_rot", "R_finger", "L_finger"],
    "block_slices": {
        "R_pos": slice(0, 3), "R_rot": slice(3, 9), "L_pos": slice(9, 12),
        "L_rot": slice(12, 18), "R_finger": slice(18, 24), "L_finger": slice(24, 30),
    },
    "block_dims": {"R_pos": 3, "R_rot": 6, "L_pos": 3, "L_rot": 6, "R_finger": 6, "L_finger": 6},
    "arm_pairs": [("L_pos", "R_pos"), ("L_rot", "R_rot"), ("L_finger", "R_finger")],
    "scale_groups": [
        {"blocks": ["R_pos", "L_pos"], "summary_key": "s_pos"},
        {"blocks": ["R_rot", "L_rot"], "summary_key": "s_rot"},
        {"blocks": ["R_finger", "L_finger"], "summary_key": "s_finger"},
    ],
    "summary_scale_fields": [
        ("ActionBlockScale_pos", "s_pos"), ("ActionBlockScale_rot", "s_rot"),
        ("ActionBlockScale_finger", "s_finger"),
    ],
    "block_types": {"R_rot": "rot6d", "L_rot": "rot6d"},
}

# GR1 humanoid arms+hands (NVIDIA Isaac-GR00T fourier_gr1_arms_only): 26-D
# [left_arm(7), right_arm(7), left_hand(6), right_hand(6)].
GR1 = {
    "action_dim": 26,
    "block_names": ["left_arm", "right_arm", "left_hand", "right_hand"],
    "block_slices": {"left_arm": slice(0, 7), "right_arm": slice(7, 14),
                     "left_hand": slice(14, 20), "right_hand": slice(20, 26)},
    "block_dims": {"left_arm": 7, "right_arm": 7, "left_hand": 6, "right_hand": 6},
    "arm_pairs": [("left_arm", "right_arm"), ("left_hand", "right_hand")],
    "scale_groups": [
        {"blocks": ["left_arm", "right_arm"], "summary_key": "s_arm"},
        {"blocks": ["left_hand", "right_hand"], "summary_key": "s_hand"},
    ],
    "summary_scale_fields": [("ActionBlockScale_arm", "s_arm"), ("ActionBlockScale_hand", "s_hand")],
    "block_types": {},
}

ACTION_SPACES = {"droid": DROID, "gripper": GRIPPER, "dex": DEX, "humanoid": HUMANOID, "gr1": GR1}


def get_action_space(name: str) -> dict:
    """Return the built-in ACTION_SPACE for ``name`` (droid/gripper/dex/humanoid)."""
    try:
        return ACTION_SPACES[name]
    except KeyError:
        raise ValueError(f"Unknown action space {name!r}; choose from {sorted(ACTION_SPACES)}") from None
