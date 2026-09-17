"""Build one shared full-validation DINO DB per DROID dataset, without policy inference."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surval.droid import load_policy_cache, _source_signature
from surval.droid_dino import (DINO_REVISION, align_dino_cache, dataset_db_path, encode_droid_rows,
                               load_shared_dino_db, save_shared_dino_db)
from surval.local_threshold.config import LocalThresholdConfig
from surval.sequential_validate import _discover_cache_groups


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--loader-parallelism", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-cache", default=str(Path(__file__).resolve().parents[1] / ".venv-droid/model-cache/torch"))
    args = parser.parse_args(argv)
    if min(args.batch_size, args.loader_parallelism) < 1:
        parser.error("batch-size and loader-parallelism must be positive")
    groups = _discover_cache_groups(args.cache_dir, mode="epoch")
    if not groups:
        parser.error("No epoch policy caches found")
    encoder, weights_hash = None, None
    seen = set()
    for group in groups:
        cache = load_policy_cache(group["cache_files"][0])
        p = cache["provenance"]
        dataset = p["dataset_name"]
        if dataset in seen:
            continue
        seen.add(dataset)
        if (p.get("producer") != "robomimic_diffusion_policy" or p.get("split") != "val"
                or p.get("max_demos") is not None or p.get("max_rows") is not None):
            raise ValueError("Builder requires an uncapped droid_policy_learning validation cache")
        source_dir = dataset_db_path(args.data_dir, dataset)
        signatures = [_source_signature(path) for path in sorted(source_dir.rglob("*"))
                      if path.is_file() and (path.name in ("dataset_info.json", "features.json")
                                             or "tfrecord" in path.name)]
        if not signatures or signatures != p["dataset_files"]:
            raise ValueError("Dataset path/files changed since policy caching")
        target = dataset_db_path(args.output_dir, dataset)
        if target.exists():
            db, chunks, metadata = load_shared_dino_db(target)
            align_dino_cache(db, chunks, metadata, cache)
            if metadata.get("encoder_revision") != DINO_REVISION:
                raise ValueError("Existing DB uses a different DINO revision; choose another output root")
            print(f"[reuse] {target}", flush=True)
            continue

        # Import the existing finite RLDS producer only for image loading.
        os.environ.setdefault("XFORMERS_DISABLED", "1")
        from robomimic.scripts.sequential_cache_checkpoints import _build_rlds_evalset
        from robomimic.utils.file_utils import config_from_checkpoint
        from surval.local_threshold.encoder import FrozenImageEncoder
        import torch

        torch.hub.set_dir(args.model_cache)
        config_path = Path(cache["checkpoint"]).parent.parent / "config.json"
        config, _ = config_from_checkpoint(algo_name="diffusion_policy",
                                          ckpt_dict={"config": config_path.read_text()})
        with config.values_unlocked():
            config.observation.image_dim = [224, 224]
        cfg = LocalThresholdConfig.for_droid_action_space(
            cache["droid_action_space"], image_views=tuple(config.observation.modalities.obs.rgb),
            use_proprio=False, state_window_size=p["observation_horizon"],
            dataset_name=dataset, data_dir=str(Path(args.data_dir).resolve()), actions_from_cache=True,
            action_chunk_size=cache["actions"].shape[1], encoder_batch_size=args.batch_size,
            encoder_device=args.device,
        )
        parts = Path(dataset).parts
        loader_root = Path(args.data_dir).joinpath(*parts[:-2]) if len(parts) > 2 else Path(args.data_dir)
        loader_name = "/".join(parts[-2:])
        print(f"[load RLDS] {dataset}", flush=True)
        evalset = _build_rlds_evalset(config, str(loader_root), loader_name,
                                     cache["action_scale"], cache["action_offset"],
                                     loader_parallelism=args.loader_parallelism)
        if encoder is None:
            encoder = FrozenImageEncoder(cfg, revision=DINO_REVISION)
            digest = hashlib.sha256()
            for key, tensor in sorted(encoder.model.state_dict().items()):
                digest.update(key.encode())
                digest.update(tensor.detach().cpu().numpy().tobytes())
            weights_hash = digest.hexdigest()
        print(f"[encode DINO] {dataset} rows={len(evalset['dataset'])}", flush=True)
        db = encode_droid_rows(evalset["dataset"], cache, cfg, encoder)
        metadata = {"dataset_name": dataset, "dataset_files": p["dataset_files"], "split": "val",
                    "observation_horizon": p["observation_horizon"],
                    "action_start_offset": cache["action_start_offset"],
                    "droid_action_space": cache["droid_action_space"],
                    "encoder_revision": DINO_REVISION, "weights_sha256": weights_hash,
                    "image_preprocess": "RLDS resize 224, ImageNet normalization, float32 CLS",
                    "feature_source": "shared_dinov2_vitb14_images_only",
                    "row_coverage": evalset["coverage"], "policy_config": _source_signature(config_path)}
        save_shared_dino_db(target, db, cache["actions"], metadata)
        print(f"[complete] {target} embeddings={db.embeddings.shape}", flush=True)


if __name__ == "__main__":
    main()
