"""Rebuild this run's baseline tables and blank real-world input template.

Run from surval/: .venv-droid/bin/python outputs/droid_baselines_20260913/build_report.py
No scores are recomputed here. User-filled outcome files are never overwritten.
"""

from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent
TASKS = {"pick_up_the_apple_and_place_it_on_the_dish": "apple",
         "put_the_pan_on_the_stove": "pan", "put_the_pet_on_the_shelve": "pet"}
METHODS = {"Loss": -1, "MSE_mean_pred": -1, "Off_Manifold_Norm": -1,
           "Off_Manifold_Norm_DINO": -1, "surval_score": 1}
EXPECTED_EPOCHS = list(range(5, 51, 5))


def write_csv(name, rows, *, preserve_existing=False):
    path = ROOT / name
    if preserve_existing and path.exists():
        return
    with path.open("x" if preserve_existing else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    payload = json.loads((ROOT / "checkpoint_metrics.json").read_text())
    original = payload["results"]
    assert len(original) == 120, "Expected all3 tasks x4 splits x10 checkpoints"
    rows, groups, physical_checkpoints = [], defaultdict(list), {}
    for source in original:
        task = TASKS[source["dataset"].split("/")[0]]
        split = source["dataset"].split("/")[-1]
        assert split in ("val5", "val10", "val20", "val30")
        assert all(source[m] is not None and math.isfinite(source[m]) for m in METHODS)
        assert source["neighbor_representation"] == "shared_dino"
        assert source["num_samples"] == 8 and source["ta"] == 8
        assert source["dino_omn"]["rows_without_neighbors"] == 0
        assert source["dino_omn"]["min_neighbors"] == source["dino_omn"]["max_neighbors"] == 5
        p = source["provenance"]
        assert p["standard_validation"] and p["max_rows"] is None and p["max_demos"] is None
        n, batch = source["num_rows"], p["batch_size"]
        batches = min(p["valid_num_steps"], math.ceil(n / batch))
        legacy_rows = min(n, batches * batch)
        row = {"task": task, "val_split": split, "epoch": source["epoch"],
               **{method: source[method] for method in METHODS},
               "num_rows": n, "num_demos": source["num_demos"],
               "legacy_validation_rows": legacy_rows, "legacy_validation_batches": batches,
               "legacy_validation_row_fraction": legacy_rows / n,
               "MSE_DINO_OMN_rows": n, "num_cache_samples": 8, "action_horizon": 8,
               "run": source["run"], "run_timestamp": source["run_timestamp"],
               "dataset": source["dataset"], "checkpoint": source["checkpoint"],
               "cache_file": source["cache_file"], "shared_threshold_dir": source["state_db_dir"]}
        rows.append(row)
        groups[(task, int(split[3:]))].append(row)
        key = (task, source["epoch"])
        entry = {"task": task, "epoch": source["epoch"], "successes": "", "trials": "",
                 "success_rate": "", "run": source["run"], "run_timestamp": source["run_timestamp"],
                 "checkpoint": source["checkpoint"]}
        assert key not in physical_checkpoints or physical_checkpoints[key] == entry
        physical_checkpoints[key] = entry
    assert set(groups) == {(t, v) for t in TASKS.values() for v in (5, 10, 20, 30)}
    rows.sort(key=lambda r: (r["task"], int(r["val_split"][3:]), r["epoch"]))
    summary, selections, coverage = [], [], []
    for (task, val_demos), group in sorted(groups.items()):
        group.sort(key=lambda r: r["epoch"])
        assert [r["epoch"] for r in group] == EXPECTED_EPOCHS
        assert len({r["shared_threshold_dir"] for r in group}) == 1
        assert len({r["num_rows"] for r in group}) == 1
        selection = {"task": task, "val_split": f"val{val_demos}"}
        for method, direction in METHODS.items():
            values = [r[method] for r in group]
            best = min(group, key=lambda r: (-direction * r[method], r["epoch"]))
            summary.append({"task": task, "val_split": f"val{val_demos}", "method": method,
                            "higher_is_better": direction > 0, "num_checkpoints": 10,
                            "best_epoch": best["epoch"], "best_value": best[method],
                            "epoch50_value": group[-1][method], "min": min(values), "max": max(values),
                            "median_across_checkpoints": statistics.median(values),
                            "mean_across_checkpoints": statistics.mean(values)})
            selection[method + "_selected_epoch"] = best["epoch"]
        selections.append(selection)
        first = group[0]
        coverage.append({"task": task, "val_split": first["val_split"], "num_checkpoints": 10,
                         "num_demos": first["num_demos"], "num_rows": first["num_rows"],
                         "legacy_validation_rows": first["legacy_validation_rows"],
                         "legacy_validation_batches": first["legacy_validation_batches"],
                         "legacy_validation_row_fraction": first["legacy_validation_row_fraction"],
                         "MSE_DINO_OMN_rows": first["num_rows"]})
    assert len(physical_checkpoints) == 30
    write_csv("checkpoint_scores.csv", rows)
    write_csv("proxy_summary.csv", summary)
    write_csv("selected_epochs.csv", selections)
    write_csv("coverage.csv", coverage)
    write_csv("real_world_outcomes_template.csv", [physical_checkpoints[k] for k in sorted(physical_checkpoints)],
              preserve_existing=True)
    for task in sorted(TASKS.values()):
        write_csv(f"{task}_checkpoint_scores.csv", [r for r in rows if r["task"] == task])

    # Validate the complete source caches against the pre-run manifest.
    cache_paths = sorted(Path("outputs/droid_cache_val5_10_20_30_20260910/conditions").rglob("seqcache_epoch_*.hdf5"))
    assert len(cache_paths) == 120
    digest = hashlib.sha256()
    for path in cache_paths:
        file_hash = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                file_hash.update(block)
        digest.update(f"{file_hash.hexdigest()}  {path}\n".encode())
    assert digest.hexdigest() == "64e8e49941e2e7e1aa5671b12b01c31300c84914e1ef71d8ff27e1cf9c1639bc"
    validation = {"status": "complete", "tasks": 3, "val_splits_per_task": 4,
                  "checkpoint_rows": 120, "physical_checkpoints": 30, "finite_proxy_values": 600,
                  "dino_db_count": 12, "dino_neighbors_per_row": 5,
                  "source_cache_manifest_sha256": digest.hexdigest(), "source_caches_unchanged": True,
                  "outcomes_provided": False, "settings": payload["settings"]}
    (ROOT / "verification.json").write_text(json.dumps(validation, indent=2))

    lines = ["# DROID 세 task: checkpoint baseline 결과", "",
             "2026-09-13. apple·pan·pet × val5/10/20/30 × epoch5,10,…,50 = **120행**.",
             "실제 서로 다른 policy checkpoint는30개이며, 각 checkpoint를4개 val split에서 평가했습니다.", "",
             "## 결과 파일", "",
             "- `checkpoint_scores.csv`: 읽기 쉬운 전체 checkpoint 값 + coverage + 원본 join key.",
             "- `apple/pan/pet_checkpoint_scores.csv`: task별40행.",
             "- `proxy_summary.csv`: task/split/method별 최솟값·최댓값·중앙값·선택 epoch 및 epoch50 값.",
             "- `selected_epochs.csv`: proxy별 선택 checkpoint. 동률이면 가장 이른 epoch.",
             "- `checkpoint_metrics.csv` / `.json`: 기존 surval 집계 API의 입력, provenance·episode 결과 포함.",
             "- `coverage.csv`: 각 지표의 실제 입력 행 수.",
             "- `real_world_outcomes_template.csv`: task·checkpoint별30행의 빈 실측 입력 양식.", "",
             "## Baseline 정의", "",
             "| 열 | 정의 | 평가 범위 | 방향 |",
             "| --- | --- | --- | --- |",
             "| Loss | diffusion noise-prediction MSE; cache 생성 시 저장된 validation loss | 최대50 batch, batch32 | 낮을수록 좋음 |",
             "| MSE_mean_pred | 8개 예측 sample을 먼저 평균한 뒤 GT와 MSE | 전체 val 행 ×8 offset ×10 action dim | 낮을수록 좋음 |",
             "| Off_Manifold_Norm | 기존 policy-encoder/batch-local OMN | 최대50 batch; 별도 policy prediction | 낮을수록 좋음 |",
             "| Off_Manifold_Norm_DINO | 전체 val 공통 DINO DB kNN(k5), 같은 chunk offset의 expert action 투영 잔차 | 전체 val 행 ×8 offset ×8 sample | 낮을수록 좋음 |",
             "| surval_score | 같은 DINO DB의 local group scale을 사용하는 현재 SURVAL 공식 | 모든 sample, 기본 Ta8 stride | 높을수록 좋음 |", "",
             "MSE와 두 OMN은 checkpoint-normalized10차원 action(gripper 포함) 기준입니다. SURVAL은 physical action의",
             "pos/rot6d block이며 gripper 제외. DINOv2 ViT-B/14, 두 exterior view, history2, image-only3072D.",
             "DINO 이웃은 self 및 같은 demo의 ±5 timestep을 제외하고 찾았습니다. Threshold q0.95/k50/min10;",
             "SURVAL chunk top fraction0.5, LSE tau1.0, cumulative product. 실제 결과로 hyperparameter를 조정하지 않았습니다.", "",
             "## Baseline별 선택 epoch", "",
             "**실제 로봇 성능이 가장 좋다는 뜻이 아닙니다.** 아래는 각 proxy 값만으로 선택한 epoch입니다.", "",
             "| Task | Split | Loss ↓ | MSE ↓ | Policy OMN ↓ | DINO OMN ↓ | SURVAL ↑ |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in selections:
        lines.append("| " + " | ".join([row["task"], row["val_split"],
                                       *[str(row[m + "_selected_epoch"]) for m in METHODS]]) + " |")
    lines += ["", "## Coverage 및 논문 해석 주의", "",
              "| Task | Split | 전체 val 행 | Loss / 기존 OMN 행 | 비율 |",
              "| --- | --- | ---: | ---: | ---: |"]
    for row in coverage:
        lines.append(f"| {row['task']} | {row['val_split']} | {row['num_rows']} | "
                     f"{row['legacy_validation_rows']} | {100 * row['legacy_validation_row_fraction']:.1f}% |")
    lines += ["", "- Loss·기존 policy OMN은 저장된 값을 추출했습니다. val20/30에서는 canonical 순서의 첫1600행까지만 포함됩니다.",
              "  기존 run_epoch는 batch 평균을 다시 동일 가중 평균하므로 마지막 작은 batch도 한 batch의 가중치를 가집니다.",
              "  논문에서 이 값을 전체-val Loss/OMN으로 표기하면 안 됩니다. 동일 coverage가 필요하면 별도 재평가가 필요합니다.",
              "- MSE와 DINO OMN은 전체 val 기준입니다. Policy OMN과 DINO OMN은 encoder뿐 아니라 이웃 범위·시간 제외·표본 수도 다릅니다.",
              "- Reference/query는 각 val split 자체입니다. val 크기를 바꾸면 평가 표본과 reference manifold가 함께 달라집니다.",
              "- Task마다 학습 run은1개입니다. checkpoint나 중첩 val split을 독립 seed로 취급한 표준오차/CI는 만들지 않았습니다.",
              "- checkpoint 간 평균/중앙값은 proxy 값의 기술통계이지 real-world 성능 또는 proxy 신뢰도의 측정값이 아닙니다.", "",
              "## Real-world 결과 입력", "",
              "`real_world_outcomes_template.csv`에서 `successes`, `trials`를 채워 주세요. 또는 `success_rate`에0–1 비율을 넣을 수 있습니다.",
              "가능하면 성공 횟수와 시행 횟수를 함께 보관해 주세요. 미측정 checkpoint는 빈칸으로 남기고,0으로 채우지 않습니다.",
              "한 task/epoch의 결과는4개 val split의 동일 checkpoint 행에 공유합니다. 실측을4번 반복 입력할 필요가 없습니다.",
              "`run`, `run_timestamp`, `checkpoint` 열은 원본 키이며 수정하지 않습니다. 서로 다른 로봇 평가 조건은 합치지 말고 별도로 알려 주세요.", "",
              "실측을 받으면 기존 aggregator의 `(run,run_timestamp,dataset,epoch,outcome)` 입력으로 확장하여",
              "Spearman/Kendall, 선택 checkpoint의 성공률, normalized regret, MMRV, hit@1/3/5를 계산할 수 있습니다.",
              "같은 task/split 안에서 동일하게 실측된 checkpoint 집합으로 방법들을 비교하고, task별 결과를 먼저 보고합니다.",
              "현재는 실측이 없으므로 correlation·regret·selection success 표를 생성하지 않았습니다.", "",
              "검증:600 proxy 값 유한,12개 DINO DB,각 행 k5 이웃,120개 원본 cache SHA256 무변경. `verification.json` 참조.", ""]
    (ROOT / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
