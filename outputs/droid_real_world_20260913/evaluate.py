"""Compute the three requested metrics using existing SURVAL aggregation.

From surval/: .venv-droid/bin/python outputs/droid_real_world_20260913/evaluate.py
Input counts and the baseline run are immutable; generated reports may be rebuilt.
"""

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / "src"))
from surval.tb_aggregate.droid import aggregate_droid_metrics, ROW_KEY


BASELINE = REPO / "outputs/droid_baselines_20260913/checkpoint_metrics.csv"
TASKS = {"pick_up_the_apple_and_place_it_on_the_dish": "apple",
         "put_the_pan_on_the_stove": "pan", "put_the_pet_on_the_shelve": "pet"}
METHODS = {"Loss": -1, "MSE_mean_pred": -1, "Off_Manifold_Norm": -1,
           "Off_Manifold_Norm_DINO": -1, "surval_score": 1}
LABELS = {"Loss": "Loss", "MSE_mean_pred": "MSE", "Off_Manifold_Norm": "Policy OMN",
          "Off_Manifold_Norm_DINO": "DINO OMN", "surval_score": "SURVAL (DINO)"}


def write_csv(filename, rows):
    with (ROOT / filename).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    data = json.loads((ROOT / "success_counts.json").read_text())
    epochs, trials = data["epoch_order"], data["trials_per_checkpoint"]
    counts = data["successes"]
    assert epochs == list(range(5, 51, 5)) and trials == 30
    assert set(counts) == set(TASKS.values())
    assert all(len(v) == len(epochs) and all(type(x) is int and 0 <= x <= trials for x in v)
               for v in counts.values())
    with BASELINE.open() as f:
        scores = list(csv.DictReader(f))
    assert len(scores) == 120
    baseline_hash = hashlib.sha256(BASELINE.read_bytes()).hexdigest()
    lookup = {}
    grouped = defaultdict(list)
    expanded = []
    for score in scores:
        task = TASKS[score["dataset"].split("/")[0]]
        epoch = int(score["epoch"])
        successes = counts[task][epochs.index(epoch)]
        key = (task, epoch)
        outcome = {"task": task, "epoch": epoch, "successes": successes, "trials": trials,
                   "success_rate": successes / trials, "run": score["run"],
                   "run_timestamp": score["run_timestamp"], "checkpoint": score["checkpoint"]}
        assert key not in lookup or lookup[key] == outcome
        lookup[key] = outcome
        expanded.append({**{key: score[key] for key in ROW_KEY}, "outcome": successes / trials,
                         "task": task, "successes": successes, "trials": trials})
        grouped[(task, score["dataset"])].append(score)
    assert len(lookup) == 30 and len(grouped) == 12
    write_csv("real_world_outcomes.csv", [lookup[k] for k in sorted(lookup)])
    write_csv("outcomes_by_split.csv", expanded)
    report = aggregate_droid_metrics(BASELINE, ROOT / "outcomes_by_split.csv", ROOT / "aggregate.json")
    assert not report["skipped"] and report["num_unlabelled_checkpoints"] == 0
    assert len(report["per_group"]) == 60
    rows = []
    for result in report["per_group"]:
        task = TASKS[result["dataset"].split("/")[0]]
        split = result["dataset"].split("/")[-1]
        method = result["method"]
        group = sorted(grouped[(task, result["dataset"])], key=lambda r: int(r["epoch"]))
        assert [int(r["epoch"]) for r in group] == epochs
        actual = np.asarray(counts[task], dtype=np.float64) / trials
        proxy = METHODS[method] * np.array([float(r[method]) for r in group])
        selected = int(np.argmax(proxy))
        gap = float(actual.max() - actual[selected])
        # Independent checks: scipy average-rank Spearman, scalar-loop MMRV,
        # direct success-count regret and selection. Keep formulas unchanged.
        expected_mmrv = sum(max(abs(actual[i] - actual[j])
                                if (proxy[i] < proxy[j]) != (actual[i] < actual[j]) else 0.0
                                for j in range(len(actual))) for i in range(len(actual))) / len(actual)
        expected_regret = (max(counts[task]) - counts[task][selected]) / (max(counts[task]) - min(counts[task]))
        np.testing.assert_allclose(result["metrics"]["spearman"], spearmanr(actual, proxy).statistic, atol=1e-12)
        np.testing.assert_allclose(result["metrics"]["mmrv"], expected_mmrv, atol=1e-12)
        np.testing.assert_allclose(result["metrics"]["nregret"], expected_regret, atol=1e-10)
        assert result["selected_epoch"] == epochs[selected]
        assert result["num_checkpoints"] == 10
        rows.append({"task": task, "val_split": split, "method": method,
                     "nregret": result["metrics"]["nregret"], "mmrv": result["metrics"]["mmrv"],
                     "spearman": result["metrics"]["spearman"],
                     "regret_pp": gap * 100, "mmrv_pp": result["metrics"]["mmrv"] * 100,
                     "selected_epoch": epochs[selected], "selected_successes": counts[task][selected],
                     "selected_success_rate": float(actual[selected]),
                     "oracle_epochs": ";".join(str(epochs[i]) for i in np.flatnonzero(actual == actual.max())),
                     "oracle_successes": max(counts[task]), "oracle_success_rate": float(actual.max()),
                     "num_checkpoints": 10, "trials_per_checkpoint": trials,
                     "run": result["run"], "run_timestamp": result["run_timestamp"], "dataset": result["dataset"]})
    rows.sort(key=lambda r: (r["task"], int(r["val_split"][3:]), list(METHODS).index(r["method"])))
    write_csv("primary_metrics.csv", rows)
    for task in sorted(TASKS.values()):
        write_csv(f"{task}_metrics.csv", [r for r in rows if r["task"] == task])
    macro = []
    for split in ("val5", "val10", "val20", "val30"):
        for method in METHODS:
            selected = [r for r in rows if r["val_split"] == split and r["method"] == method]
            assert len(selected) == 3
            macro.append({"val_split": split, "method": method, "num_tasks": 3,
                          **{m: float(np.mean([r[m] for r in selected]))
                             for m in ("nregret", "mmrv", "spearman", "regret_pp", "mmrv_pp")}})
    write_csv("task_macro_metrics.csv", macro)
    lines = ["# Real-world 평가: regret / MMRV / Spearman", "",
             "입력 순서: epoch **5,10,…,50**, checkpoint마다 **30회**. 총30개 checkpoint,900 trials.",
             "같은 실측 결과를 val5/10/20/30의 동일 checkpoint에 연결했습니다. val split을 독립 실측으로 세지 않습니다.", "",
             "## 정의", "",
             "- **nRegret ↓** = (최고 성공률 − proxy가 선택한 checkpoint의 성공률) / (최고 − 최저 성공률 +1e−12).",
             "  기존 surval의 `nregret` 정의이며, 정규화 전 regret은 `regret_pp` 열에 percentage points로 저장했습니다.",
             "- **MMRV ↓** = 각 checkpoint에 대해 proxy 순위가 실제 성공률 순위와 어긋나는 상대의 최대 성공률 차이를 구한 뒤 평균.",
             "  기존 코드의 strict-< 비교를 그대로 사용합니다. 성공률0–1 단위이며,0.10은10pp입니다. `mmrv_pp`도 저장했습니다.",
             "- **Spearman ρ ↑** = 실제 성공률과 방향을 맞춘 proxy의 순위 상관. Loss/MSE/OMN은 부호 반전, SURVAL은 원래 부호.",
             "  동률은 평균 순위를 사용하며, 음수면 실제 성공률과 역방향입니다.",
             "- Proxy 동률이면 가장 이른 epoch를 선택합니다. 실제 최고 성공률의 동률은 모두 oracle epoch로 기록했습니다.", "",
             "## 실제 성공 횟수", "",
             "| Task | " + " | ".join(str(e) for e in epochs) + " |",
             "| --- | " + " | ".join("---:" for _ in epochs) + " |"]
    for task in ("apple", "pan", "pet"):
        lines.append("| " + task + " | " + " | ".join(str(x) for x in counts[task]) + " |")
    lines += ["", "각 셀은 **nRegret / MMRV / Spearman** 순서이며 소수 셋째 자리까지 표시했습니다.", ""]
    by_key = {(r["task"], r["val_split"], r["method"]): r for r in rows}
    for task in ("apple", "pan", "pet"):
        best = max(counts[task])
        best_epochs = [epochs[i] for i, x in enumerate(counts[task]) if x == best]
        lines += [f"## {task}", "", f"최고 실측 성공률: {best}/30 = {best / 30:.1%}; epoch {best_epochs}.", "",
                  "| Proxy | val5 | val10 | val20 | val30 |", "| --- | --- | --- | --- | --- |"]
        for method in METHODS:
            cells = [LABELS[method]]
            for split in ("val5", "val10", "val20", "val30"):
                r = by_key[(task, split, method)]
                cells.append(f"{r['nregret']:.3f} / {r['mmrv']:.3f} / {r['spearman']:+.3f}")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## 파일과 해석 범위", "",
              "- `primary_metrics.csv`: 전체60개 task/split/method 조합, 세 메인 metric 및 선택/oracle checkpoint, raw regret(pp).",
              "- `task_macro_metrics.csv`: 동일 split에서 task별 metric을 동일 가중 평균. checkpoint를 섞어서 상관을 계산하지 않습니다.",
              "- `real_world_outcomes.csv`: 입력한 성공 횟수와30 trials를 보존한30행 파일.",
              "- `outcomes_by_split.csv`: 기존 집계 API용120행. `aggregate.json`은 기존 라이브러리의 전체 집계 결과.",
              "- 모든60개 조합은 동일한10개 epoch를 사용했습니다. 누락/제외 checkpoint가 없습니다.",
              "- 기존 Loss·policy OMN은 val20/30에서 첫1600행까지만 평가된 저장값입니다. MSE·DINO OMN은 전체 val입니다.",
              "- SURVAL과 DINO OMN의 reference DB는 각 val split 자체입니다. split 간 비교에는 reference 크기 변화도 포함됩니다.",
              "- Task별 학습 run이1개이며, 현재 표는 checkpoint당30회에서 관측된 성공률의 point estimate입니다.",
              "  seed-level CI나 유의성 검정을 주장하지 않습니다. 동일 실측을 재사용한4개 val split도 독립 실험이 아닙니다.",
              "- 실측을 보고 checkpoint selection 규칙이나 SURVAL hyperparameter를 변경하지 않았습니다.", ""]
    (ROOT / "REPORT.md").write_text("\n".join(lines))
    assert hashlib.sha256(BASELINE.read_bytes()).hexdigest() == baseline_hash
    verification = {"status": "passed", "num_real_world_checkpoints": 30, "num_trials": 900,
                    "num_expanded_rows": 120, "num_method_groups": 60, "checkpoints_per_group": 10,
                    "spearman_check": "scipy.stats.spearmanr", "mmrv_check": "independent scalar pair loop",
                    "regret_check": "independent integer success-count gap/range", "missing_outcomes": 0,
                    "baseline_csv_sha256_unchanged": baseline_hash,
                    "regret_definition": "surval normalized regret; raw regret also reported in percentage points",
                    "command": ".venv-droid/bin/python outputs/droid_real_world_20260913/evaluate.py"}
    (ROOT / "verification.json").write_text(json.dumps(verification, indent=2))
    print(json.dumps(verification, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
