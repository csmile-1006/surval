# SurVAL 결과를 새 서버에서 복원하기

2026-09-17 백업 기준. Linux x86_64와 기존 `/workspace` 경로를 사용한다.
원본 서버의 인증정보나 가상환경을 복사하지 않고, 코드·결과·폰트를 복원한다.

## 1. 무엇까지 복원되는가

| 작업 | 필요한 자료 |
| --- | --- |
| 기존 PDF/PNG/CSV 열기 | 기본 결과 백업. 최신 그림은 보완 백업도 적용 |
| 저장된 점수의 Spearman/MMRV/NRegret 검증, 그림 재생성 | 위 자료 + 고정 코드 + CPU 그림 환경 + 폰트 |
| 저장된 prediction cache로 SurVAL/OMN 재계산 | 기본 백업의 HDF5/NPZ/DB + 해당 분석 코드·설정 |
| checkpoint에서 prediction cache 새로 생성 | 별도 보관한 checkpoint와 원본 RLDS 데이터 + 추론 환경 + Octo patch |

기본 백업은 **5,568개 파일**, 압축 약 **20GB**, 원본 약 **20.7GB**다.
checkpoint 가중치·원본 데이터·가상환경·인증정보는 포함하지 않는다.
데이터와 checkpoint는 사용자가 별도로 업로드한 것으로, 여기서는 그 원격 경로와
완전성을 검증하지 않았다. 아래 그림 재생성에는 거대한 가중치/데이터가 필요 없다.

보완 백업은 기본 백업 이후 변경된 결과, 최신 합본 ZIP, 환경 inventory,
폰트 4개와 OFL 라이선스, 학습 config 3개, split manifest 3개, Octo patch를 포함한다.
Octo는 커밋하거나 push하지 않았다. 다른 세션의 별도 `outputs/backups/`는 포함하지 않는다.

## 2. 다운로드 및 무결성 검증

필요 도구: `git`, `gcloud`, `zstd`/`unzstd`, `tar`, `jq`, `uv`, Python 3.11.
GitHub의 private repo 읽기 권한과 GCS object 읽기 권한을 새 서버에서 인증한다.
기존 서버의 credential 파일은 백업하지 않는다. 결과 복원만 하더라도 다운로드·압축 해제·
작업 사본을 고려해 약 **80GB 이상 여유 공간**을 권장한다. 가중치와 데이터 공간은 별도다.

아래는 **새 서버의 빈 작업 경로**에서 실행한다. 기존 작업이 있으면 먼저 별도로 보존하고
대상 경로를 검토한다. 명령은 의도한 복원 파일을 덮어쓸 수 있으므로 현재 작업 서버에서 실행하지 않는다.

```bash
set -euo pipefail
mkdir -p /workspace/restore_downloads/base /workspace/restore_downloads/supplement
BACKUP_GCS=gs://riselab_robot_data/surval_iclr_droid/backups/20260917

cd /workspace/restore_downloads/base
gcloud storage cp "$BACKUP_GCS/surval_artifacts_20260917T111946Z.tar.zst" .
gcloud storage cp "$BACKUP_GCS/SHA256SUMS" "$BACKUP_GCS/_UPLOAD_COMPLETE.json" .
sha256sum -c SHA256SUMS
test "$(jq -r .status _UPLOAD_COMPLETE.json)" = uploaded_verified

cd /workspace/restore_downloads/supplement
gcloud storage cp "$BACKUP_GCS/repro_20260917/surval_repro_20260917.tar.zst" .
gcloud storage cp "$BACKUP_GCS/repro_20260917/SHA256SUMS" \
  "$BACKUP_GCS/repro_20260917/_UPLOAD_COMPLETE.json" \
  "$BACKUP_GCS/repro_20260917/README.md" .
sha256sum -c SHA256SUMS
test "$(jq -r .status _UPLOAD_COMPLETE.json)" = uploaded_verified

mkdir unpacked
tar --use-compress-program=unzstd -xf surval_repro_20260917.tar.zst -C unpacked
cd unpacked
sha256sum -c PAYLOAD_SHA256SUMS
```

기본 archive SHA256:
`53d7a705b8ac0fca3bbd2952f5ee10dcde04a218c912ee3debf520801b061de1`.
기본 `manifest.json`의 `upload_status: not_uploaded`는 빌드 시점 기록이다.
최종 업로드 완료 여부는 `_UPLOAD_COMPLETE.json`을 따른다. 원래 백업은 수정하지 않는다.

## 3. 정확한 코드 버전과 결과 배치

보완 백업의 `metadata/code_refs.json`이 최종 코드 SHA의 기준이다.
움직이는 `icra` 브랜치 최신값 대신 **이 파일에 기록된 SHA**를 checkout한다.

```bash
RESTORE_SUPPLEMENT=/workspace/restore_downloads/supplement/unpacked
for repo in surval droid_policy_learning surval_openpi; do
  test ! -e "/workspace/$repo"
  repo_url=$(jq -r --arg repo "$repo" '.[$repo].remote' "$RESTORE_SUPPLEMENT/metadata/code_refs.json")
  repo_sha=$(jq -r --arg repo "$repo" '.[$repo].commit' "$RESTORE_SUPPLEMENT/metadata/code_refs.json")
  git clone --no-checkout "$repo_url" "/workspace/$repo"
  git -C "/workspace/$repo" checkout --detach "$repo_sha"
done

cd /workspace/restore_downloads/base
mkdir unpacked
tar --use-compress-program=unzstd -xf surval_artifacts_20260917T111946Z.tar.zst -C unpacked
# Git에서 가져온 소스는 보존하고, 누락된 기존 결과 파일을 채운다.
cp -an unpacked/. /workspace/
# 기본 백업 위에 검증된 최신 결과/작은 메타데이터를 적용한다.
cp -a "$RESTORE_SUPPLEMENT/workspace/." /workspace/
```

결과 위치:

- `/workspace/surval_results_combined/{droid,openpi}/`: 자동 축 범위 그림·점수·메트릭.
- `/workspace/surval_results_combined/comine/{droid,openpi}/`: 공통 축 범위 aggregate/per-task 그림.
- `/workspace/surval_results_combined.zip`: 최신 전체 합본.
- `/workspace/surval/outputs/`, `/workspace/surval_openpi/outputs/`: 캐시·HP 탐색·검증·분석 원본.

`comine`은 기존 폴더명을 그대로 보존했다. 경로를 임의로 바꾸면 저장된 절대경로와
provenance가 맞지 않을 수 있다. 코드 상태는 각 repo에서 `git status --short`로 확인한다.

## 4. 폰트와 CPU 그림 환경

```bash
RESTORE_SUPPLEMENT=/workspace/restore_downloads/supplement/unpacked
install -d /root/.local/share/fonts/IosevkaNerdFont
cp -a "$RESTORE_SUPPLEMENT/assets/fonts/IosevkaNerdFont/." \
  /root/.local/share/fonts/IosevkaNerdFont/

uv venv --python 3.11 /workspace/surval_openpi/.venv-plots
uv pip install --python /workspace/surval_openpi/.venv-plots/bin/python \
  -r /workspace/surval_openpi/docs/requirements-surval-plots.txt
export PYTHONPATH=/workspace/surval/src:/workspace/surval/scripts
export CUDA_VISIBLE_DEVICES=''
export MPLCONFIGDIR=/tmp/surval_restore_mpl
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2

cd /workspace/surval_openpi
.venv-plots/bin/python -c 'import surval; print(surval.__file__)'
.venv-plots/bin/python scripts/plot_scatter_axes_test.py
```

`surval.__file__`은 `/workspace/surval/src/surval/` 아래여야 한다.
OpenPI pyproject/uv.lock의 예전 SurVAL pin에 의존하지 않고 복원한 코드를 사용한다.
그림 전용 경로는 `uv sync`나 GPU Torch/JAX/TensorFlow 설치가 필요 없다.
폰트 내부 family는 `Iosevka NF`이며 조용히 다른 폰트로 대체하지 않는다.
루트 사용자가 아닌 서버에서는 폰트 디렉터리 접근 권한과 렌더러의 `FONT_DIR`을 조정해야 한다.

## 5. 원본을 보존하면서 그림 재생성

아래 공통 축 렌더러는 저장된 CSV만 읽고, 각 task의 Spearman/MMRV/NRegret을 다시
계산해 기존 값과 비교한다. 학습·캐시 생성·HP 재튜닝은 하지 않는다.

```bash
cd /workspace/surval_openpi
test ! -e /workspace/reproduced_surval_results
mkdir /workspace/reproduced_surval_results
cp -a /workspace/surval_results_combined/openpi \
  /workspace/surval_results_combined/droid /workspace/reproduced_surval_results/
.venv-plots/bin/python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, '/workspace/surval_openpi/scripts')
import plot_combined_fixed_axes as plot
plot.ROOT = Path('/workspace/reproduced_surval_results')
plot.OUT = plot.ROOT / 'comine'
plot.main()
PY
```

예상: `comine/`에 PDF **70개**, PNG **70개**, 옆에 `comine.zip`.
공통 산점도 축은 x `[-3.5, 3.0]`, y `[-3.0, 5.5]`.
Per-task 산점도만 y축 제목·눈금·눈금 숫자가 없고, grid와 데이터 범위는 유지된다.
Aggregate와 epoch/step 곡선의 y축 표시는 유지된다. PDF의 Iosevka 임베딩,
마커 잘림·주석 겹침·입력 원본 불변성도 렌더러가 검증한다.

자동 축 버전의 원래 생성 명령도 보존되어 있다:

```bash
# OpenPI: 이 명령은 복원된 outputs/surval_figures 아래 기존 그림을 갱신한다.
cd /workspace/surval_openpi
.venv-plots/bin/python scripts/plot_pi05_ta12_results.py --variant task_qk

# DROID: 새 출력 폴더를 지정하며 기존 결과는 보존한다.
cd /workspace/surval
/workspace/surval_openpi/.venv-plots/bin/python scripts/plot_droid_ta8_results.py \
  --ta 3 --output /workspace/reproduced_surval_results/droid_auto
```

원래 선택: DROID epoch5 encoder/Ta3, OpenPI step200 encoder/Ta12,
둘 다 task별 k/q. 원본 HP와 checkpoint 제외 규칙을 변경하지 않는다.
그림의 통계는 task별 계산 후 같은 가중치로 평균한 값이며 pooled 지표가 아니다.
Bootstrap/유의성 검정을 추가하지 않는다. PDF 생성 시각 등으로 파일 바이트는
달라질 수 있으므로 수치·데이터·폰트·표현 규칙으로 재현 여부를 판단한다.

## 6. 체크포인트부터 캐시를 다시 만들 때만 필요한 추가 준비

1. 별도로 업로드한 RLDS 데이터와 가중치를 원래 `/workspace/data` 및 checkpoint
   경로로 복원한다. config JSON만 있다고 실제 가중치가 복원된 것은 아니다.
2. `metadata/environments/droid-effective.json`과 `openpi-effective.json`은 실제
   Python/패키지/VCS/editable 출처 기록이다. 완성된 설치 lockfile로 간주하지 않는다.
   DROID는 Python3.10/NumPy1.26.4/TF2.15/Torch2.0.1+cu118 조합이었다.
   원래 `.venv-droid`는 `/venv/droid_policy`를 상속하므로 디렉터리 복사만으로 복원되지 않는다.
   기반 환경의 NumPy2 상태를 복제하지 않는다. OpenPI 추론 환경은 Python3.11 별도 구성이다.
3. DROID에는 다음 Octo patch가 필요하다. 코드를 commit/push하지 않고 적용한다.

```bash
RESTORE_SUPPLEMENT=/workspace/restore_downloads/supplement/unpacked
test ! -e /workspace/octo
git clone https://github.com/octo-models/octo.git /workspace/octo
git -C /workspace/octo checkout --detach \
  "$(jq -r '.octo.commit' "$RESTORE_SUPPLEMENT/metadata/code_refs.json")"
git -C /workspace/octo apply --check "$RESTORE_SUPPLEMENT/metadata/octo-rlds-row-ids.patch"
git -C /workspace/octo apply "$RESTORE_SUPPLEMENT/metadata/octo-rlds-row-ids.patch"
```

4. DROID 언어 인코더 DistilBERT, DINO 분석 시 DINOv2 가중치와 기록된 revision을
   준비한다. 모델 다운로드 캐시는 결과 archive에 포함하지 않았다.
5. 새 GPU에 기존 CUDA wheel이 맞는지 확인한다. 최신 GPU에 cu118이 동작한다고
   가정하지 않는다. seed/sample 수/batch size/action horizon도 원래 실행값을 사용한다.
6. [DROID.md](DROID.md)의 작은 `--max-demos/--max-rows` smoke부터 실행한다.
   다운로드 후 checkpoint mtime이 달라지면 cache provenance 재사용 검사가 거부될 수 있다.
   검사를 끄지 말고 **새 cache 출력 폴더**를 사용한다.

저장된 prediction cache가 논문 수치 재현의 기준 입력이다. 새 GPU에서 확률적 추론을
다시 수행했을 때 bitwise 동일한 cache까지 보장하는 것은 아니다.

## 7. 검증 범위와 완료 판단

기존 서버에서 독립된 CPU 전용 가상환경에 고정 패키지를 설치하고 그림/메트릭 경로를
검증했다. 이는 원래 학습 환경을 재사용하지 않는 검사지만, 새 서버에서 전체 RLDS→정책
추론을 재실행한 검증은 아니다. 추론 환경의 완전한 새 설치는 별도 작업이다.

복원 완료 기준: 두 archive의 SHA256 통과, payload checksum 통과, 고정 Git SHA 확인,
로컬 SurVAL import 확인, per-task 축 테스트 통과, 재생성된 PDF/PNG 개수·폰트·메트릭 확인.
`_UPLOAD_COMPLETE.json`이 없거나 `uploaded_verified`가 아니면 업로드 완료로 간주하지 않는다.
