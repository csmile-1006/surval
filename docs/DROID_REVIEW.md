# DROID 기준 SURVAL 연계 검증 — 2026-09-10

## 결론과 범위

현재 로컬 DROID diffusion-policy 체크포인트는 **추가 학습 없이** val1/5/10/20/30별 cache → SURVAL → MSE/Loss/OMN → custom success-rate 집계 경로를 사용할 수 있다. 실제 pet epoch-5 체크포인트의 CPU smoke로 추론과 파일 출력까지 검증했다. 다만 이는 4행 및 12행의 제한된 smoke이며, 전체 150개 checkpoint×split 조합을 실행한 연구 결과가 아니다.

보내주신 리뷰는 openpi pi05 경로를 중심으로 한 것이다. 이번 검증은 robomimic/Octo DROID를 기준으로 한다. 검증 중 다른 세션이 공유 surval adapter에 step/physical/joint-action 지원을 추가했으므로 그 변경은 보존했다. openpi의 pin·checkpoint 업로드·학습 경로 자체를 이번 작업에서 수정하거나 재검증한 것은 아니다.

## 데이터와 체크포인트

6개 droid_split_manifest.json을 전부 읽어 다음을 확인했다.

| 태스크 | train demos | val 구성 | 로컬 DROID checkpoints |
| --- | ---: | --- | ---: |
| apple → dish | 20 | 1/5/10/20/30 | 10 |
| pan → stove | 20 | 1/5/10/20/30 | 10 |
| pet → shelve | 20 | 1/5/10/20/30 | 10 |
| right cup → left cup | 20 | 1/5/10/20 | 0 |
| three blocks → tray | 70 | 1/5/10/20/30 | 0 |
| three cups → tower | 70 | 1/5/10/20/30 | 0 |

모든 태스크에서 config별 train 파일 **집합**이 같고, train/val 교집합이 없으며 val 집합이 엄격히 중첩된다. 해당 config 폴더도 모두 존재한다. 이는 manifest의 source 파일 목록에 대한 검증이며, 영상 내용 수준 중복이나 co-training base DROID 전체와의 중복 검사까지 의미하지 않는다.

따라서 데이터 셀은 29개지만, 현재 DROID 체크포인트가 있는 태스크는 3개여서 즉시 대상은 **15개 task×val 셀, 150개 checkpoint×val 평가**다. 세 run은 모두 epochs 5,10,…,50이고 합계 약 27.3 GiB가 로컬에 있다. 저장 config는 base droid + 해당 task의 val30 config를 사용한다. 같은 checkpoint에 --dataset-name만 각 val config로 바꾸면 된다.

근거: /workspace/data/*/droid_split_manifest.json 및 /workspace/droid_policy_learning/checkpoints/*/*/config.json.

## 리뷰의 4개 blocker는 DROID에서 어떻게 바뀌나

| 리뷰 항목 | DROID 판정 |
| --- | --- |
| B1: openpi의 오래된 surval pin | DROID 로컬 CLI에는 해당하지 않음. surval/scripts는 로컬 src를 먼저 import한다. 이 실행에 commit/push는 필요하지 않다. 배포·다른 머신의 설치본 재현성은 별도 문제다. |
| B2: feature/provenance 없는 cache | 수정된 DROID 생산자는 encoder features, normalization, t+1 offset, complete/provenance를 저장한다. 이번 변경은 독립 row manifest까지 추가했다. 기존 legacy cache는 재생성이 필요하다. |
| B3: joint_velocity와 pos_rot6d 불일치 | 현재 DROID 3개 run은 모두 10-D absolute pos(3)/rot6d(6)/gripper(1). 학습과 cache가 동일한 droid_dataset_transform을 쓰고 생산자는 다른 action key 구성을 거부한다. pos_rot6d 재학습 요구는 현재 DROID에는 없다. |
| B4: 로컬 checkpoint 부재/373GB 다운로드 | DROID는 30개, 약 27.3 GiB가 이미 로컬에 있다. 현재 목적에 gs:// 다운로드나 streaming wrapper가 필요 없다. |

실제 DROID 실행의 blocker였던 NumPy 2.2.6 / TensorFlow 2.15 ABI 문제는 사용자 승인 후 별도 .venv-droid의 호환 패키지로 해결했다. 기존 /venv/droid_policy의 NumPy는 여전히 2.2.6이며 그 환경은 변경하지 않았다.

또한 epoch-only baseline glob 문제는 원래 DROID epoch cache에는 없었다. 현재 공유 작업트리에서는 다른 세션이 step autodetection도 추가했으므로, 기존 리뷰의 epoch-only 설명은 더 이상 현 상태 전체를 설명하지 않는다.

근거: [cache producer](/workspace/droid_policy_learning/robomimic/scripts/sequential_cache_checkpoints.py), [RLDS transform](/workspace/droid_policy_learning/robomimic/utils/rlds_utils.py), [cache adapter](/workspace/surval/src/surval/droid.py), [baseline extractor](/workspace/surval/src/surval/tb_aggregate/seqcache_metrics.py).

## 기존 DROID scorer와 surval-library 경로 차이

| 항목 | 기존 DROID cache scorer | 현재 surval DROID 경로 |
| --- | --- | --- |
| threshold | inter-demo 진행률 기반 또는 intra-demo action 차이로 추정한 block scale | 현재 checkpoint의 inference encoder, 관측 history 전체를 flatten한 feature의 cosine kNN, query별 local threshold |
| rotation | 10-D layout의 SO(3) block 사용 | 같은 SO(3) 기하를 사용하되 checkpoint normalization을 먼저 역변환 |
| 연속 gate | exp(-max(error-1,0)/tau); tau를 주거나 episode IQR 사용 가능 | library의 exp(-max(error-1,0)); legacy tau는 gate에 사용하지 않음 |
| block 결합 | N번째로 작은 block probability | 고정된 logsumexp smooth-min |
| 시간 결합 | cumprod 또는 weighted-sum 옵션 | 현재 경로는 cumprod 후 prefix 평균 |
| hard/soft | 두 경로를 노출 | 외부 선택 옵션 없음. 내부 library의 legacy soft 이름만 사용 |
| baseline 출력 | 기존 validation 로그/old cache 중심 | checkpoint CSV/JSON의 Loss, OMN, MSE 5종과 별도 outcome 집계 |

두 경로의 raw SURVAL 숫자를 그대로 같은 지표의 이전/이후 값으로 비교하면 안 된다. threshold 정의와 block 집계가 바뀌었다. 새 경로의 MSE는 저장된 normalized action에서 계산하며, SURVAL/threshold는 physical pos와 SO(3) units를 사용한다. 현재 observation t에 대한 예측과 GT 시작점은 모두 original action t+1이다.

근거: [old scorer](/workspace/droid_policy_learning/robomimic/scripts/sequential_validate_from_cache_base.py:850), [library gate/aggregation](/workspace/surval/src/surval/sequential_validate.py:1153), [new adapter](/workspace/surval/src/surval/droid.py).

## 남아 있는 방법론 문제 — 중요도 순

### 1. val을 reference와 query에 동시에 쓰는 문제: 맞지만 무조건 무효는 아님

현재 build_policy_thresholds는 그 val cache의 feature/GT로 DB와 global/local threshold를 함께 만든다. val1→val30에서 평가 표본과 기준이 동시에 달라진다. 따라서 질문이 “고정된 기준으로 정책 품질을 추정할 때 val을 몇 개 써야 하는가”라면 교란 요인이다.

반대로 “총 validation budget 안에서 calibration과 scoring까지 함께 수행할 때 얼마나 유용한가”를 묻는다면 그 자체로 정의 가능한 실험이다. 모든 결과가 무효라거나 val1은 반드시 해석 불가능하다는 단정은 과하다. 어떤 예산을 평가하는지 명시해야 한다.

same_demo_allowed=True에서는 ±5 timestep 밖의 동일 데모 이웃을 사용할 수 있다. val1이 반드시 local fallback이라는 뜻은 아니다. False이면 단일 데모에서 이웃이 사라지고, 현재 global 계산도 같은 필터를 써 유효 pair가 없으면 0을 남긴다. 단순히 “global fallback이 해결해 준다”고 보아서는 안 된다. 최소한 fallback 비율과 block scale 분포를 함께 보고해야 한다.

**권고:** val 표본 수의 효과가 목적이면 task train source 집합을 reference로 고정하고 val은 query로만 사용한다. “고정”은 reference episode 집합에 대한 말이며 policy-dependent encoder는 checkpoint마다 다시 계산한다. train에 노출된 행동 reference에 대한 적합도를 측정하는 설계라는 점도 명시해야 한다.

이는 train=False를 True로 바꾸는 한 줄 변경이 아니다. 현재 threshold API는 DB 자체의 각 row를 query로 사용하고 scorer는 (demo,t) lookup을 요구한다. 별도 val query feature → train reference kNN → **val row key를 가진 threshold map** 생성이 필요하다. query GT를 reference pool에 넣지 않아야 한다. 이번에는 연구 정의를 임의로 바꾸지 않았다.

### 2. DROID OMN은 리뷰의 global-val OMN과 다르며 더 직접적인 문제가 있음

[train_on_batch](/workspace/droid_policy_learning/robomimic/algo/diffusion_policy.py:281)는 마지막 observation feature를 Ta번 expand한다. [compute_off_manifold_errors](/workspace/droid_policy_learning/robomimic/utils/loss_utils.py:211)는 이를 B×Ta개 state로 flatten하고, batch 안에서 L2 kNN을 찾으며 diagonal 하나만 제외한다.

현재 Ta=8, k=5에서는 동일 관측의 다른 horizon 항목 7개가 사실상 같은 feature다. 다른 window와 동률인 특수 경우를 제외하면 그 7개가 가장 가까워, k=5가 다른 state/episode 대신 **같은 관측의 미래 expert action**으로 채워질 수 있다. 별도 CPU 구성에서 이웃 100%가 같은 observation window에 속함을 확인했다.

따라서 DROID OMN은 batch 크기·구성, horizon, padding과 feature 반복 구조에 의존한다. global val manifold를 만들고 query를 평가한다는 설명과 다르다. 다만 expert action이 비슷하다는 이유만으로 잔차가 0이 되는 것은 아니다. 모두 e1인 expert와 e2 prediction 반례에서는 OMN=1이었다.

**개선 후보:** 다른 observation row만 reference 후보로 삼고 temporal/source exclusion을 적용하거나, 고정 train reference를 사용하는 명시적인 새 OMN을 만든다. 기존 학습-log baseline과 숫자가 달라지므로 조용히 기존 OMN 이름으로 대체하면 안 된다. 추가로 k=1 경로는 선형 span projection이 아닌 최근접 action 차이를 반환하는 별도 정의라는 점도 있다.

### 3. Loss/OMN의 표본 범위·가중치와 EMA 혼합

- 기본 --valid-num-steps=50, batch-size=32이면 Loss/OMN은 최대 1,600행만 본다. MSE와 SURVAL은 cache 전체에서 계산한다. 큰 val split에서 같은 데이터 범위가 아닐 수 있다.
- 기존 TrainUtils.run_epoch는 batch scalar들의 단순 평균이다. 마지막 작은 batch도 같은 무게다. MSE extractor는 전체 scalar element 평균이므로 가중치가 다르다.
- validation Loss는 action reconstruction MSE가 아니라 랜덤 timestep/noise에 대한 **noise-prediction MSE**이며 prediction horizon 16을 쓴다.
- Loss와 OMN neighbor feature는 self.nets(non-EMA)를 사용한다. cache prediction과 SURVAL feature는 EMA inference model을 사용한다. OMN은 EMA prediction + non-EMA last-frame feature의 조합이다.

**개선 후보:** 전체 split을 평가하는 baseline 모드, 실제 row 수 가중 평균, EMA/non-EMA 출처 명시를 추가한다. 기존 학습 로그와 맞춘 historical baseline인지 배포 EMA policy 자체의 validation loss인지 먼저 구분해야 한다. 이번 loader 변경은 이를 의도적으로 유지한다.

### 4. 부가적인 비교 조건

MSE_per_dim_norm은 해당 val의 GT 표준편차를 분모로 쓰므로 val 크기에 따라 기준이 바뀌는 추가 baseline이다. 기본 MSE는 checkpoint의 고정 normalizer를 사용하므로 이 문제와 다르다. 또한 누적 SURVAL은 trajectory 길이에 영향을 받고 episode 평균인 반면 MSE는 frame/element 가중 평균이다.

중첩된 val1/5/10/20/30은 독립 표본이 아니다. 하나의 고정 nested split으로부터 얻은 결과만으로 “일반적으로 N demo면 충분”하다고 일반화하지 말고, 필요하면 episode subset 반복/episode-level resampling을 별도 설계해야 한다. 동일 checkpoint의 real-world success outcome은 val 설정마다 같은 값으로 연결해야 하며 별도의 독립 실험으로 세지 않는다.

## 이번에 적용한 loader

[materialize_validation_rows](/workspace/surval/src/surval/rlds_cache.py)는 안정적 source ID와 원래 timestep을 독립 metadata pass로 확인한다. subset은 source ID 사전순으로 선택하며 window를 만든 뒤 자르므로 capped 마지막 행의 future GT가 변하지 않는다. 이후 read/map/interleave/decode를 기본 4 worker로 처리하고 누락·중복을 검사한 뒤 source/timestep 고정 순서로 정렬한다. Torch 추론은 shuffle=False, drop_last=False를 유지한다.

각 checkpoint의 모든 prediction sample과 feature row를 다시 확인하고, row 수·episode 수·key SHA256을 provenance에 저장한다. 소비자도 manifest와 cache key를 대조한다. 이 강화된 계약은 robomimic producer에 한정하여 동시 추가된 openpi adapter를 보존했다. 기존 decoded-data materialization은 유지하므로 대규모 데이터의 RAM 제한은 여전히 존재하며 streaming loader는 이번 범위가 아니다.

실제 10-D DROID val5에서 worker 1/4 모두 5개 episode, 626행, action shape (16,10), 같은 전체 key hash와 모든 decoded 값이 확인됐다. 단일 CPU 측정은 2.894초 vs 0.935초였으며 metadata pass를 포함하지만 반복 성능 측정이나 전체 모델 실행 속도의 보장은 아니다.

## 실행 환경과 확인된 출력

/workspace/surval/.venv-droid는 기존 /venv/droid_policy의 패키지를 읽기 전용으로 재사용하는 **별도 venv overlay**다. 모든 패키지를 독립 복제한 self-contained 환경은 아니다. numpy 1.26.4, faiss-cpu 1.13.2, sklearn 1.7.2, OpenCV 4.11, TF metadata 1.15, protobuf 3.20.3 등의 변경은 새 venv 안에만 설치했다. 기존 torch 2.0.1 / TF 2.15를 사용한다.

언어 인코더 파일이 없어 첫 offline smoke는 실패했다. 필요한 기존 DistilBERT 파일을 새 venv의 model-cache에 받아 재실행했고 실제 checkpoint 추론이 통과했다. Torchvision은 첫 모델 생성 시 기존 pretrained ResNet50 파일을 표준 Torch cache에 받았다. GPU 작업은 건드리지 않았다.

- 실행 결과: [12-row / 2-sample checkpoint metrics](/workspace/surval/outputs/droid_smoke_20260910/evaluation_multi/checkpoint_metrics.csv)
- 실행/검증 기록: [/workspace/surval/.loop-engineering/20260910T065508Z-droid-loader-review/](/workspace/surval/.loop-engineering/20260910T065508Z-droid-loader-review/final-report.md)

smoke의 temporal-radius=0, min-neighbors=2 설정은 제한된 행으로 파일/수식 경로를 검증하기 위한 것이며 최종 실험 설정이나 real-world 성능 증거가 아니다. custom success-rate 값은 제공되지 않아 실제 상관 결과는 아직 없고, 집계 로직은 별도 합성 데이터로 검증했다.

## 다음 결정

val 표본 수의 효과만 비교하려면 **고정 train reference + val query**를 권고한다. 진행한다면 SURVAL query/reference 분리와 OMN의 observation-level 이웃 정의를 함께 명시하고, 기존 OMN도 historical baseline으로 남기는 것이 안전하다. action-space 재학습이나 checkpoint 재다운로드는 현재 DROID 3개 태스크에 필요하지 않다.
