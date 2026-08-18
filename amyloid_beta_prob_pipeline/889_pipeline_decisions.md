# Amyloid Beta 양성 예측 파이프라인 — N=889 최종 설계 결정

> 최종 갱신: 2026-08-14
> 대상: `Project/03. vaccine_naive_889_dataset/02_amyloid_beta_prob_pipeline.ipynb`,
> `Project/amyloid_beta_prob_pipeline/`

---

## 0. 배경 — 왜 N=889인가

- `N=455` 버전은 소검사 단위 지표(RT 등)를 모두 갖고 있었으나 **성능이 나오지 않아 유기**
- `N=889` 버전은 소검사 RT가 없는 케이스를 포함 → **RT 등 추가 feature를 포기하고
  기존 composite score만으로 진행**하기로 확정
- 즉 이번 작업은 기존 파이프라인의 **답습 + 정리**이며, 새 feature 도입은 없음

---

## 1. 데이터

`1st_cognitive_status` × `1st_amyloid_status` 교차표 (N=889)

| | 양성 | 음성 | Total |
|:--|--:|--:|--:|
| SCD | 35 | 179 | 214 |
| MCI | 142 | 261 | 403 |
| Dementia | 185 | 87 | 272 |
| **Total** | **362** | **527** | **889** |

- 전체 유병률 **40.7%**
- **APOE 결측 없음** (complete case 고민 불필요)
- SCST composite: **18개 z-score** (demographic-stratified z-score, 사전 지정)
- SCD 단독은 양성 35명으로 너무 적어 **SCD + MCI 병합**

---

## 2. 알고리즘 — 확정 사항

| 항목 | 결정 | 근거 |
|:--|:--|:--|
| 알고리즘 | Logistic Regression 단독 | RF/SHAP 제거. 배포 가능한 계수가 목표 |
| 규제 | **없음** (No LASSO, No Ridge) | OR/CI/p-value를 확증적으로 보고하기 위함 |
| 변수선택 | 없음 (18개 전부 사전 지정) | post-selection inference 문제 제거 |
| 스케일링 | **StandardScaler 제거** | ①z-score 이중표준화 방지 ②HTML 이식 단순화 ③OR 해석 직관화 |
| 표본 분할 | **없음** — 전체 889로 refit 1회 | 데이터 기반 선택 단계가 없으므로 hold-out 불필요 |
| 성능 추정 | **5-fold CV의 OOF 예측만** | apparent 성능은 낙관 편향이라 아예 계산 안 함 |
| OR 출처 | **최종 refit 모델 1개** | fold별 OR 평균은 통계적으로 무의미 |

### 2.1 StandardScaler를 뺀 이유 (중요)

1. composite는 이미 demographic-stratified z-score → 재표준화 시 "규준 대비"가
   "이 표본 대비"로 바뀌어 OR 해석이 달라짐
2. 계수가 원 스케일이라 **JSON에 계수만 넣으면 HTML에서 그대로 계산 가능**
   (scaler의 `mean_`/`scale_` 불필요)
3. OR 해석이 직관적: composite "z-score 1점당", AGE "1세당", 이진변수 "집단 간 대비"

### 2.2 fold별 OR 평균을 폐기한 이유

- OR은 로그 스케일에서 대칭 → `mean(exp(β)) ≠ exp(mean(β))` (Jensen 부등식)
  → 이전 리포트의 OR은 체계적으로 **부풀려져 있었음**
- p-value의 평균은 어떤 가설검정의 결과도 아님
- LASSO를 없앤 지금은 모든 fold가 같은 feature set → CV로 계수를 볼 이유 자체가 없음

### 2.3 CV의 역할

> CV는 "최종 모델"이 아니라 **"모델링 절차"**를 평가한다.

feature 사전 지정 + 규제 없음 + 하이퍼파라미터 없음 → 절차가 완전히 결정론적이므로
절차에 대한 성능 추정치를 그 산물인 최종 모델에 귀속시키는 것이 정당 (TRIPOD 방식).
또한 CV 모델은 n×0.8, 최종 모델은 n 전체로 적합 → CV는 최종 성능을 **약간 과소평가**
(보수적 방향이라 안전).

CV가 없으면 못 하는 것: ①정직한 성능 숫자 ②Youden threshold ③DeLong 비교
④정직한 calibration 확인

---

## 3. Group × Model 구조

covariate는 하나씩이 아니라 **step 단위**로 누적 (sex/AGE/education은 임상적으로 항상 함께 확보됨).

### Group 1. All (n=889)
1. `M1_All` — composite 18개만
2. `M2_All` — + sex, age, education_year
3. `M3_All` — + `1st_cognitive_status` (is_MCI, is_Dementia)
4. `M4_All` — + APOE

### Group 2. SCD_MCI (n=617)
1. `M1_SCD_MCI` → 2. `M2_SCD_MCI` (+demo) → 3. `M3_SCD_MCI` (+is_MCI) → 4. `M4_SCD_MCI` (+APOE)

`is_Dementia`는 이 subgroup에서 상수 → **자동 제거됨** (절편과 완전공선 방지)

### Group 3. Dementia (n=272)
1. `M1_Dementia` → 2. `M2_Dementia` (+demo) → 3. `M3_Dementia` (+APOE)

인지상태가 상수이므로 covariate로 사용 불가

### EPV (events per variable, 소수 클래스 기준)

| Group | 소수 클래스 | Model 범위 | EPV |
|:--|:--|:--|:--:|
| All | 양성 362 | M1→M4 | 20.1 → 15.1 ✓ |
| SCD_MCI | 양성 177 | M1→M4 | 9.8 → 7.4 △ |
| Dementia | **음성 87** | M1→M3 | 4.8 → 4.0 ✗ |

**Dementia는 EPV가 낮아 계수 과대추정/separation 위험.** 합의: 일단 진행하되
자동 진단 플래그(⚠)를 달아두고, 실제로 뜨면 그때 composite 축소 또는 Firth 보정 검토.
(AUC가 괜찮은 것과 계수가 믿을 만한 것은 별개임에 유의)

---

## 4. Application — 모델 자동 선택 규칙

정보 획득 순서: **인지기능검사 → 혈액검사**

| cognitive status | APOE | 사용 모델 |
|:--:|:--:|:--|
| Unknown | Unknown | `M2_All` |
| Known | Unknown | `M2_SCD_MCI` 또는 `M2_Dementia` |
| Unknown | Known | PASS — cognitive status 입력을 먼저 요청 |
| Known | Known | `M4_SCD_MCI` 또는 `M3_Dementia` |

`M3_All`은 분석용으로만 존재하고 Application 경로에서는 쓰이지 않음 (의도된 설계).

---

## 5. Top-K biomarker 규칙 (확정)

1. `p < 0.05` 인 변수만 후보
2. 후보를 **`|β| = |log(OR)|` 내림차순** 정렬
3. 상위 **K=3** 하이라이트
4. 각주: "다중비교 미보정 — 탐색적 해석"

### |OR|이 아니라 |log(OR)|인 이유

OR은 1 기준 비대칭 → OR=2와 OR=0.5는 크기가 같은데 `|OR|`로는 2가 4배 커 보임.
이 데이터는 **인지 z-score가 높을수록 amyloid 음성**이라 composite 대부분의 OR이 1 미만
→ `|OR|` 정렬 시 핵심 마커가 꼴찌로 밀리고 AGE(OR≈1.08) 같은 게 위로 올라오는 사고 발생.

| 변수 | OR | \|OR\| 순위 | \|log OR\| | 올바른 순위 |
|:--|:--:|:--:|:--:|:--:|
| delayed_free_recall | 0.45 | 3위 | 0.80 | **1위** |
| APOE_e4_carrier | 1.90 | 1위 | 0.64 | 2위 |
| AGE | 1.08 | 2위 | 0.077 | 3위 |

### 검토했으나 채택하지 않은 것

- **β×SD 표준화 효과크기**: 단위가 다른 변수(AGE 1세당 vs z-score 1점당) 비교를
  공정하게 만들지만, 사용자 결정으로 **미채택**. 계수 크기 그대로 사용.
- **Top-K 대상을 composite로 제한**: APOE가 성능 기여가 가장 컸다는 기존 관찰이 있어
  **전체 변수 대상으로 유지**. (이분형 변수의 β는 실재하는 집단 대비라 해석에 문제없음)

---

## 6. Calibration

- **노트북에서 확인만.** Reliability diagram + Brier score + calibration slope/intercept
- **Platt scaling / Isotonic 절대 사용 금지** — 배포 식이 sigmoid를 두 번 통과하는
  구조가 되어 "학습된 식 그대로 이식"이 깨짐
- LR을 MLE로 적합하면 학습 데이터에서 자동으로 calibration-in-the-large 성립 →
  원래 잘 보정된 편. OOF로 확인해야 의미 있음
- 현장 유병률이 표본(40.7%)과 다르면 **절편만** 보정:
  `β0_new = β0 + log[(p_new/(1−p_new)) / (p_old/(1−p_old))]`
  → JSON의 `intercept` 필드 하나만 고치면 됨
- `class_weight='balanced'` 금지 — 절편이 왜곡되어 확률이 망가짐

---

## 7. 코드 구조 (6개 모듈, 약 1,700줄)

| 파일 | 역할 |
|:--|:--|
| `config.py` | GROUPS/feature/Top-K 규칙/진단 임계값 등 설정만 |
| `data_prep.py` | target 매핑, sex 더미, APOE→e4 carrier, 인지상태 더미, group 필터 |
| `modeling.py` | CV(OOF), Youden threshold, 최종 refit, 계수 추출 |
| `stats_tests.py` | LRT(최종모델 1회), DeLong(pooled OOF) |
| `reporting.py` | CV 성능, OR표(Top-K), 회귀식, calibration |
| `run_experiments.py` | 오케스트레이션, JSON export, golden set |

### 삭제된 것

RandomForest 전부 / LASSO 변수선택 / SHAP / grouped permutation importance /
RF 하이퍼파라미터 탐색 / StandardScaler / `split_train_test` / `report_test_performance` /
fold 기반 `report_lr_odds_ratio`

### 신규

`report_final_or` (Top-K 포함) / `format_formula` / `report_calibration` /
`export_all_models_json` / `make_golden_set` / `drop_degenerate_features` /
separation 자동 진단 / EPV 자동 출력

### 사용법

```python
import data_prep
import run_experiments as run

df, cov_map = data_prep.preprocess(df_composite_raw)
summary_df, detail = run.run_all_experiments(
    df, cov_map,
    target_folder='amyloid_beta_prob_pipeline',
    result_folder='889_final',
)
run.print_full_report(detail)
run.export_all_models_json(detail, cov_map, '../amyloid_beta_prob_pipeline/models.json')
run.make_golden_set(detail, df, cov_map, 'M4_SCD_MCI',
                    '../amyloid_beta_prob_pipeline/golden_M4_SCD_MCI.csv')
```

---

## 8. HTML 배포 (다음 단계)

### 원칙

- **엑셀 파일을 GitHub에 올리지 않음.** 회사에서 환자 데이터가 생길 때마다 직접
  업로드/다운로드하는 방식
- 모든 연산이 **브라우저 내에서만** 수행, 네트워크 전송 0
- "연구용 참고 도구, 진단 목적 아님" 고지 필요

### JSON 구조 (Python ↔ HTML 유일한 연결고리)

```
{
  "meta": {...},
  "application_rule": {...},
  "encoding": {...},          // 모든 모델 공통, 최상위 1회만
  "models": {
    "M2_All": {
      "intercept": ..., "coefficients": {...}, "features": [...],
      "formula": "...", "or_table": [{feature, beta, OR, ci_lower,
                                      ci_upper, p_value, top_k}],
      "n": ..., "n_events": ..., "cv_auc_mean": ..., "threshold_youden": ...
    }, ...
  }
}
```

**11개 모델 전부 export** (총 101 KB — 용량/속도 문제 없음).
HTML에서 `models["M2_All"]` 식으로 골라 사용.

### HTML 계산부 (이게 전부)

```js
let logit = m.intercept;
for (const [k, v] of Object.entries(m.coefficients)) logit += v * x[k];
const p = 1 / (1 + Math.exp(-logit));
```

### 화면 표시 항목

- P(amyloid+) 확률
- 최종 모형 수식
- 현재 적용된 모델명 (예: `M2_SCD_MCI`)
- 각 계수의 **Odds Ratio (95% CI), p-value** — "모델 계수 기준"임을 명시
- Top-K 변수 하이라이트

**환자별 예측확률의 CI는 표시하지 않음** — "내 amyloid 상태가 이 범위"로
오독될 위험. 표시하는 CI는 **계수의 CI**뿐임을 라벨로 명확히 할 것.

### 남은 작업

- [ ] Raw SCST Excel 업로드 → 환자별 cognitive status / APOE 입력 UI
      (미입력 시 None 버튼. cognitive status 없이 APOE만 입력 시 순서 안내)
- [ ] 대시보드 표시 → Excel 다운로드
- [ ] **Python ↔ JS parity test**: `make_golden_set()`으로 뽑은 CSV의 `python_prob`과
      JS 결과가 소수점 6자리까지 일치하는지 확인
- [ ] GitHub Pages 배포 + 전 과정 `.GIF`를 README에 삽입

---

## 9. 검증 완료 사항 (합성 데이터, 실제 분포 재현)

- 전 과정 무에러 실행 (preprocess → CV → refit → OR/수식/LRT/calibration → JSON)
- **JSON 계수 이식 정확도**: 11개 모델 전부 손계산 vs `predict_proba` 최대 오차
  `1.1e-05` (계수 6자리 반올림에서 기인, 허용 범위)
- SCD_MCI에서 `is_Dementia` 상수 컬럼 자동 제거 확인
- separation 진단 플래그 정상 발동 확인
- EPV 자동 출력 및 10 미만 경고 확인

---

## 10. 실제 데이터 실행 시 확인할 것

1. **Dementia 그룹에 ⚠ separation 플래그가 뜨는가** → 뜨면 composite 축소/Firth 검토
2. **calibration slope가 1에서 크게 벗어나는가** → 과적합 신호
3. **DeLong에서 M2→M3(인지상태) 및 M3→M4(APOE) 향상이 유의한가**
   → "채혈까지 할 가치가 있는가"에 대한 근거
4. composite의 SD가 1에서 크게 벗어나는 것이 있는가 → 규준(norming) 점검 신호