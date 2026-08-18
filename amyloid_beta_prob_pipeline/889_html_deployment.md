# HTML 배포 도구 — 진행 상황 (N=889 파이프라인 후속)

> 최종 갱신: 2026-08-14 · 모델 설계는 `claude/889_pipeline_decisions.md` 참고

---

## 산출물

| 파일 | 내용 |
|:--|:--|
| `build_index.py` | HTML 템플릿이 문자열로 내장된 **단일 생성기**. 이 파일 하나로 index.html 재생성 가능 |
| `models.json` | `run.export_all_models_json()` 결과. 11개 모델, 101 KB |
| `index.html` | 생성물 (97 KB). 직접 편집 금지 |

### 폴더 구조

```
Release/            (또는 Deployment)
├── build_index.py
└── index.html
results/
└── 889_amyloid_beta_prob_pipeline/
    ├── models.json
    ├── golden_M4_SCD_MCI.csv
    └── summary.csv
```

### 실행

```bash
cd Release
python build_index.py                       # models.json 자동 탐색
python build_index.py --verify ../results/889_.../golden_M4_SCD_MCI.csv
```

`models.json` 탐색 순서: 같은 폴더 → `../results/*/models.json` → `../../results/*/models.json`.
후보가 여럿이면 최신 파일 사용. `--json`으로 직접 지정 가능.

---

## 설계 결정

**JSON을 HTML 안에 인라인** — `file://`에서 `fetch()`는 CORS로 차단되므로 미리 합침.
로컬 더블클릭 / GitHub Pages / 사내 공유 어디서든 동일 동작, 파일 하나만 전달하면 끝.

**계산부는 3줄**

```js
let logit = m.intercept;
for (const [k, v] of Object.entries(m.coefficients)) logit += v * x[k];
const p = 1 / (1 + Math.exp(-logit));
```

StandardScaler를 쓰지 않아 scaler 정보가 불필요. Platt/isotonic 보정도 미적용(의도적).

**SheetJS는 cdnjs에서 로드** — 사내망 차단 시 Excel 입출력 불가(확률 계산은 동작).
감지해서 경고 배너 표시. 필요하면 라이브러리를 인라인한 오프라인 버전(약 1 MB) 제작 가능.

---

## 화면

1. **업로드** — SCST 원본 xlsx. 시트 13개 중 `COMMON` 자동 선택(변경 가능)
2. **컬럼 확인** — 18개 z-score + AGE/education_year/sex/user_ID 자동 매핑, 수동 교정 가능
3. **환자 표** — 인지상태·APOE 수동 입력(기본 Unknown), 적용 모델 뱃지, 확률 막대. 일괄 적용 지원
4. **상세 패널**(행 클릭) — 확률 hero, 모델 정보(n·유병률·CV AUC), 전체 OR 표(mini forest plot, Top-3 ★ 하이라이트), 접이식 회귀식

**입력 선택지 (자유 입력 없음 — 오타 방지)**

- `cognitive_status`: Unknown / SCD / MCI / Dementia
- `APOE`: Unknown / E2/E2 / E2/E3 / E2/E4 / E3/E3 / E3/E4 / E4/E4
  → `APOE_e4_carrier`는 `E4` 포함 여부로 자동 파생

**순서 강제** — 인지상태 없이 APOE만 입력하면 계산 차단 + 배너 안내
(인지기능검사 → 혈액검사 순서)

**표시하지 않는 것** — 환자별 예측확률의 CI. "내 amyloid 상태가 이 범위"로 오독될 위험.
표시되는 CI/p-value는 **모델 계수** 기준임을 화면에 명시.

---

## Excel 출력 컬럼 (고정, 이 순서)

```
SCST_DATE, user_ID, NAME, Institution, sex, AGE, education_year,
cognitive_status, APOE, model_used,
P_amyloid_beta_positive_raw, P_amyloid_beta_positive_percent
```

- 미입력은 빈 칸(null). `header` 명시로 전원 미입력이어도 컬럼 유지
- 두 번째 시트 `모델정보`: 생성일시, 원본파일, 대상 인원, 사용 모델별 학습 n·CV AUC, 면책 문구

---

## 검증 완료

| 항목 | 결과 |
|:--|:--|
| JSON 계수 → 브라우저 확률 vs `python_prob` | 최대 오차 **9.2e-06** (golden set 20명) |
| 실제 SCST 원본 xlsx 업로드 | COMMON 시트 자동 선택, 18개 z-score 전부 자동 매핑 성공 |
| 모델 자동 선택 | M2_All / M2_SCD_MCI / M3_Dementia / 차단 4가지 조합 확인 |
| 순서 강제 | APOE만 입력 시 차단 + 배너 정상 |
| Excel 다운로드 | 12컬럼 순서·null 처리 확인 |

**발견·수정된 버그** — 성별 인코딩. 파일에 `남성`만 있으면 첫 값을 여성으로 잡아
전원 `sex_여성=1`이 될 뻔함(에러 없이 확률만 틀림). 이제 여성 후보가 없으면
"해당 없음(전원 남성)"으로 두고 경고 표시.

---

## 미채택 결정

- **환자별 SHAP/기여도 표시** — 선형 로지스틱에서 `φ_j = β_j(x_j - x̄_j)`로 정확히 계산
  가능하지만 **하지 않기로 함**. Top-K는 모델 전체 공통 3개만 표시.
- **Uniform shrinkage** — Dementia calibration slope 0.62~0.68이지만 보정하지 않고
  한계로 명시하기로 함.
- **원본 전체 컬럼 포함 옵션** — 출력 컬럼 12개 고정으로 결정하며 제거.

---

## 남은 작업

- [ ] **검사 완료 데이터로 실검증** — `*_z_score`에 값이 채워져 나오는지만 확인하면 됨
      (컬럼명·시트 구조는 이미 검증 완료. 값만 있으면 추가 작업 없음)
      → 회사 문서보안 승인 대기 중
- [ ] **GitHub repo 생성 + Pages 연결** (public repo로 진행 결정)
- [ ] **`.GIF` 촬영 → README 삽입** — 승인 후 본인이 직접 진행

### repo에 올릴 것

`build_index.py`, `models.json`, `index.html` (환자 데이터는 일절 포함하지 않음)

포트폴리오 Quarto 사이트와는 **별도 repo**로 운영. 같은 repo에 둘 경우
Quarto의 `index.qmd` → `index.html` 렌더와 충돌하므로 `app/index.html` 등 하위 경로 사용 필요.