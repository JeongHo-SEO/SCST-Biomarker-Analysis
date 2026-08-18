"""
config.py
---------
Alzheimer's amyloid PET 양성 예측 프로젝트(N=889)의 "실험 정의" 전용 파일.

이 파일에는 실행 로직이 없습니다. Group x Model 조합을 어떻게 정의하는지,
어떤 feature를 쓰는지 등 "설정값"만 모아둡니다.

--- 이전 버전(RF+SHAP 포함)과 달라진 점 ------------------------------------
1. RF / LASSO / SHAP 전부 제거 -> Logistic Regression 단일 알고리즘
2. train/test split 제거 -> 전체 N=889로 최종 refit 1회
   (성능 추정은 5-fold CV의 out-of-fold 예측으로만 수행)
3. StandardScaler 제거 -> 계수가 원 스케일 그대로라 OR 해석과 HTML 배포가 직결
4. Group 재정의: SCD/MCI 개별 -> SCD_MCI 병합 (SCD 단독은 양성 35명으로 너무 적음)
5. covariate를 "하나씩" 이 아니라 "step 단위"로 누적
   (sex/AGE/education_year는 임상적으로 함께 얻어지므로 한 덩어리로 추가)
----------------------------------------------------------------------------

target: 1st_amyloid_status (양성 vs 음성)
"""

# ============================================================
# 0. 전역 상수
# ============================================================

# 전체 파이프라인 공통 seed. 여기 값을 바꾸면 모든 모듈에 일괄 반영됨.
RANDOM_STATE = 42

# 5-fold CV. 성능 추정 + Youden threshold + DeLong 비교의 유일한 근거.
#
# ** CV의 역할에 대해 (중요) **
# CV는 "최종 모델"을 평가하는 것이 아니라 "모델링 절차"를 평가함.
# 이번 설계는 feature 사전 지정 / 규제 없음 / 하이퍼파라미터 없음이라
# 절차가 완전히 결정론적이므로, 절차에 대한 성능 추정치를 그 절차의 산물인
# 최종 모델에 그대로 귀속시키는 것이 정당함 (TRIPOD 권고 방식).
# 또한 CV 각 모델은 n*0.8로, 최종 모델은 n 전체로 적합되므로 CV 추정치는
# 최종 모델 성능을 "약간 과소평가"함 - 보수적으로 틀리는 방향이라 안전.
N_CV_SPLITS = 5

TARGET = '1st_amyloid_status'

# target 매핑: data_prep.preprocess()에서 한 번만 적용
TARGET_MAP = {'음성': 0, '양성': 1}

# 원본 데이터에서 무조건 제외할 컬럼
ID_COLS = ['user_ID']
UNUSED_COLS = ['education_category', '졸업여부']

# 인지 상태 컬럼 (그룹 필터링 + covariate 더미 생성 양쪽에 사용)
COGNITIVE_STATUS_COL = '1st_cognitive_status'
COGNITIVE_STATUS_REFERENCE = 'SCD'  # is_MCI, is_Dementia 더미의 기준 카테고리

# 더미 컬럼 생성 순서를 데이터 등장 순서가 아니라 여기서 "고정"함.
# (df.unique()에 의존하면 데이터가 바뀔 때 컬럼 순서가 달라져 재현성이 깨짐)
COGNITIVE_STATUS_CATEGORIES = ['MCI', 'Dementia']


# ============================================================
# 1. Feature 정의
# ============================================================

# composite_scores : 모든 Group x Model 공통으로 항상 포함되는 18개 신경심리검사
# z-score. Demographic-stratified z-score이므로 이미 규준 대비 표준화되어 있음
# -> StandardScaler를 추가로 적용하지 않음 (이중 표준화가 되어 "규준 대비"라는
#    해석이 "이 표본 대비"로 바뀌어 버림).
COMPOSITE_SCORES = [
    'visual_forward_z_score', 'visual_backward_z_score',
    'cowat_fruits_z_score', 'cowat_di_z_score', 'naming_z_score',
    'block_design_z_score',
    'time_orientation_total_z_score',
    'immediate_free_recall_trial_1_z_score', 'immediate_free_recall_trial_2_z_score',
    'immediate_free_recall_trial_3_z_score',
    'delayed_free_recall_z_score',
    'word_recognition_true_positive_z_score', 'word_recognition_true_negative_z_score',
    'place_recognition_z_score',
    'trailmaking_part_a_time_z_score', 'trailmaking_part_a_error_z_score',
    'trailmaking_part_b_time_z_score', 'trailmaking_part_b_error_z_score',
]

# get_dummies(drop_first=True)로 "한 번만" 더미화할 컬럼.
# APOE는 6-genotype 원-핫이 아니라 "e4 carrier 여부" 이진 변수로 별도 처리
# (data_prep.add_apoe_e4_carrier 참고).
ONEHOT_ONCE_COLS = ['sex']

# covariate 이름 -> 실제 컬럼명 매핑 (data_prep.preprocess()가 실행 시점에 자동 완성).
# AGE / education_year / APOE는 컬럼 1개짜리라 여기에 고정.
# sex, 1st_cognitive_status는 더미화 후 실제 생긴 컬럼명으로 preprocess()가 채움.
COVARIATE_COLUMN_MAP = {
    'AGE': ['AGE'],
    'education_year': ['education_year'],
    'APOE': ['APOE_e4_carrier'],
}

# 사람이 읽는 covariate 이름 (리포트/JSON용)
COVARIATE_LABELS = {
    'sex': '성별',
    'AGE': '연령',
    'education_year': '교육연수',
    'APOE': 'APOE e4 보유',
    COGNITIVE_STATUS_COL: '인지 상태(의사 진단)',
}


# ============================================================
# 2. Group x Model 정의
# ============================================================
# filter          : None이면 전체 df, 아니면 (컬럼명, [허용값들]) 튜플로 isin 필터링
# covariate_steps : "한 번에 같이 추가되는 covariate 묶음"의 리스트.
#                   M(k+1) = COMPOSITE_SCORES + covariate_steps[:k] 전부
# model_names     : 길이 = len(covariate_steps) + 1
#
# sex/AGE/education_year를 하나씩이 아니라 한 덩어리로 묶은 이유:
# 실제 적용 시 이 셋은 항상 함께 확보되므로, "sex만 있고 AGE는 없는" 모델은
# 임상적으로 쓸 일이 없음. 사다리를 실제 정보 획득 순서와 일치시킴.
#
# Application 매핑 (환자 정보가 많을수록 상위 모델):
#   cognitive_status  APOE      사용 모델
#   -----------------------------------------------------------
#   Unknown           Unknown   M2_All
#   Known             Unknown   M2_SCD_MCI  또는 M2_Dementia
#   Unknown           Known     (인지검사 -> 혈액검사 순서이므로 cognitive_status 입력 요청)
#   Known             Known     M4_SCD_MCI  또는 M3_Dementia

_DEMOGRAPHIC_STEP = ['sex', 'AGE', 'education_year']

GROUPS = {
    'All': {
        'filter': None,
        'covariate_steps': [
            _DEMOGRAPHIC_STEP,          # -> M2_All
            [COGNITIVE_STATUS_COL],     # -> M3_All  (is_MCI, is_Dementia)
            ['APOE'],                   # -> M4_All
        ],
        'model_names': ['M1_All', 'M2_All', 'M3_All', 'M4_All'],
    },
    'SCD_MCI': {
        'filter': (COGNITIVE_STATUS_COL, ['SCD', 'MCI']),
        'covariate_steps': [
            _DEMOGRAPHIC_STEP,          # -> M2_SCD_MCI
            [COGNITIVE_STATUS_COL],     # -> M3_SCD_MCI (is_MCI만 남음, is_Dementia는 상수라 자동 제거)
            ['APOE'],                   # -> M4_SCD_MCI
        ],
        'model_names': ['M1_SCD_MCI', 'M2_SCD_MCI', 'M3_SCD_MCI', 'M4_SCD_MCI'],
    },
    'Dementia': {
        'filter': (COGNITIVE_STATUS_COL, ['Dementia']),
        'covariate_steps': [
            _DEMOGRAPHIC_STEP,          # -> M2_Dementia
            ['APOE'],                   # -> M3_Dementia
        ],
        # Dementia 그룹 안에서는 1st_cognitive_status가 상수이므로 covariate로 못 씀
        'model_names': ['M1_Dementia', 'M2_Dementia', 'M3_Dementia'],
    },
}


# ============================================================
# 3. Top-K biomarker 하이라이트 규칙
# ============================================================
# 1) p < TOP_K_ALPHA 인 변수만 후보로 남긴다
# 2) 그 후보들을 |beta| = |log(OR)| 내림차순으로 정렬한다
# 3) 상위 TOP_K개를 하이라이트한다
#
# ** |OR|이 아니라 |log(OR)|로 정렬하는 이유 **
# OR은 1을 기준으로 비대칭이라 OR=2와 OR=0.5는 "크기가 같고 방향만 반대"인데
# |OR|로 재면 2가 4배 커 보임. 특히 이 데이터는 인지 z-score가 높을수록 amyloid
# 음성이라 대부분의 composite OR이 1보다 작음 -> |OR| 정렬 시 가장 강력한 마커가
# 오히려 꼴찌로 밀리고 AGE(OR~1.08) 같은 게 위로 올라오는 사고가 남.
# 회귀계수의 절댓값이 곧 로그오즈 스케일에서의 효과 크기.
TOP_K = 3
TOP_K_ALPHA = 0.05


# ============================================================
# 4. Separation(완전분리) 진단 임계값
# ============================================================
# 표본이 작은 그룹(특히 Dementia: 음성 87명 / 파라미터 22개, EPV~4.0)에서는
# quasi-separation이 일어나 계수가 폭발할 수 있음. 그 계수가 그대로 HTML에
# 배포되면 위험하므로, 아래 조건에 걸리는 행을 리포트에서 자동 플래그함.
#   - OR이 SEPARATION_OR_MAX 이상이거나 1/SEPARATION_OR_MAX 이하
#   - 95% CI 폭(상한/하한 비)이 SEPARATION_CI_RATIO_MAX 이상
# 플래그가 뜨면 해당 그룹은 composite 축소 또는 Firth 보정을 검토할 것.
SEPARATION_OR_MAX = 100.0
SEPARATION_CI_RATIO_MAX = 1e3


# ============================================================
# 5. Calibration 리포트 설정 (노트북 전용)
# ============================================================
# ** HTML/JSON에는 절대 반영하지 않음. **
# Platt scaling이나 isotonic regression을 붙이면 배포 식이 sigmoid를 두 번
# 통과하는 구조가 되어 "학습된 식 그대로 이식"이라는 목표가 깨짐.
# 여기서는 out-of-fold 예측이 잘 보정되어 있는지 "확인만" 함.
#
# 실제 임상 현장의 유병률이 표본 유병률(362/889 = 40.7%)과 다르면 절편만
# 보정하면 되고, 그 값은 JSON에서 intercept 필드 하나만 고치면 됨:
#   beta0_new = beta0 + log( (p_new/(1-p_new)) / (p_old/(1-p_old)) )
CALIBRATION_N_BINS = 10