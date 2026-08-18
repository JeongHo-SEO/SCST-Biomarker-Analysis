"""
config.py
---------
Alzheimer's amyloid PET 양성 예측 프로젝트의 "실험 정의" 전용 파일.

이 파일에는 실행 로직이 없습니다. Group(All/SCD/MCI/Dementia) x Model(M1~M6)
조합을 어떻게 정의하는지, 어떤 feature를 쓰는지 등 "설정값"만 모아둡니다.
실제 실행은 run_experiments.py, 함수 구현은 data_prep.py / modeling.py /
stats_tests.py / reporting.py에 있습니다.

target: 1st_amyloid_status (양성 vs 음성)
"""

# ============================================================
# 0. 전역 상수
# ============================================================

# 전체 파이프라인 공통 seed. 여기 값을 바꾸면 모든 모듈에 일괄 반영됨
# (다른 모듈은 전부 `from config import RANDOM_STATE`로 이 값을 가져다 씀).
RANDOM_STATE = 42

# train : test 비율. 8:2 고정 split (Group/Model 전체가 동일한 split을 공유).
TEST_SIZE = 0.2

# 5-fold CV (1.1.1, 1.1.2 단계에서 공통으로 사용)
N_CV_SPLITS = 5

TARGET = '1st_amyloid_status'

# target 매핑: data_prep.preprocess()에서 한 번만 적용
#   df['y_amyloid'] = df[TARGET].map(TARGET_MAP)
TARGET_MAP = {'음성': 0, '양성': 1}

# 원본 데이터에서 무조건 제외할 컬럼
ID_COLS = ['user_ID']
UNUSED_COLS = ['education_category', '졸업여부']

# 인지 상태 컬럼 (그룹 필터링 + covariate 더미 생성 양쪽에 사용)
COGNITIVE_STATUS_COL = '1st_cognitive_status'
COGNITIVE_STATUS_REFERENCE = 'SCD'  # is_MCI, is_Dementia 더미의 기준 카테고리

# ============================================================
# 1. Feature 정의
# ============================================================

# composite_scores : 모든 Group x Model 공통으로 항상 포함되는 18개 신경심리검사 z-score
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
# APOE는 6-genotype 원-핫이 아니라 "e4 carrier 여부" 이진 변수로 별도 처리하므로
# (data_prep.add_apoe_e4_carrier 참고) 여기 포함하지 않음.
ONEHOT_ONCE_COLS = ['sex']

# covariate_order -> 실제 컬럼명 매핑 (data_prep.preprocess()가 실행 시점에 자동 완성).
# AGE, education_year, APOE_e4_carrier는 더미화 없이 컬럼 1개짜리로 그대로 매핑됨.
# sex, 1st_cognitive_status는 더미화 후 실제 생긴 컬럼명으로 preprocess()가 채워 넣음.
COVARIATE_COLUMN_MAP = {
    'AGE': ['AGE'],
    'education_year': ['education_year'],
    'APOE': ['APOE_e4_carrier'],
}


# ============================================================
# 2. Group 정의
# ============================================================
# filter        : None이면 전체 df 사용, 아니면 (컬럼명, 값) 튜플로 필터링
# covariate_order: 누적 추가 순서 (원래 이름 기준. 실제 컬럼은 COVARIATE_COLUMN_MAP 참조)
# model_names    : 슬라이싱 결과에 매길 이름 (길이 = len(covariate_order) + 1)
#
# APOE가 이제 컬럼 1개(carrier)로 줄어서, All 그룹 M6도 "18 + 5(carrier 포함 covariate 5개)"
# = 23개 feature가 됨 (예전 6-genotype 방식일 때의 28개에서 축소).

GROUPS = {
    'All': {
        'filter': None,
        'covariate_order': ['1st_cognitive_status', 'sex', 'AGE', 'education_year', 'APOE'],
        'model_names': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6'],
    },
    'SCD': {
        'filter': (COGNITIVE_STATUS_COL, 'SCD'),
        'covariate_order': ['sex', 'AGE', 'education_year', 'APOE'],
        'model_names': ['M1_SCD', 'M2_SCD', 'M3_SCD', 'M4_SCD', 'M5_SCD'],
    },
    'MCI': {
        'filter': (COGNITIVE_STATUS_COL, 'MCI'),
        'covariate_order': ['sex', 'AGE', 'education_year', 'APOE'],
        'model_names': ['M1_MCI', 'M2_MCI', 'M3_MCI', 'M4_MCI', 'M5_MCI'],
    },
    'Dementia': {
        'filter': (COGNITIVE_STATUS_COL, 'Dementia'),
        'covariate_order': ['sex', 'AGE', 'education_year', 'APOE'],
        'model_names': ['M1_Dementia', 'M2_Dementia', 'M3_Dementia', 'M4_Dementia', 'M5_Dementia'],
    },
}

MODEL_TYPES = ['LR', 'RF']


# ============================================================
# 3. RF 하이퍼파라미터 탐색 범위 (RandomizedSearchCV용)
# ============================================================
# scipy.stats 분포 객체를 씀 -> 매 iteration마다 후보값을 "연속적으로" 무작위 샘플링.
# GridSearchCV처럼 이산 후보를 손으로 일일이 지정할 필요가 없음.
#
# max_depth, min_samples_leaf 등의 상한을 너무 크게 잡지 않은 이유:
# 표본이 889개(그룹별로는 훨씬 적음) 수준이라, 지나치게 깊은 트리는 쉽게 과적합됨.
RF_N_ITER = 30  # RandomizedSearchCV 시도 횟수 (가볍게 튜닝 - 필요시 늘리면 됨)