"""
config.py
---------
치매 진단(SCD+MCI vs Dementia) 프로젝트의 "실험 정의" 전용 파일.
기존 amyloid 버전에서 바뀐 곳만 [변경] 으로 표시했습니다.

target: 1st_cognitive_status 를 이진화 (SCD, MCI -> 0 / Dementia -> 1)
"""

# ============================================================
# 0. 전역 상수
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_CV_SPLITS = 5

# [변경] target 이 amyloid -> cognitive status 이진화
TARGET = '1st_cognitive_status'
TARGET_MAP = {'SCD': 0, 'MCI': 0, 'Dementia': 1}

# [변경] 전처리 후 만들어질 y 컬럼 이름 (기존 'y_amyloid' 자리)
TARGET_Y = 'y_dementia'

# 원본 데이터에서 무조건 제외할 컬럼
ID_COLS = ['user_ID']
# [변경] amyloid 는 이번 분석에서 완전히 제외 - feature 로 섞여 들어가는 사고 방지
UNUSED_COLS = ['education_category', '졸업여부', '1st_amyloid_status']

# 인지 상태 컬럼. [변경] 이제 covariate 가 아니라 target 의 원천이며,
# split 층화(SCD/MCI/Dementia 3범주)에만 계속 사용합니다.
COGNITIVE_STATUS_COL = '1st_cognitive_status'


# ============================================================
# 1. Feature 정의
# ============================================================

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

ONEHOT_ONCE_COLS = ['sex']

COVARIATE_COLUMN_MAP = {
    'AGE': ['AGE'],
    'education_year': ['education_year'],
    'APOE': ['APOE_e4_carrier'],
}


# ============================================================
# 2. Group 정의
# ============================================================
# [변경] 인지상태가 target 이 되면서 SCD/MCI/Dementia 하위 그룹 필터는 삭제.
#        (그룹 안에서 target 이 상수가 되어 학습 자체가 불가능)
#        covariate_order 에서도 1st_cognitive_status 제거 - 정답이므로 leakage.
#        따라서 All 단일 그룹 + M1~M5 (feature 18 ~ 22개).

GROUPS = {
    'All': {
        'filter': None,
        'covariate_order': ['sex', 'AGE', 'education_year', 'APOE'],
        'model_names': ['M1', 'M2', 'M3', 'M4', 'M5'],
    },
}

MODEL_TYPES = ['LR', 'RF']


# ============================================================
# 3. RF 하이퍼파라미터 탐색 범위
# ============================================================
RF_N_ITER = 30