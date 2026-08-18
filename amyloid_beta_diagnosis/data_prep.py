"""
data_prep.py
------------
원본 df를 받아 "실험에 바로 쓸 수 있는 형태"로 만드는 전처리 전용 모듈.

이 파일에서 하는 일은 딱 4가지:
    1) target 매핑          (음성/양성 -> 0/1)
    2) sex 더미화            (get_dummies, drop_first=True)
    3) APOE -> e4 carrier   (6-genotype 문자열을 "e4 allele 보유 여부" 이진값으로 축소)
    4) 1st_cognitive_status 더미화 (reference='SCD' 고정, is_MCI / is_Dementia 생성)
    5) train/test 8:2 stratified split

전처리 로직 자체(어떤 값을 어떻게 바꾸는지)는 여기, "무엇을 covariate로 쓸지"는
config.py, "이 covariate들로 모델을 어떻게 학습할지"는 modeling.py 로 역할을 나눴습니다.

사용 예 (노트북에서):
    from data_prep import preprocess, split_train_test

    df, covariate_column_map = preprocess(df_composite_raw)
    df_train, df_test = split_train_test(df)
"""

import pandas as pd
from sklearn.model_selection import train_test_split

import config as cfg


# ============================================================
# 1. APOE -> e4 carrier 이진 변수
# ============================================================

def add_apoe_e4_carrier(df, col='APOE', new_col='APOE_e4_carrier'):
    """
    APOE 유전형 문자열(예: 'E3/E4', 'E2/E3', 'E4/E4' ...)을
    "e4 allele를 하나라도 갖고 있는가(carrier)" 이진 변수로 변환.

    E4/E?, E?/E4 (heterozygous) 와 E4/E4 (homozygous) 모두 carrier=1.
    문자열에 'E4'가 포함되는지만 보는 단순 매칭이라, 표기 순서(E3/E4 vs E4/E3)에
    영향받지 않음. 대소문자 섞여 있을 수 있어 대문자로 통일 후 비교.

    결측치(NaN)는 carrier 여부를 알 수 없으므로 그대로 NaN 유지
    (모델 fit 전에 결측 처리 방식을 별도로 정해야 함 - 이 함수 책임 밖).

    반환: df에 new_col(0/1, 결측은 NaN)이 추가된 복사본
    """
    df = df.copy()
    genotype = df[col].astype(str).str.upper()
    is_missing = df[col].isna()

    carrier = genotype.str.contains('E4').astype('float')  # 0.0 / 1.0
    carrier[is_missing] = pd.NA

    df[new_col] = carrier
    return df


# ============================================================
# 2. 1st_cognitive_status -> is_MCI, is_Dementia (reference 고정 더미)
# ============================================================

def add_cognitive_status_dummies(df, col=cfg.COGNITIVE_STATUS_COL,
                                  reference=cfg.COGNITIVE_STATUS_REFERENCE):
    """
    1st_cognitive_status(SCD/MCI/Dementia) -> is_MCI, is_Dementia 더미 생성.
    reference(SCD)는 두 더미가 모두 0인 상태로 흡수됨 (표준 dummy coding).

    원본 1st_cognitive_status 컬럼은 그룹 필터링(All/SCD/MCI/Dementia)에
    계속 써야 하므로 삭제하지 않고 그대로 둠.
    """
    cats = [c for c in df[col].unique() if c != reference]
    dummies = pd.get_dummies(df[col]).reindex(columns=[reference] + cats, fill_value=0)
    dummies = dummies.drop(columns=reference).add_prefix('is_')
    return pd.concat([df.copy(), dummies], axis=1)


# ============================================================
# 3. 최초 전처리 (전체 파이프라인 진입점, 딱 1회 실행)
# ============================================================

def preprocess(df_raw):
    """
    config.py에 정의된 규칙대로 원본 df를 1회 전처리.

    수행 순서:
        1) y_amyloid 생성 (TARGET_MAP 적용, 매핑 안 되는 값 있으면 에러)
        2) sex : get_dummies(drop_first=True)
        3) APOE : add_apoe_e4_carrier() 로 이진 변수 생성
        4) 1st_cognitive_status : add_cognitive_status_dummies() 적용
        5) COVARIATE_COLUMN_MAP 자동 완성 (sex, 1st_cognitive_status의
           실제 더미 컬럼명을 "전/후 컬럼 집합 차이"로 탐지해서 채움.
           AGE / education_year / APOE는 컬럼 1개짜리라 config.py에 이미 고정되어 있음)

    반환:
        df                    : 전처리 완료된 DataFrame (df_raw는 변경하지 않음)
        covariate_column_map  : dict, 완성된 컬럼 매핑
    """
    df = df_raw.copy()

    # 1) target 매핑
    df['y_amyloid'] = df[cfg.TARGET].map(cfg.TARGET_MAP)
    if df['y_amyloid'].isna().any():
        n_na = df['y_amyloid'].isna().sum()
        raise ValueError(
            f"y_amyloid 매핑 후 NaN {n_na}개 발생 - {cfg.TARGET}의 값이 "
            f"TARGET_MAP={cfg.TARGET_MAP}에 없는 카테고리를 포함하고 있는지 확인 필요"
        )

    covariate_column_map = dict(cfg.COVARIATE_COLUMN_MAP)  # AGE/education_year/APOE 복사

    # 2) sex 더미화 (전/후 컬럼 집합 차이로 실제 생긴 컬럼명 탐지)
    before_cols = set(df.columns)
    df = pd.get_dummies(df, columns=cfg.ONEHOT_ONCE_COLS, drop_first=True)
    new_dummy_cols = set(df.columns) - before_cols
    for base_col in cfg.ONEHOT_ONCE_COLS:
        covariate_column_map[base_col] = sorted(
            c for c in new_dummy_cols if c.startswith(base_col + '_')
        )

    # 3) APOE -> e4 carrier (컬럼 1개, config.py에 이미 매핑되어 있어 자동완성 불필요)
    df = add_apoe_e4_carrier(df, col='APOE', new_col='APOE_e4_carrier')

    # 4) cognitive_status 더미화
    before_cols = set(df.columns)
    df = add_cognitive_status_dummies(df)
    new_dummy_cols = set(df.columns) - before_cols
    covariate_column_map[cfg.COGNITIVE_STATUS_COL] = sorted(new_dummy_cols)

    return df, covariate_column_map


# ============================================================
# 4. Train / Test split (8:2, 전체 실험이 공유하는 단일 split)
# ============================================================

def split_train_test(df, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE):
    """
    전체 df를 한 번만 8:2로 나눔. All/SCD/MCI/Dementia 그룹별로 따로 split하지 않고
    여기서 나온 train_idx/test_idx를 모든 그룹이 공유 -> "동일한 피험자는 항상
    동일한 split(train or test)에 속한다"는 일관성 보장.

    stratify 기준: 1st_cognitive_status x y_amyloid 결합 컬럼.
    -> 각 (인지상태, amyloid 양성여부) 조합의 비율이 train/test에서 최대한 유지됨
       (예: MCI-양성 비율이 train/test에서 크게 다르지 않도록).

    반환: df_train, df_test (둘 다 df.copy() 기반, 원본 index 유지)
    """
    strata = (
        df[cfg.COGNITIVE_STATUS_COL].astype(str) + '_' + df['y_amyloid'].astype(str)
    )

    train_idx, test_idx = train_test_split(
        df.index, test_size=test_size, stratify=strata, random_state=random_state,
    )
    df_train = df.loc[train_idx].copy()
    df_test = df.loc[test_idx].copy()
    return df_train, df_test