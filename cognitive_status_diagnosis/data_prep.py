"""
data_prep.py
------------
원본 df -> 실험에 바로 쓸 수 있는 형태로 만드는 전처리 전용 모듈.

이 파일에서 하는 일:
    1) target 매핑          (SCD/MCI -> 0, Dementia -> 1)
    2) sex 더미화            (get_dummies, drop_first=True)
    3) APOE -> e4 carrier   (6-genotype 문자열 -> e4 보유 여부 이진값)
    4) train/test 8:2 stratified split

[변경] 기존 amyloid 버전 대비:
    - add_cognitive_status_dummies() 삭제
      (is_MCI / is_Dementia 는 이제 정답 그 자체 -> leakage)
    - 1st_amyloid_status 등 UNUSED_COLS 를 실제로 drop
    - split 층화 기준이 SCD/MCI/Dementia 3범주
      (y(0/1)로 층화하면 음성군 안의 SCD:MCI 비율이 train/test 에서 틀어짐)

사용 예 (노트북에서):
    from data_prep import preprocess, split_train_test

    df, covariate_column_map = preprocess(df_composite_raw)
    df_train, df_test = split_train_test(df)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config as cfg


# ============================================================
# 1. APOE -> e4 carrier 이진 변수
# ============================================================

def add_apoe_e4_carrier(df, col='APOE', new_col='APOE_e4_carrier'):
    """
    APOE 유전형 문자열('E3/E4', 'E2/E3', 'E4/E4' ...) -> e4 allele 보유 여부(0/1).
    heterozygous(E?/E4) 와 homozygous(E4/E4) 모두 carrier=1.
    문자열에 'E4'가 포함되는지만 보므로 표기 순서(E3/E4 vs E4/E3)에 영향받지 않음.
    """
    df = df.copy()
    genotype = df[col].astype(str).str.upper()
    is_missing = df[col].isna()

    carrier = genotype.str.contains('E4').astype('float64')
    carrier[is_missing] = np.nan

    df[new_col] = carrier
    return df


# ============================================================
# 2. 최초 전처리 (전체 파이프라인 진입점, 딱 1회 실행)
# ============================================================

def preprocess(df_raw, verbose=True):
    """
    수행 순서:
        1) y_dementia 생성 (TARGET_MAP 적용, 매핑 안 되는 값 있으면 에러)
        2) UNUSED_COLS 제거 (1st_amyloid_status 포함)
        3) sex : get_dummies(drop_first=True)
        4) APOE : add_apoe_e4_carrier()
        5) COVARIATE_COLUMN_MAP 자동 완성 (sex 의 실제 더미 컬럼명을 탐지해서 채움)

    ** 원본 1st_cognitive_status 컬럼은 남겨둡니다 **
       split 층화와, 나중에 "예측확률을 SCD/MCI/Dementia 별로 나눠 보기" 에 씁니다.
       feature 리스트는 COMPOSITE_SCORES + covariate_column_map 에서만 만들어지므로
       이 컬럼이 모델에 들어갈 일은 없습니다.

    반환: df, covariate_column_map
    """
    df = df_raw.copy()

    # 1) target 매핑
    df[cfg.TARGET_Y] = df[cfg.TARGET].map(cfg.TARGET_MAP)
    if df[cfg.TARGET_Y].isna().any():
        bad = sorted(set(df.loc[df[cfg.TARGET_Y].isna(), cfg.TARGET].astype(str)))
        raise ValueError(
            f"{cfg.TARGET_Y} 매핑 후 NaN 발생 - {cfg.TARGET} 에 TARGET_MAP 에 없는 값이 있음: {bad}\n"
            f"현재 TARGET_MAP = {cfg.TARGET_MAP}"
        )
    df[cfg.TARGET_Y] = df[cfg.TARGET_Y].astype(int)

    if verbose:
        n_pos = int(df[cfg.TARGET_Y].sum())
        print(f"[target] {cfg.TARGET_Y}: 0(SCD+MCI)={len(df) - n_pos}, 1(Dementia)={n_pos} "
              f"(유병률 {n_pos / len(df) * 100:.1f}%)")
        print(df[cfg.COGNITIVE_STATUS_COL].value_counts().to_string())

    # 2) 사용하지 않는 컬럼 제거 (amyloid 포함)
    to_drop = [c for c in (cfg.ID_COLS + cfg.UNUSED_COLS) if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)
        if verbose:
            print(f"[drop] 제외한 컬럼: {to_drop}")

    covariate_column_map = dict(cfg.COVARIATE_COLUMN_MAP)

    # 3) sex 더미화 (전/후 컬럼 집합 차이로 실제 생긴 컬럼명 탐지)
    before_cols = set(df.columns)
    df = pd.get_dummies(df, columns=cfg.ONEHOT_ONCE_COLS, drop_first=True)
    new_dummy_cols = set(df.columns) - before_cols
    for c in new_dummy_cols:
        df[c] = df[c].astype('float64')
    for base_col in cfg.ONEHOT_ONCE_COLS:
        covariate_column_map[base_col] = sorted(
            c for c in new_dummy_cols if c.startswith(base_col + '_')
        )

    # 4) APOE -> e4 carrier
    df = add_apoe_e4_carrier(df, col='APOE', new_col='APOE_e4_carrier')

    return df, covariate_column_map


# ============================================================
# 3. Train / Test split (8:2)
# ============================================================

def split_train_test(df, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE, verbose=True):
    """
    전체 df 를 한 번만 8:2 로 나눔.

    [변경] stratify 기준 = 1st_cognitive_status 3범주 (SCD / MCI / Dementia).
    y(0/1)로 층화하면 음성군(SCD+MCI) 안에서 SCD:MCI 비율이 train/test 사이에
    틀어질 수 있습니다. 음성군이 이질적 혼합이라 이 비율이 성능에 직접 영향을 주므로
    3범주로 층화해서 두 집단 구성까지 맞춥니다.

    반환: df_train, df_test (원본 index 유지)
    """
    train_idx, test_idx = train_test_split(
        df.index, test_size=test_size,
        stratify=df[cfg.COGNITIVE_STATUS_COL], random_state=random_state,
    )
    df_train = df.loc[train_idx].copy()
    df_test = df.loc[test_idx].copy()

    if verbose:
        tab = pd.DataFrame({
            'train': df_train[cfg.COGNITIVE_STATUS_COL].value_counts(),
            'test': df_test[cfg.COGNITIVE_STATUS_COL].value_counts(),
        }).fillna(0).astype(int)
        print(f"[split] train={len(df_train)}, test={len(df_test)}")
        print(tab.to_string())

    return df_train, df_test