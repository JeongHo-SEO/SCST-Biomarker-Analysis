"""
data_prep.py
------------
원본 df를 받아 "실험에 바로 쓸 수 있는 형태"로 만드는 전처리 전용 모듈.

이 파일에서 하는 일은 4가지:
    1) target 매핑          (음성/양성 -> 0/1)
    2) sex 더미화            (get_dummies, drop_first=True)
    3) APOE -> e4 carrier   (6-genotype 문자열을 "e4 allele 보유 여부" 이진값으로 축소)
    4) 1st_cognitive_status 더미화 (reference='SCD' 고정, is_MCI / is_Dementia 생성)

** train/test split은 더 이상 하지 않습니다. **
전체 N=889로 최종 모델을 refit하고, 성능은 5-fold CV의 out-of-fold 예측으로만
추정합니다 (config.py의 N_CV_SPLITS 주석 참고). LASSO를 제거하면서 "데이터를 보고
결정하는 단계"가 하나도 없어졌기 때문에, 별도 hold-out을 떼어 표본을 줄일 이유가
없어졌습니다.

사용 예 (노트북에서):
    from data_prep import preprocess

    df, covariate_column_map = preprocess(df_composite_raw)
"""

import numpy as np
import pandas as pd

import config as cfg


# ============================================================
# 1. APOE -> e4 carrier 이진 변수
# ============================================================

def add_apoe_e4_carrier(df, col='APOE', new_col='APOE_e4_carrier'):
    """
    APOE 유전형 문자열(예: 'E3/E4', 'E2/E3', 'E4/E4' ...)을
    "e4 allele를 하나라도 갖고 있는가(carrier)" 이진 변수로 변환.

    E4/E?, E?/E4 (heterozygous) 와 E4/E4 (homozygous) 모두 carrier=1.
    문자열에 'E4'가 포함되는지만 보는 단순 매칭이라 표기 순서(E3/E4 vs E4/E3)에
    영향받지 않음. 대소문자 섞여 있을 수 있어 대문자로 통일 후 비교.

    결측치는 np.nan으로 유지 (pd.NA를 쓰면 float Series의 dtype이 object로 튀어
    statsmodels에 넘길 때 문제가 생김). 다만 이 프로젝트 데이터에는 APOE 결측이
    없으므로 실제로는 발생하지 않아야 하며, 발생 시 preprocess()에서 경고함.

    반환: df에 new_col(0.0/1.0, 결측은 np.nan)이 추가된 복사본
    """
    df = df.copy()
    genotype = df[col].astype(str).str.upper()
    is_missing = df[col].isna()

    carrier = genotype.str.contains('E4').astype(float)  # 0.0 / 1.0
    carrier[is_missing] = np.nan

    df[new_col] = carrier
    return df


# ============================================================
# 2. 1st_cognitive_status -> is_MCI, is_Dementia (reference 고정 더미)
# ============================================================

def add_cognitive_status_dummies(df, col=cfg.COGNITIVE_STATUS_COL,
                                  reference=cfg.COGNITIVE_STATUS_REFERENCE,
                                  categories=None):
    """
    1st_cognitive_status(SCD/MCI/Dementia) -> is_MCI, is_Dementia 더미 생성.
    reference(SCD)는 두 더미가 모두 0인 상태로 흡수됨 (표준 dummy coding).

    카테고리 순서를 config.COGNITIVE_STATUS_CATEGORIES로 "고정"함.
    (df[col].unique()는 데이터 등장 순서에 의존해서, 데이터가 조금만 바뀌어도
     컬럼 순서가 달라짐. 배포용 계수를 다루는 파이프라인에서는 재현성이 중요.)

    원본 1st_cognitive_status 컬럼은 그룹 필터링에 계속 써야 하므로 삭제하지 않음.

    dtype은 float으로 캐스팅함 - pandas 2.x의 get_dummies는 bool을 반환하는데,
    bool 컬럼을 statsmodels에 넘기면 설계행렬 구성에서 문제가 생길 수 있음.
    """
    categories = categories or cfg.COGNITIVE_STATUS_CATEGORIES

    unseen = set(df[col].dropna().unique()) - set(categories) - {reference}
    if unseen:
        raise ValueError(
            f"{col}에 예상하지 못한 값이 있습니다: {sorted(unseen)}. "
            f"config.COGNITIVE_STATUS_CATEGORIES({categories})와 "
            f"COGNITIVE_STATUS_REFERENCE({reference})를 확인하세요."
        )

    out = df.copy()
    for cat in categories:
        out[f'is_{cat}'] = (df[col] == cat).astype(float)
    return out


# ============================================================
# 3. 최초 전처리 (전체 파이프라인 진입점, 딱 1회 실행)
# ============================================================

def preprocess(df_raw, verbose=True):
    """
    config.py에 정의된 규칙대로 원본 df를 1회 전처리.

    수행 순서:
        1) y_amyloid 생성 (TARGET_MAP 적용, 매핑 안 되는 값 있으면 에러)
        2) sex : get_dummies(drop_first=True) -> float 캐스팅
        3) APOE : add_apoe_e4_carrier()
        4) 1st_cognitive_status : add_cognitive_status_dummies()
        5) COVARIATE_COLUMN_MAP 자동 완성
        6) 결측/dtype 점검 (statsmodels는 NaN이 있으면 그냥 에러가 남)

    반환:
        df                    : 전처리 완료된 DataFrame (df_raw는 변경하지 않음)
        covariate_column_map  : dict, 완성된 컬럼 매핑
    """
    df = df_raw.copy()

    # 1) target 매핑
    df['y_amyloid'] = df[cfg.TARGET].map(cfg.TARGET_MAP)
    if df['y_amyloid'].isna().any():
        n_na = int(df['y_amyloid'].isna().sum())
        raise ValueError(
            f"y_amyloid 매핑 후 NaN {n_na}개 발생 - {cfg.TARGET}의 값이 "
            f"TARGET_MAP={cfg.TARGET_MAP}에 없는 카테고리를 포함하고 있는지 확인 필요"
        )
    df['y_amyloid'] = df['y_amyloid'].astype(int)

    covariate_column_map = {k: list(v) for k, v in cfg.COVARIATE_COLUMN_MAP.items()}

    # 2) sex 더미화 (전/후 컬럼 집합 차이로 실제 생긴 컬럼명 탐지)
    before_cols = set(df.columns)
    df = pd.get_dummies(df, columns=cfg.ONEHOT_ONCE_COLS, drop_first=True)
    new_dummy_cols = set(df.columns) - before_cols
    for base_col in cfg.ONEHOT_ONCE_COLS:
        cols = sorted(c for c in new_dummy_cols if c.startswith(base_col + '_'))
        df[cols] = df[cols].astype(float)  # pandas 2.x는 bool을 반환함
        covariate_column_map[base_col] = cols

    # 3) APOE -> e4 carrier
    df = add_apoe_e4_carrier(df, col='APOE', new_col='APOE_e4_carrier')

    # 4) cognitive_status 더미화 (컬럼명/순서는 config에서 고정)
    df = add_cognitive_status_dummies(df)
    covariate_column_map[cfg.COGNITIVE_STATUS_COL] = [
        f'is_{c}' for c in cfg.COGNITIVE_STATUS_CATEGORIES
    ]

    # 5) 결측 점검 - 모델에 실제로 들어갈 컬럼만 확인
    model_cols = list(cfg.COMPOSITE_SCORES)
    for cols in covariate_column_map.values():
        model_cols.extend(cols)
    model_cols = [c for c in dict.fromkeys(model_cols) if c in df.columns]

    n_missing = df[model_cols].isna().sum()
    n_missing = n_missing[n_missing > 0]
    if len(n_missing) > 0:
        raise ValueError(
            "모델에 들어갈 컬럼에 결측치가 있습니다. statsmodels는 NaN을 처리하지 "
            f"못하므로 먼저 해결해야 합니다:\n{n_missing.to_string()}"
        )

    df[model_cols] = df[model_cols].astype(float)

    if verbose:
        n = len(df)
        n_pos = int(df['y_amyloid'].sum())
        print(f"[preprocess] n={n}, 양성={n_pos} ({n_pos / n:.1%}), 음성={n - n_pos}")
        print(f"[preprocess] 모델 투입 가능 컬럼 {len(model_cols)}개, 결측 없음")
        print(f"[preprocess] covariate_column_map = {covariate_column_map}")

    return df, covariate_column_map


# ============================================================
# 4. Group 필터링
# ============================================================

def filter_group(df, group_filter):
    """
    config.GROUPS[*]['filter']를 받아 해당 subgroup만 잘라냄.
    None이면 전체를 그대로 반환.

    filter 형식: (컬럼명, [허용값들])  예) ('1st_cognitive_status', ['SCD', 'MCI'])
    """
    if group_filter is None:
        return df.copy()

    col, allowed = group_filter
    if isinstance(allowed, str):
        allowed = [allowed]
    return df[df[col].isin(allowed)].copy()