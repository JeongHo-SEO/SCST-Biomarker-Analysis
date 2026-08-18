"""
stats_tests.py
--------------
"통계적으로 유의미한가?"에 답하는 가설검정 함수 전용 모듈.
모델을 학습/예측하는 로직(modeling.py)과 분리해서, "이 파일에 있는 함수는
전부 p-value를 반환한다"는 걸 이름만 보고 알 수 있게 함.

포함된 검정:
    1) lr_joint_test : 더미화되어 여러 컬럼으로 나뉜 covariate 하나가
                       "모델에 전체적으로 기여하는가" (Likelihood Ratio Test)
    2) delong_test   : 같은 피험자에 대한 두 모델의 AUC가 통계적으로 다른가 (DeLong)

** 이전 버전과의 결정적 차이 **
LASSO를 제거했으므로 post-selection inference 문제가 사라졌습니다.
이전 코드에는 "LASSO로 선택된 feature 위에서 검정하는 것이므로 확증적
(confirmatory) p-value가 아니라 exploratory 참고용"이라는 경고가 붙어 있었는데,
feature가 사전 지정되고 규제가 없는 지금은 그 경고가 필요 없습니다.
LRT와 Wald p-value 모두 확증적으로 해석 가능합니다.

또한 LRT를 fold별로 5번 하고 평균내던 방식도 폐기했습니다. 검정통계량이나
p-value의 평균은 어떤 가설검정의 결과도 아니기 때문입니다.
검정은 **전체 표본으로 적합한 최종 모델 위에서 단 한 번** 수행합니다.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

import config as cfg
from modeling import _fit_logit, drop_degenerate_features


# ============================================================
# 1. Likelihood Ratio Test (더미화된 covariate의 "전체 효과" 검정)
# ============================================================

def lr_joint_test(X, y, features, group_map, label='', verbose=True):
    """
    귀무가설(H0): group_map에 속한 그룹의 더미 계수가 전부 0이다
                  (= 그 covariate 전체가 모델에 기여하지 않는다)

    검정통계량: LR_stat = 2 * (logL_full - logL_reduced) ~ chi2(df = 그 그룹의 컬럼 수)

    **전체 표본에서 단 한 번** 수행. full 모델과 reduced 모델을 각각 적합해 비교.

    왜 필요한가: 1st_cognitive_status처럼 더미 2개(is_MCI, is_Dementia)로 나뉜
    변수는 개별 Wald p-value 두 개만 봐서는 "이 변수 전체가 필요한가"에 답할 수
    없음. 예를 들어 둘 다 p=0.08이어도 함께 넣으면 유의할 수 있음.

    group_map : {'1st_cognitive_status': ['is_MCI', 'is_Dementia'], ...}
                컬럼이 1개뿐인 covariate(AGE, APOE_e4_carrier 등)는 Wald test와
                결과가 사실상 같으므로 넣을 필요 없음.

    반환: DataFrame (group, n_columns_tested, LR_stat, df, LRT_p_value)
    """
    features = drop_degenerate_features(X[features], verbose=False)

    model_full = _fit_logit(X, y, features, tag=f'{label} LRT full', verbose=verbose)
    if model_full is None:
        return pd.DataFrame(columns=['group', 'n_columns_tested', 'LR_stat', 'df', 'LRT_p_value'])

    records = []
    for group_name, group_cols in group_map.items():
        cols_in_model = [c for c in group_cols if c in features]
        if len(cols_in_model) == 0:
            continue  # 이 subgroup에서 상수라 제외된 경우 (예: SCD_MCI의 is_Dementia)

        reduced_feats = [f for f in features if f not in cols_in_model]
        model_reduced = _fit_logit(X, y, reduced_feats,
                                   tag=f'{label} LRT reduced/{group_name}', verbose=verbose)
        if model_reduced is None:
            continue

        df_ = len(cols_in_model)
        lr_stat = max(2 * (model_full.llf - model_reduced.llf), 0.0)  # 수치오차로 인한 미세 음수 방지
        records.append({
            'group': group_name,
            'n_columns_tested': df_,
            'columns': ', '.join(cols_in_model),
            'LR_stat': lr_stat,
            'df': df_,
            'LRT_p_value': float(chi2.sf(lr_stat, df_)),
        })

    if len(records) == 0:
        return pd.DataFrame(columns=['group', 'n_columns_tested', 'LR_stat', 'df', 'LRT_p_value'])

    return pd.DataFrame(records).sort_values('LR_stat', ascending=False).reset_index(drop=True)


# ============================================================
# 2. DeLong test (같은 피험자에 대한 두 모델의 AUC 비교)
# ============================================================
# 참고: Sun, X. and Xu, W. (2014), "Fast Implementation of DeLong's Algorithm
# for Comparing the Areas Under Correlated Receiver Operating Characteristic
# Curves." IEEE Signal Processing Letters.
#
# hold-out test set이 없어졌으므로, 비교 대상은 **pooled out-of-fold 예측**임.
# 같은 group 안에서 M1~M4는 동일한 y와 표본을 쓰므로 StratifiedKFold 분할이
# 완전히 같고, modeling.pooled_predictions()가 index 순으로 정렬해 반환하므로
# "같은 피험자, 같은 순서"가 보장됨.

def _compute_midrank(x):
    """DeLong 계산용 mid-rank (동점 값 처리 포함)."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted, m):
    """predictions_sorted: shape (2, n). 양성(1)이 앞 m개로 정렬되어 있어야 함."""
    n = predictions_sorted.shape[1] - m
    pos, neg = predictions_sorted[:, :m], predictions_sorted[:, m:]
    k = predictions_sorted.shape[0]

    tx = np.vstack([_compute_midrank(pos[r]) for r in range(k)])
    ty = np.vstack([_compute_midrank(neg[r]) for r in range(k)])
    tz = np.vstack([_compute_midrank(predictions_sorted[r]) for r in range(k)])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    delongcov = np.cov(v01) / m + np.cov(v10) / n
    return aucs, delongcov


def delong_test(y_true, y_prob_1, y_prob_2, label_1='model_1', label_2='model_2'):
    """
    같은 피험자·같은 순서의 두 예측확률을 받아 AUC 차이가 유의한지 검정.
    H0: AUC_1 == AUC_2

    y_true, y_prob_1, y_prob_2 : pandas Series를 넣으면 index로 자동 정렬해서
                                  순서가 어긋나는 사고를 막음.

    반환: dict(auc_{label_1}, auc_{label_2}, auc_diff, z_stat, p_value)
    """
    # Series면 index 기준으로 교집합 정렬 (순서 어긋남 방지)
    if isinstance(y_prob_1, pd.Series) and isinstance(y_prob_2, pd.Series):
        common = y_prob_1.index.intersection(y_prob_2.index)
        if len(common) != len(y_prob_1) or len(common) != len(y_prob_2):
            raise ValueError(
                "두 모델의 예측 대상 피험자가 다릅니다. DeLong 검정은 동일 표본을 "
                "요구합니다 (같은 Group의 모델끼리만 비교하세요)."
            )
        y_prob_1 = y_prob_1.loc[common]
        y_prob_2 = y_prob_2.loc[common]
        if isinstance(y_true, pd.Series):
            y_true = y_true.loc[common]

    y_true = np.asarray(y_true)
    order = np.argsort(-y_true, kind='mergesort')  # 양성(1)이 앞으로
    m = int(y_true[order].sum())

    preds = np.vstack([np.asarray(y_prob_1)[order], np.asarray(y_prob_2)[order]])
    aucs, cov = _fast_delong(preds, m)

    auc_diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]

    if var <= 0:
        z, p = np.nan, np.nan
    else:
        z = auc_diff / np.sqrt(var)
        p = 2 * norm.sf(abs(z))

    return {
        f'auc_{label_1}': float(aucs[0]),
        f'auc_{label_2}': float(aucs[1]),
        'auc_diff': float(auc_diff),
        'z_stat': float(z) if np.isfinite(z) else np.nan,
        'p_value': float(p) if np.isfinite(p) else np.nan,
    }