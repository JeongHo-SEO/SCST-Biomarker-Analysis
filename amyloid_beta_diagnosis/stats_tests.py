"""
stats_tests.py
--------------
"통계적으로 유의미한가?"에 답하는 가설검정 함수 전용 모듈.
모델을 학습/예측하는 로직(modeling.py)과 분리해서, "이 파일에 있는 함수는
전부 p-value를 반환한다"는 걸 이름만 보고 알 수 있게 함.

포함된 검정:
    1) lr_joint_test  : (LR) 더미화되어 여러 컬럼으로 나뉜 covariate 하나가
                         "모델에 전체적으로 기여하는가" (Likelihood Ratio Test)
    2) delong_test    : (LR/RF 공통) 같은 test set에서 두 모델의 AUC가
                         통계적으로 다른가 (DeLong method)
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2, norm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import config as cfg
from modeling import lasso_select_features

RANDOM_STATE = cfg.RANDOM_STATE


# ============================================================
# 1. Likelihood Ratio Test (LR 전용, 더미화된 covariate의 "전체 효과" 검정)
# ============================================================

def lr_joint_test(X, y, group_map, use_lasso=True,
                   n_splits=cfg.N_CV_SPLITS, random_state=RANDOM_STATE):
    """
    귀무가설(H0): group_map에 속한 그룹의 더미 계수가 전부 0이다
                  (= 그 covariate 전체가 모델에 기여하지 않는다)
    검정통계량: LR_stat = 2 * (logL_full - logL_reduced) ~ chi2(df = 그 fold에서
                실제 선택된 그룹 내 컬럼 수)

    modeling.run_cv_pipeline()과 동일한 StratifiedKFold(같은 random_state)로
    fold를 재현해서, 매 fold의 train_fold 위에서 full/reduced 모델을 각각 적합.

    group_map : {'1st_cognitive_status': ['is_MCI', 'is_Dementia'], ...}
                (더미화되어 컬럼이 2개 이상인 covariate만 넣으면 됨 -
                 컬럼 1개짜리는 이 검정 없이도 Wald test로 충분함)

    반환:
        long_df    : fold x group 별 LR_stat, df, LRT_p_value
                      (그룹 컬럼이 해당 fold에서 LASSO 선택 시 전부 탈락하면 제외 -
                       "이미 계수 0으로 빠진 것"이라 검정 자체가 성립 안 함)
        summary_df : group별 요약 (검정된 fold 수, LR_stat/p-value 평균, p<0.05 비율)

    ** post-selection inference 주의: LASSO로 선택된 feature 위에서 검정하는 것이므로
       확증적(confirmatory) p-value가 아니라 exploratory 참고용임. **
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    records = []

    for fold_i, (train_idx, _) in enumerate(skf.split(X, y), start=1):
        X_tr, y_tr = X.iloc[train_idx].copy(), y.iloc[train_idx]

        feats = lasso_select_features(X_tr, y_tr, random_state) if use_lasso else list(X_tr.columns)

        for group_name, group_cols in group_map.items():
            cols_in_fold = [c for c in group_cols if c in feats]
            if len(cols_in_fold) == 0:
                continue  # 이 fold에서 그룹이 통째로 LASSO 탈락 -> 검정 불가

            reduced_feats = [f for f in feats if f not in cols_in_fold]

            try:
                model_full = _fit_logit(X_tr, y_tr, feats)
                model_reduced = _fit_logit(X_tr, y_tr, reduced_feats)

                if model_full is None or model_reduced is None:
                    print(f"  [Fold {fold_i} / {group_name}] Logit 수렴 실패 - 이 검정 제외")
                    continue
            except np.linalg.LinAlgError:
                print(f"  [Fold {fold_i} / {group_name}] singular Hessian - 이 검정 제외")
                continue

            df = len(cols_in_fold)
            lr_stat = max(2 * (model_full.llf - model_reduced.llf), 0)  # 수치오차로 미세 음수 방지
            p_value = chi2.sf(lr_stat, df)

            records.append({
                'fold': fold_i, 'group': group_name, 'n_columns_tested': df,
                'LR_stat': lr_stat, 'df': df, 'LRT_p_value': p_value,
            })

    long_df = pd.DataFrame(records)
    if len(long_df) == 0:
        print("모든 fold에서 검정 가능한 그룹이 없습니다.")
        empty_summary = pd.DataFrame(columns=[
            'group', 'n_folds_tested', 'LR_stat_mean', 'LRT_p_value_mean', 'prop_folds_p<0.05'
        ])
        return long_df, empty_summary

    summary_df = (
        long_df.groupby('group')
        .agg(n_folds_tested=('fold', 'count'),
             LR_stat_mean=('LR_stat', 'mean'),
             LRT_p_value_mean=('LRT_p_value', 'mean'))
        .reset_index()
    )
    prop_sig = long_df.groupby('group')['LRT_p_value'].apply(lambda s: (s < 0.05).mean()).rename('prop_folds_p<0.05')
    summary_df = summary_df.merge(prop_sig, on='group').sort_values(
        'LR_stat_mean', ascending=False
    ).reset_index(drop=True)

    return long_df, summary_df


def _fit_logit(X_tr, y_tr, feats):
    """intercept-only(빈 feats) 케이스까지 처리하는 Logit fit 헬퍼. 실패 시 None."""
    if len(feats) == 0:
        X_c = np.ones((len(X_tr), 1))
    else:
        scaler = StandardScaler().fit(X_tr[feats])
        X_c = sm.add_constant(scaler.transform(X_tr[feats]), has_constant='add')

    model = sm.Logit(y_tr.values, X_c).fit(disp=0, maxiter=200)
    return model if model.mle_retvals.get('converged', True) else None


# ============================================================
# 2. DeLong test (같은 test set에서 두 모델의 AUC 비교)
# ============================================================
# 참고: Sun, X. and Xu, W. (2014), "Fast Implementation of DeLong's Algorithm
# for Comparing the Areas Under Correlated Receiver Operating Characteristic
# Curves." IEEE Signal Processing Letters.

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
    같은 test set(=같은 피험자, 같은 순서)에서 나온 두 모델의 예측확률을 받아
    AUC 차이가 통계적으로 유의미한지 검정 (예: M1_baseline vs M6_full covariate).

    H0: AUC_1 == AUC_2

    y_true             : 실제 라벨 (1차원, 0/1)
    y_prob_1, y_prob_2  : 반드시 동일한 y_true, 동일한 피험자 순서에 대한 예측확률
                          (예: modeling.predict_proba()를 같은 X_test에 대해
                          두 번 - 모델1, 모델2 - 호출한 결과)

    반환: dict(auc_{label_1}, auc_{label_2}, auc_diff, z_stat, p_value)
    """
    y_true = np.asarray(y_true)
    order = np.argsort(-y_true, kind='mergesort')  # 양성(1)이 앞으로 오도록 정렬
    m = int(y_true[order].sum())  # 양성 개수

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
        f'auc_{label_1}': aucs[0], f'auc_{label_2}': aucs[1],
        'auc_diff': auc_diff, 'z_stat': z, 'p_value': p,
    }