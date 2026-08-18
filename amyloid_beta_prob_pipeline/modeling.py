"""
modeling.py
-----------
"모델을 어떻게 학습/예측하는가"에 관한 함수 전부.
통계적 유의성 검정(LRT, DeLong)은 stats_tests.py, 성능 리포트/그래프는
reporting.py로 분리했으니 그쪽을 참고.

전체 흐름 (Group x Model 하나에 대해):
    1) run_cv_pipeline()   : 5-fold CV -> out-of-fold(OOF) 예측 수집 (성능 추정 전용)
    2) youden_threshold()  : pooled OOF 예측에서 최적 threshold 계산
    3) fit_final_lr()      : 전체 표본으로 최종 모델 1회 적합 (OR/CI/p-value + 배포용 계수)
    4) predict_proba()     : 최종 모델로 임의의 X에 대한 확률 계산

--- 이전 버전에서 삭제된 것 ------------------------------------------------
  RandomForest 전부, LASSO 변수선택, SHAP, grouped permutation importance,
  RF 하이퍼파라미터 탐색, StandardScaler
----------------------------------------------------------------------------

** StandardScaler를 쓰지 않는 이유 (중요) **
  1) composite score는 이미 demographic-stratified z-score라 이중 표준화가 됨.
     StandardScaler를 걸면 "규준 대비 z"가 "이 표본 대비 z"로 바뀌어 OR 해석이
     달라짐.
  2) 스케일링을 안 하면 계수가 원 스케일이라 그대로 HTML에 이식 가능.
     JSON에 scaler의 mean_/scale_을 실어 보낼 필요가 없음.
  3) OR 해석이 직관적: composite는 "z-score 1점당", AGE는 "1세당",
     education_year는 "1년당", 이진 변수는 "집단 간 대비".
  4) z-score(SD 1)와 AGE(60~90) 정도의 스케일 차이로는 Newton-Raphson 수렴에
     문제가 없음.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve

import config as cfg

RANDOM_STATE = cfg.RANDOM_STATE


# ============================================================
# 0. 설계행렬 점검 헬퍼
# ============================================================

def drop_degenerate_features(X, verbose=True, label=''):
    """
    subgroup을 잘라내면 어떤 컬럼은 상수가 되어버림.
    (대표적으로 SCD_MCI 그룹에서 is_Dementia는 전부 0, Dementia 그룹에서
     is_MCI는 전부 0이고 is_Dementia는 전부 1)

    상수 컬럼이 설계행렬에 들어가면 절편과 완전공선이 되어 Hessian이 singular가
    되고 적합 자체가 실패함. 여기서 미리 걸러내고 무엇을 뺐는지 알려줌.

    반환: 살아남은 feature 이름 리스트
    """
    keep, dropped = [], []
    for c in X.columns:
        if X[c].nunique(dropna=False) <= 1:
            dropped.append(c)
        else:
            keep.append(c)

    if verbose and dropped:
        prefix = f'[{label}] ' if label else ''
        print(f"  {prefix}상수 컬럼 제외 (이 subgroup에서 값이 하나뿐): {dropped}")
    return keep


def _design_matrix(X, features):
    """절편을 붙인 설계행렬(DataFrame). 컬럼 이름이 살아 있어야 params가 이름으로 나옴."""
    return sm.add_constant(X[features].astype(float), has_constant='add')


# ============================================================
# 1. 5-fold CV (성능 추정 전용)
# ============================================================

def run_cv_pipeline(X, y, features=None, n_splits=cfg.N_CV_SPLITS,
                    random_state=RANDOM_STATE, verbose=True):
    """
    5-fold CV로 out-of-fold(OOF) 예측을 모음. **성능 추정 전용**이며,
    여기서 나온 계수는 리포트에 쓰지 않음 (OR/CI/p-value는 fit_final_lr의 결과만 사용).

    ** fold별 OR을 평균내지 않는 이유 **
    OR은 로그 스케일에서 대칭이라 mean(exp(beta)) != exp(mean(beta))이고,
    p-value의 평균은 어떤 가설검정의 결과도 아님. 게다가 LASSO를 없앤 지금은
    모든 fold가 동일한 feature set을 쓰므로 CV로 계수를 볼 이유 자체가 없음.

    X, y      : DataFrame / Series (index 정렬 필수)
    features  : 사용할 컬럼 리스트. None이면 X의 전체 컬럼.

    반환: fold_results (list of dict)
          각 dict: fold, features, index(원본 index), y_true, y_prob
          (수렴 실패한 fold는 제외되며 로그로 안내)
    """
    features = list(features) if features is not None else list(X.columns)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_results = []

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        # fold train 안에서 상수가 되어버린 컬럼이 있으면 그 fold에서만 제외
        feats = drop_degenerate_features(X_tr[features], verbose=False)

        model = _fit_logit(X_tr, y_tr, feats, tag=f'Fold {fold_i}', verbose=verbose)
        if model is None:
            continue

        y_prob = np.asarray(model.predict(_design_matrix(X_te, feats)))

        fold_results.append({
            'fold': fold_i,
            'features': feats,
            'index': X_te.index.to_numpy(),
            'y_true': y_te.to_numpy(),
            'y_prob': y_prob,
        })

    if verbose and len(fold_results) < n_splits:
        print(f"  [CV] {len(fold_results)}/{n_splits} fold만 사용됨")

    return fold_results


def _fit_logit(X, y, features, tag='', verbose=True):
    """statsmodels Logit 적합. 수렴 실패/특이행렬이면 None 반환."""
    try:
        model = sm.Logit(y.to_numpy().astype(float), _design_matrix(X, features)).fit(
            disp=0, maxiter=200
        )
    except (np.linalg.LinAlgError, Exception) as e:  # PerfectSeparationError 포함
        if verbose:
            print(f"  [{tag}] Logit 적합 실패 ({type(e).__name__}: {e}) - 제외")
        return None

    if not model.mle_retvals.get('converged', True):
        if verbose:
            print(f"  [{tag}] Logit 수렴 실패 (converged=False) - 제외")
        return None
    return model


def pooled_predictions(fold_results):
    """
    fold_results 전체의 OOF 예측을 원본 index 기준으로 이어붙이고 **index 순으로 정렬**.

    index 정렬이 중요한 이유: DeLong 검정은 "같은 피험자, 같은 순서"인 두 예측
    벡터를 요구함. 같은 group 안에서 M1~M4는 y와 표본 수가 같아 StratifiedKFold
    분할이 동일하지만, index로 정렬해 두면 그 가정에 의존하지 않아도 안전함.

    반환: y_true (Series), y_prob (Series) - 둘 다 원본 index를 가짐
    """
    idx = np.concatenate([r['index'] for r in fold_results])
    y_true = np.concatenate([r['y_true'] for r in fold_results])
    y_prob = np.concatenate([r['y_prob'] for r in fold_results])

    s_true = pd.Series(y_true, index=idx, name='y_true').sort_index()
    s_prob = pd.Series(y_prob, index=idx, name='y_prob').sort_index()
    return s_true, s_prob


# ============================================================
# 2. Youden's Index (threshold)
# ============================================================

def youden_threshold(y_true, y_prob):
    """
    pooled OOF 예측값에서 Youden's Index(= TPR - FPR 최대화)로 최적 threshold를
    한 번 계산. fold마다 따로 구해 평균내는 것보다 특정 fold 노이즈에 덜 흔들림.

    최종 모델의 in-sample 예측이 아니라 OOF에서 뽑는 이유: in-sample에서 고르면
    threshold도 낙관적으로 편향되기 때문.

    주의: sklearn roc_curve()는 맨 앞에 threshold=inf인 "가짜 시작점"을 넣음.
    모델이 거의 무작위라 Youden's J가 동점이 되면 argmax가 이 지점을 골라
    threshold=inf(전부 음성 분류)가 나올 수 있어 미리 제외함.

    반환: best_threshold(float), roc_df(DataFrame)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_j = tpr - fpr
    roc_df = pd.DataFrame({
        'fpr': fpr, 'tpr': tpr, 'threshold': thresholds, 'youden_j': youden_j,
    })

    finite = np.isfinite(thresholds)
    best_threshold = float(thresholds[finite][int(np.argmax(youden_j[finite]))])
    return best_threshold, roc_df


# ============================================================
# 3. 최종 모델 refit (전체 표본 1회)
# ============================================================

def fit_final_lr(X, y, features=None, label='', verbose=True):
    """
    **전체 표본**으로 최종 Logistic Regression을 1회 적합.
    이 모델이 리포트의 OR/CI/p-value와 HTML 배포용 계수의 유일한 출처.

    스케일링을 하지 않으므로 model.params가 곧 원 스케일 계수이고,
    exp(params)가 곧 해석 가능한 Odds Ratio.

    반환: dict(model_type, model, features, label, n, n_events)
          predict_proba()에 그대로 넘길 수 있는 형태.
    """
    features = list(features) if features is not None else list(X.columns)
    features = drop_degenerate_features(X[features], verbose=verbose, label=label)

    model = _fit_logit(X, y, features, tag=f'{label} 최종 refit', verbose=verbose)
    if model is None:
        raise RuntimeError(
            f"[{label}] 최종 모델 적합에 실패했습니다. 완전분리(separation)이거나 "
            f"공선성이 심한 경우일 수 있습니다 - feature를 줄이거나 Firth 보정을 검토하세요."
        )

    return {
        'model_type': 'LR',
        'model': model,
        'features': features,
        'label': label,
        'n': int(len(y)),
        'n_events': int(np.asarray(y).sum()),
    }


def predict_proba(fitted, X):
    """fit_final_lr()가 반환한 dict를 받아 양성 확률을 계산."""
    return np.asarray(fitted['model'].predict(_design_matrix(X, fitted['features'])))


# ============================================================
# 4. 배포용 계수 추출 (HTML/JSON 이식의 핵심)
# ============================================================

def extract_coefficients(fitted):
    """
    최종 모델에서 intercept와 feature별 계수를 뽑음.
    스케일러가 없으므로 이 값이 그대로 HTML에서 쓰일 수 있음:

        logit = intercept + sum_j coef[j] * x_j
        P     = 1 / (1 + exp(-logit))

    반환: (intercept: float, coef: dict[str, float])
    """
    params = fitted['model'].params
    intercept = float(params['const'])
    coef = {f: float(params[f]) for f in fitted['features']}
    return intercept, coef