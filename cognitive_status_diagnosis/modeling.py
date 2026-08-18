"""
modeling.py
-----------
"모델을 어떻게 학습/예측하는가"에 관한 함수 전부.
통계적 유의성 검정(LRT, DeLong)은 stats_tests.py, 성능 리포트/그래프는
reporting.py로 분리했으니 그쪽을 참고.

전체 흐름 (Group x Model 하나에 대해):
    1) run_cv_pipeline()            : train(80%) 안에서 5-fold CV
                                       -> Model 간 비교, threshold/hyperparameter 탐색용
    2) youden_threshold()           : (LR 전용) CV의 pooled 예측값에서 최적 threshold 계산
    3) tune_rf_hyperparams()        : (RF 전용) RandomizedSearchCV로 하이퍼파라미터 탐색
    4) fit_final_lr() / fit_final_rf(): train(80%) 전체로 최종 모델 refit
    5) predict_proba()              : 최종 모델로 train(refit 성능 확인용)/test 예측

주의: LASSO 변수선택은 반드시 "그 시점에 쓰는 train 데이터 내부에서만" 수행.
      5-fold CV에서는 매 fold의 train_fold 안에서, 최종 refit에서는 train 80%
      전체 안에서 - 어느 경우든 test로 쓸 데이터를 미리 들여다보지 않음 (leakage 방지).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.inspection import permutation_importance
from scipy.stats import randint, uniform

import config as cfg

RANDOM_STATE = cfg.RANDOM_STATE


# ============================================================
# 1. LASSO 변수선택
# ============================================================

def lasso_select_features(X, y, random_state=RANDOM_STATE):
    """
    주어진 데이터 안에서만 LASSO(L1 logistic, 내부 5-fold CV로 규제강도 C 탐색)로
    변수를 선택. 5-fold CV의 각 fold train에도, 최종 refit(train 80% 전체)에도
    똑같이 이 함수를 재사용함 - "어디에 적용하느냐"만 호출하는 쪽에서 다름.

    여기서 쓰는 스케일러는 변수선택 전용이라 fit 후 바로 버림 (선택된 컬럼만
    다시 쓰는 최종 학습용 스케일러는 호출하는 쪽에서 selected feature 기준으로
    별도로 fit해야 함 - 전체 컬럼 기준 스케일러를 selected 컬럼에 재사용하면
    fit 때 본 feature 수와 안 맞아 에러남).

    반환: 선택된 feature 이름 리스트. 전부 탈락하면(coef 전부 0) 안전장치로
          원래 feature 전체를 반환 (호출하는 쪽에서 별도 처리 불필요).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lasso_cv = LogisticRegressionCV(
        penalty='l1', solver='liblinear', cv=5, Cs=20, max_iter=5000,
        class_weight='balanced', random_state=random_state,
    )
    lasso_cv.fit(X_scaled, y)

    coef = lasso_cv.coef_[0]
    selected = list(X.columns[coef != 0])
    return selected if len(selected) > 0 else list(X.columns)


# ============================================================
# 2. 5-fold CV (Model 간 비교, threshold/hyperparameter 탐색용 - 1.1.1 단계)
# ============================================================

def run_cv_pipeline(X, y, model_type, use_lasso=True,
                     n_splits=cfg.N_CV_SPLITS, random_state=RANDOM_STATE,
                     rf_params=None, group_map=None):
    """
    X, y : train(80%) 안에서의 DataFrame/Series (index 정렬 필수)
    model_type : 'LR' (statsmodels Logit) 또는 'RF' (RandomForestClassifier)
    use_lasso : True면 각 fold의 train_fold 내부에서 LASSO 변수선택 후 학습
    rf_params : RF 하이퍼파라미터 dict (tune_rf_hyperparams()의 결과를 그대로 넣으면 됨).
                None이면 기본값(n_estimators=500)만 사용.
    group_map : {'1st_cognitive_status': ['is_MCI', 'is_Dementia'], ...} 형태.
                RF에서 grouped permutation importance를 같이 계산하고 싶을 때만 지정.

    반환: fold_results (list of dict), 각 dict는
          fold, model, features, y_true, y_prob (+ RF는 perm_importance,
          group_map 지정 시 grouped_perm_importance 추가)
          (LR이 수렴 실패한 fold는 결과에서 제외됨 - 로그로 안내)
    """
    if model_type not in ('LR', 'RF'):
        raise ValueError("model_type must be 'LR' or 'RF'")

    rf_params = dict(rf_params) if rf_params else {'n_estimators': 500}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_results = []

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        feats = lasso_select_features(X_tr, y_tr, random_state) if use_lasso else list(X_tr.columns)

        scaler = StandardScaler().fit(X_tr[feats])
        X_tr_s = scaler.transform(X_tr[feats])
        X_te_s = scaler.transform(X_te[feats])

        if model_type == 'LR':
            model, y_prob = _fit_predict_lr(X_tr_s, y_tr, X_te_s, fold_i)
            if model is None:
                continue  # 수렴 실패 - 이 fold 제외

            fold_result = {
                'fold': fold_i, 'model': model, 'features': feats,
                'y_true': y_te.values, 'y_prob': np.asarray(y_prob),
            }

        else:  # RF
            model = RandomForestClassifier(
                class_weight='balanced', random_state=random_state, **rf_params,
            ).fit(X_tr_s, y_tr)
            y_prob = model.predict_proba(X_te_s)[:, 1]

            perm = permutation_importance(
                model, X_te_s, y_te, scoring='roc_auc',
                n_repeats=30, random_state=random_state, n_jobs=1,
            )

            fold_result = {
                'fold': fold_i, 'model': model, 'features': feats,
                'y_true': y_te.values, 'y_prob': np.asarray(y_prob),
                'perm_importance': perm.importances_mean,
            }
            if group_map is not None:
                fold_result['grouped_perm_importance'] = grouped_permutation_importance(
                    model, X_te_s, y_te, feats, group_map, random_state=random_state,
                )

        fold_results.append(fold_result)

    return fold_results


def _fit_predict_lr(X_tr_s, y_tr, X_te_s, fold_i):
    """statsmodels Logit fit + predict. 수렴 실패 시 (None, None) 반환."""
    X_tr_c = sm.add_constant(X_tr_s, has_constant='add')
    X_te_c = sm.add_constant(X_te_s, has_constant='add')
    try:
        model = sm.Logit(y_tr.values, X_tr_c).fit(disp=0, maxiter=200)
        if not model.mle_retvals.get('converged', True):
            print(f"  [Fold {fold_i}] Logit 수렴 실패 (converged=False) - 이 fold 제외")
            return None, None
        return model, model.predict(X_te_c)
    except np.linalg.LinAlgError:
        print(f"  [Fold {fold_i}] Logit 수렴 실패 (singular Hessian) - 이 fold 제외")
        return None, None


def pooled_predictions(fold_results):
    """fold_results 전체에서 y_true/y_prob를 이어붙임 (OOF pooled 예측값)."""
    y_true = np.concatenate([r['y_true'] for r in fold_results])
    y_prob = np.concatenate([r['y_prob'] for r in fold_results])
    return y_true, y_prob


# ============================================================
# 3. Youden's Index (LR 전용 threshold 최적화)
# ============================================================

def youden_threshold(y_true, y_prob):
    """
    pooled(OOF) 예측값에서 Youden's Index(= TPR - FPR 최대화)로 최적 threshold를
    한 번만 계산. fold마다 따로 계산해서 평균내는 방식 대신, 5개 fold의 예측값을
    전부 모아(pool) 그 위에서 한 번 계산하는 게 특정 fold의 노이즈에 덜 흔들림.

    주의: sklearn의 roc_curve()는 항상 맨 앞에 threshold=inf(=fpr 0, tpr 0)인
    "가짜 시작점"을 하나 끼워 넣음. 모델이 거의 무작위(AUC≈0.5)에 가까워
    여러 지점의 Youden's J가 동점이 되면, np.argmax가 이 가짜 시작점을 그대로
    골라 threshold=inf(=모든 예측을 음성으로 분류)가 나오는 사고가 생길 수 있음
    -> argmax 계산에서는 이 inf 지점을 제외.

    반환: best_threshold, roc_df (fpr, tpr, thresholds, youden_j 를 담은 DataFrame,
          inf 지점 포함 - plot_roc_curve 등 시각화에는 그대로 사용 가능)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_j = tpr - fpr
    roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': thresholds, 'youden_j': youden_j})

    finite_mask = np.isfinite(thresholds)
    best_idx_in_finite = int(np.argmax(youden_j[finite_mask]))
    best_threshold = float(thresholds[finite_mask][best_idx_in_finite])

    return best_threshold, roc_df


# ============================================================
# 4. RF 하이퍼파라미터 탐색 (RandomizedSearchCV)
# ============================================================

# scipy.stats 분포 객체 -> 매 iteration 연속값(실수)으로 무작위 샘플링됨.
# GridSearchCV처럼 이산 후보를 직접 지정할 필요가 없음.
RF_PARAM_DIST = {
    'n_estimators':      randint(200, 800),
    'max_depth':         randint(3, 25),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf':  randint(1, 15),
    'max_features':      uniform(0.1, 0.9),  # [0.1, 1.0) 사이 연속값 (feature 비율)
}


def tune_rf_hyperparams(X, y, param_dist=None, n_iter=cfg.RF_N_ITER,
                         cv=cfg.N_CV_SPLITS, random_state=RANDOM_STATE, n_jobs=1):
    """
    RandomizedSearchCV로 RF 하이퍼파라미터 탐색 (scoring='roc_auc').

    여기 넘기는 X는 이미 LASSO 등으로 골라둔 feature만 담긴 상태여야 함
    (feature 선택과 하이퍼파라미터 탐색을 한 함수에서 같이 하지 않음 - 역할 분리).

    반환: best_params (dict, run_cv_pipeline의 rf_params로 그대로 넘기면 됨),
          search (fitted RandomizedSearchCV 객체 - search.cv_results_로 전체 탐색 결과 확인 가능)
    """
    param_dist = param_dist or RF_PARAM_DIST

    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)

    base_rf = RandomForestClassifier(class_weight='balanced', random_state=random_state)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        base_rf, param_dist, n_iter=n_iter, cv=skf,
        scoring='roc_auc', random_state=random_state, n_jobs=n_jobs,
    )
    search.fit(X_s, y)
    return search.best_params_, search


# ============================================================
# 5. Grouped (Joint) Permutation Importance
# ============================================================

def grouped_permutation_importance(model, X_te_s, y_te, feats, group_map,
                                    n_repeats=30, random_state=RANDOM_STATE):
    """
    RF 전용 - 더미화되어 여러 컬럼으로 나뉜 변수(예: is_MCI + is_Dementia =
    1st_cognitive_status)를 "같은 permutation 순서로 동시에" 섞어서, 그 변수
    전체(reference 카테고리 정보까지 포함)를 지웠을 때 AUC가 얼마나 떨어지는지 측정.

    개별 permutation_importance(is_MCI만 섞기)와 다른 점:
    개별 방식은 SCD(=is_MCI 0, is_Dementia 0)가 "기준선"으로 고정된 채 MCI 대비만
    보는 것이라 SCD 자체의 정보는 결과에 안 드러남. 반면 이 함수는 두 더미를 같이
    섞어서 (0,0) 조합이 어디로 가는지도 함께 뒤섞이므로, SCD를 포함한 변수 전체의
    효과를 측정함. reporting.plot_combined_importance()에서 개별 결과와
    나란히 비교할 때 이 차이를 염두에 두면 됨.

    X_te_s : 스케일링된 test fold (numpy array), 컬럼 순서가 feats와 일치해야 함
    feats  : 이 fold에서 실제 학습에 쓰인 feature 이름 리스트

    반환: DataFrame (group, n_columns_in_fold, grouped_permutation_importance_mean/std)
          그룹 컬럼이 이 fold의 feats에 하나도 없으면(LASSO로 전부 탈락) 결과에서 제외.
    """
    rng = np.random.RandomState(random_state)
    baseline_score = roc_auc_score(y_te, model.predict_proba(X_te_s)[:, 1])

    records = []
    for group_name, group_cols in group_map.items():
        col_idx = [feats.index(c) for c in group_cols if c in feats]
        if len(col_idx) == 0:
            continue

        drop_scores = []
        for _ in range(n_repeats):
            X_perm = X_te_s.copy()
            perm_order = rng.permutation(X_perm.shape[0])
            X_perm[:, col_idx] = X_perm[perm_order][:, col_idx]
            permuted_score = roc_auc_score(y_te, model.predict_proba(X_perm)[:, 1])
            drop_scores.append(baseline_score - permuted_score)

        records.append({
            'group': group_name,
            'n_columns_in_fold': len(col_idx),
            'grouped_permutation_importance_mean': np.mean(drop_scores),
            'grouped_permutation_importance_std': np.std(drop_scores),
        })

    cols = ['group', 'n_columns_in_fold',
            'grouped_permutation_importance_mean', 'grouped_permutation_importance_std']
    if len(records) == 0:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(records).sort_values(
        'grouped_permutation_importance_mean', ascending=False
    ).reset_index(drop=True)


# ============================================================
# 6. 최종 모델 refit (train 80% 전체로 1회 학습 - 1.1.2 / 1.2 단계용)
# ============================================================
# CV(run_cv_pipeline)는 "Model 비교/threshold·hyperparameter 탐색"이 목적이라
# fold마다 다른 feature를 선택할 수 있음. 반면 여기서는 실제 report에 쓸,
# train/test에 공통 적용할 "단일 최종 모델" 하나를 만드는 것이 목적.

def fit_final_lr(X_train, y_train, features):
    """
    train 80% 전체로 최종 LR(statsmodels Logit) 모델을 학습.
    features는 보통 lasso_select_features(X_train, y_train)의 결과를 그대로 넘기면 됨.

    반환: dict(model, scaler, features) - predict_proba()에 그대로 넘기는 형태.
    """
    scaler = StandardScaler().fit(X_train[features])
    X_s = scaler.transform(X_train[features])
    X_c = sm.add_constant(X_s, has_constant='add')

    model = sm.Logit(y_train.values, X_c).fit(disp=0, maxiter=200)
    if not model.mle_retvals.get('converged', True):
        print("  [최종 LR refit] 수렴 실패 (converged=False) - 결과 해석에 주의")

    return {'model_type': 'LR', 'model': model, 'scaler': scaler, 'features': features}


def fit_final_rf(X_train, y_train, features, rf_params=None, random_state=RANDOM_STATE):
    """
    train 80% 전체로 최종 RF 모델을 학습.
    rf_params는 tune_rf_hyperparams()의 best_params를 그대로 넘기면 됨.

    반환: dict(model, scaler, features)
    """
    rf_params = dict(rf_params) if rf_params else {'n_estimators': 500}
    scaler = StandardScaler().fit(X_train[features])
    X_s = scaler.transform(X_train[features])

    model = RandomForestClassifier(
        class_weight='balanced', random_state=random_state, **rf_params,
    ).fit(X_s, y_train)

    return {'model_type': 'RF', 'model': model, 'scaler': scaler, 'features': features}


def predict_proba(fitted, X):
    """
    fit_final_lr / fit_final_rf가 반환한 dict를 받아 y_prob(양성 클래스 확률) 계산.
    train(refit 성능 확인)이든 test든 동일하게 이 함수 하나로 처리.
    """
    X_s = fitted['scaler'].transform(X[fitted['features']])
    if fitted['model_type'] == 'LR':
        X_c = sm.add_constant(X_s, has_constant='add')
        return np.asarray(fitted['model'].predict(X_c))
    else:
        return fitted['model'].predict_proba(X_s)[:, 1]


# ============================================================
# 7. SHAP (RF 전용, 최종 refit 모델을 test set에 적용 - 1.2 단계)
# ============================================================
# 계산은 항상 최종 refit 모델(fit_final_rf 결과) 기준.
# "어느 데이터에 대해" 계산하느냐가 핵심 선택인데, 여기서는 test set을 기본값으로 둠:
# permutation importance도 test set 기준으로 계산하고 있어서(run_cv_pipeline),
# "성능에 기여하는 정도(permutation)"와 "개인별 예측에 기여하는 방향(SHAP)"을
# 같은 대상(test set) 위에서 봐야 두 지표를 나란히 비교하는 의미가 있음.
# train set에 대해 계산하면 모델이 학습 과정에서 외운 패턴까지 "중요하다"고
# 나올 수 있어 일반화 관점에서는 test 쪽이 더 정직한 해석.

def compute_shap_values(fitted, X):
    """
    fit_final_rf()가 반환한 dict와, SHAP을 계산할 데이터(X, 보통 test set)를 받아
    TreeExplainer로 SHAP value를 계산.

    반환: dict(shap_values (n_samples, n_features, 양성 클래스 기준),
              expected_value (baseline 예측 확률), X_scaled (스케일링된 입력,
              summary plot에 사용할 실제 값), features (컬럼 순서))
    """
    import shap  # 지연 import - shap 없이도 나머지 모듈은 정상 동작하도록

    if fitted['model_type'] != 'RF':
        raise ValueError("SHAP은 현재 RF(TreeExplainer) 전용으로만 구현되어 있음")

    X_s = fitted['scaler'].transform(X[fitted['features']])
    explainer = shap.TreeExplainer(fitted['model'])
    raw_values = explainer.shap_values(X_s)

    # sklearn/shap 버전에 따라 shap_values가 (n, p, 2) 배열로 나오거나
    # [클래스0 배열, 클래스1 배열] 리스트로 나오는 두 경우가 있어 둘 다 처리
    if isinstance(raw_values, list):
        shap_values = raw_values[1]  # 양성(1) 클래스
        expected_value = explainer.expected_value[1]
    elif raw_values.ndim == 3:
        shap_values = raw_values[:, :, 1]
        expected_value = explainer.expected_value[1]
    else:
        shap_values = raw_values
        expected_value = explainer.expected_value

    return {
        'shap_values': shap_values,
        'expected_value': expected_value,
        'X_scaled': X_s,
        'features': fitted['features'],
    }