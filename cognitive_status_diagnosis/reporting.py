"""
reporting.py
------------
학습된 모델/CV 결과를 "사람이 읽을 수 있는 표/그래프"로 정리하는 함수 전용.
모델을 학습하는 로직은 modeling.py, 유의성 검정은 stats_tests.py 참고.

CV 성능(report_cv_performance)과 최종 test 성능(report_test_performance)을
일부러 별개 함수로 뒀음: CV는 fold가 여러 개라 fold별 mean±std를 낼 수 있지만,
test는 held-out set 하나뿐이라 fold 개념이 없음 - 억지로 하나의 함수로 합치면
"fold가 1개인 CV"처럼 보여서 오해 소지가 있음.
(참고: 최종 test AUC는 점추정치만 보고 - CV 단계에서 이미 fold별 변동성을
 report하고 있고, test는 최종 확인 목적이라 별도 CI 계산은 생략함. 추후
 리뷰어가 test AUC 불확실성을 요구하면 stats_tests.delong_test로 확장 가능)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, roc_auc_score, roc_curve


# ============================================================
# 1. CV 성능 리포트 (1.1.1 단계 - fold별 mean ± std)
# ============================================================

def report_cv_performance(fold_results, threshold=0.5, verbose=True):
    """
    fold별 F1 / AUC + 전체 fold test 합산(pooled) classification_report 출력.
    threshold: LR은 modeling.youden_threshold()의 결과를, RF는 기본 0.5를 넘기면 됨.

    반환: fold별 metric DataFrame (모든 fold 실패 시 빈 DataFrame)
    """
    if len(fold_results) == 0:
        if verbose:
            print("=== 모든 fold가 수렴 실패로 제외됨 (0/N fold) - 결과 없음 ===")
        return pd.DataFrame(columns=['fold', 'f1', 'auc'])

    all_y_true, all_y_prob, fold_metrics = [], [], []

    for res in fold_results:
        y_true, y_prob = res['y_true'], res['y_prob']
        y_pred = (y_prob >= threshold).astype(int)

        f1 = f1_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = np.nan  # test fold에 클래스가 하나만 있는 경우

        fold_metrics.append({'fold': res['fold'], 'f1': f1, 'auc': auc})
        all_y_true.extend(y_true)
        all_y_prob.extend(y_prob)

    metrics_df = pd.DataFrame(fold_metrics)

    if verbose:
        print("=== Fold별 F1 / AUC ===")
        print(metrics_df.to_string(index=False))
        print(f"\nF1  mean ± std : {metrics_df['f1'].mean():.3f} ± {metrics_df['f1'].std():.3f}")
        print(f"AUC mean ± std : {metrics_df['auc'].mean():.3f} ± {metrics_df['auc'].std():.3f}")

        all_y_pred = (np.array(all_y_prob) >= threshold).astype(int)
        print("\n=== Pooled classification_report (전체 fold test 합산) ===")
        print(classification_report(all_y_true, all_y_pred))

    return metrics_df


def plot_roc_curve(fold_results, title='ROC Curve (CV folds)'):
    """
    fold별 ROC curve를 한 그래프에 overlay.
    CV(1.1.1) 단계에서 Youden's Index 확인용으로만 사용 - train/test 성능
    리포팅(1.1.2, 1.2)에서는 AUC 값 자체를 비교하는 쪽을 권장 (ROC curve 자체는
    모델 개수가 많아지면 겹쳐 보기 어려움).
    """
    if len(fold_results) == 0:
        print(f"[{title}] 모든 fold가 실패해 ROC curve를 그릴 수 없습니다.")
        return None

    fig, ax = plt.subplots(figsize=(6, 6))
    for res in fold_results:
        fpr, tpr, _ = roc_curve(res['y_true'], res['y_prob'])
        auc = roc_auc_score(res['y_true'], res['y_prob'])
        ax.plot(fpr, tpr, alpha=0.7, label=f"Fold {res['fold']} (AUC={auc:.2f})")

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    return fig


# ============================================================
# 2. Train-refit / Test 성능 리포트 (1.1.2, 1.2 단계 - fold 없는 단일 세트)
# ============================================================

def report_test_performance(y_true, y_prob, threshold, label='test', verbose=True):
    """
    fold 개념이 없는 단일 세트(train 80% 전체 refit 성능 또는 최종 test 20%)에
    대한 AUC / classification_report 출력.

    threshold : CV 단계(youden_threshold)에서 정한 값을 그대로 사용
                (여기서 다시 최적화하지 않음 - train/test 각각에서 threshold를
                새로 고르면 "test에서 가장 좋아 보이는 지점"을 고르는 것이 되어
                일반화 성능 평가라는 test set의 목적에 어긋남)

    반환: dict(label, n, auc, f1, threshold)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = np.nan
    f1 = f1_score(y_true, y_pred)

    if verbose:
        print(f"=== {label} 성능 (n={len(y_true)}, threshold={threshold:.3f}) ===")
        print(f"AUC : {auc:.3f}")
        print(classification_report(y_true, y_pred))

    return {'label': label, 'n': len(y_true), 'auc': auc, 'f1': f1, 'threshold': threshold}


def plot_auc_comparison(perf_records, title='AUC Comparison'):
    """
    report_test_performance() 결과(dict)들의 리스트를 받아 AUC 막대그래프로 비교.
    예: [train_perf_M1, test_perf_M1, train_perf_M2, test_perf_M2, ...]
    perf_records 각 dict에 'label'과 'auc' 키가 있으면 됨 (Model 이름을 label에
    미리 넣어서 넘기면 x축에 그대로 표시됨).
    """
    df = pd.DataFrame(perf_records)
    if len(df) == 0:
        print(f"[{title}] 비교할 결과가 없습니다.")
        return None

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(df) + 2), 4.5))
    bars = ax.bar(df['label'], df['auc'], color='steelblue', alpha=0.85)
    ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)
    ax.set_ylabel('AUC')
    ax.set_ylim(0.5, 1.0)
    ax.set_title(title)
    ax.axhline(0.5, color='gray', linewidth=0.8, linestyle='--')
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    fig.tight_layout()
    return fig


# ============================================================
# 3. LR - Odds Ratio 리포트
# ============================================================

def report_lr_odds_ratio(fold_results):
    """
    LR 모델(CV fold_results) 전용. Feature | OR | 95% CI | p-value 요약.

    LASSO가 fold마다 다른 변수를 선택할 수 있어서:
    - long_df : fold별 개별 결과 (raw)
    - summary : feature별로 선택된 fold에서만 평균낸 값 + 선택 빈도(selected_ratio)

    ** post-selection inference 주의: LASSO 선택 이후의 p-value라 확증적
       (confirmatory) 해석이 아닌 exploratory 참고용. **
    """
    if len(fold_results) == 0:
        print("모든 fold가 실패해 OR 테이블을 만들 수 없습니다.")
        empty_summary = pd.DataFrame(columns=[
            'feature', 'selected_folds_by_lasso', 'OR_mean',
            'CI_lower_mean', 'CI_upper_mean', 'p_value_mean', 'selected_ratio'
        ])
        empty_long = pd.DataFrame(columns=['fold', 'feature', 'OR', 'CI_lower', 'CI_upper', 'p_value'])
        return empty_summary, empty_long

    records = []
    for res in fold_results:
        model = res['model']
        feats = ['const'] + res['features']
        params, conf, pvals = np.asarray(model.params), np.asarray(model.conf_int()), np.asarray(model.pvalues)

        for i, feat in enumerate(feats):
            if feat == 'const':
                continue
            records.append({
                'fold': res['fold'], 'feature': feat, 'OR': np.exp(params[i]),
                'CI_lower': np.exp(conf[i][0]), 'CI_upper': np.exp(conf[i][1]), 'p_value': pvals[i],
            })

    long_df = pd.DataFrame(records)
    summary = (
        long_df.groupby('feature')
        .agg(selected_folds_by_lasso=('fold', 'count'), OR_mean=('OR', 'mean'),
             CI_lower_mean=('CI_lower', 'mean'), CI_upper_mean=('CI_upper', 'mean'),
             p_value_mean=('p_value', 'mean'))
        .reset_index()
    )
    summary['selected_ratio'] = summary['selected_folds_by_lasso'].astype(str) + '/' + str(len(fold_results))
    summary = summary.sort_values('selected_folds_by_lasso', ascending=False).reset_index(drop=True)

    return summary, long_df


# ============================================================
# 4. RF - Permutation Importance 리포트
# ============================================================

def report_rf_importance(fold_results):
    """
    RF 모델(CV fold_results) 전용. Feature | Importance 5-fold Mean/Std 요약.
    Permutation Importance(roc_auc 기준, n_repeats=30) 사용 - Gini/MDI
    importance(feature_importances_)는 연속형 변수를 과대평가하는 편향이 있어 배제.

    LASSO로 fold마다 feature set이 다를 수 있어 선택된 fold에서만 평균/표준편차 계산.
    """
    if len(fold_results) == 0:
        print("모든 fold가 실패해 importance 테이블을 만들 수 없습니다.")
        empty_summary = pd.DataFrame(columns=[
            'feature', 'selected_folds_by_lasso', 'importance_mean', 'importance_std', 'selected_ratio'
        ])
        return empty_summary, pd.DataFrame(columns=['fold', 'feature', 'importance'])

    records = []
    for res in fold_results:
        for feat, imp in zip(res['features'], res['perm_importance']):
            records.append({'fold': res['fold'], 'feature': feat, 'importance': imp})

    long_df = pd.DataFrame(records)
    summary = (
        long_df.groupby('feature')
        .agg(selected_folds_by_lasso=('fold', 'count'), importance_mean=('importance', 'mean'),
             importance_std=('importance', 'std'))
        .reset_index()
    )
    summary['selected_ratio'] = summary['selected_folds_by_lasso'].astype(str) + '/' + str(len(fold_results))
    summary = summary.sort_values('importance_mean', ascending=False).reset_index(drop=True)

    return summary, long_df


def plot_rf_importance(summary, title='RF Feature Importance (Permutation, mean ± std)', top_n=15):
    """summary(report_rf_importance 결과)를 받아 importance_mean 막대 + std 에러바로 표시."""
    if len(summary) == 0:
        print(f"[{title}] summary가 비어있어 plot을 그릴 수 없습니다.")
        return None

    plot_df = summary.head(top_n).iloc[::-1]
    xerr = plot_df['importance_std'].fillna(0)  # 1-fold짜리는 std NaN -> 에러바 0으로 표시

    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(plot_df) + 1)))
    bars = ax.barh(plot_df['feature'], plot_df['importance_mean'],
                   xerr=xerr, capsize=3, alpha=0.8, color='steelblue', ecolor='black')
    ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=8)
    ax.set_xlabel('Permutation Importance (AUC drop)')
    ax.set_title(title)
    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
    fig.tight_layout()
    return fig


# ============================================================
# 5. Grouped importance 시각화 (단독 / 개별과 통합)
# ============================================================

def plot_grouped_importance(grouped_importance_long, title='Grouped Permutation Importance', top_n=15):
    """
    modeling.grouped_permutation_importance()가 fold별로 쌓인 long-format DataFrame을
    받아 group별 평균 ± std를 막대로 표시 (그룹들끼리만 비교).
    개별 feature와 나란히 비교하려면 plot_combined_importance()를 사용.
    """
    if len(grouped_importance_long) == 0:
        print(f"[{title}] 데이터가 비어있어 plot을 그릴 수 없습니다.")
        return None

    summary = (
        grouped_importance_long.groupby('group')['grouped_permutation_importance_mean']
        .agg(['mean', 'std']).reset_index()
        .rename(columns={'mean': 'importance_mean', 'std': 'importance_std'})
        .sort_values('importance_mean', ascending=False)
    )

    plot_df = summary.head(top_n).iloc[::-1]
    xerr = plot_df['importance_std'].fillna(0)

    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(plot_df) + 1)))
    bars = ax.barh(plot_df['group'], plot_df['importance_mean'],
                   xerr=xerr, capsize=3, alpha=0.8, color='darkorange', ecolor='black')
    ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=8)
    ax.set_xlabel('Grouped Permutation Importance (AUC drop)')
    ax.set_title(title)
    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
    fig.tight_layout()
    return fig


def report_combined_importance(imp_summary, grouped_importance_long, group_labels):
    """
    개별 feature importance(report_rf_importance 결과)와 grouped(joint) importance
    (예: 1st_cognitive_status 전체)를 하나의 표로 통합.

    'importance_type' 컬럼으로 각 행이 개별 feature인지, 여러 컬럼을 묶은 grouped
    importance인지 표시함 - 표만 봐도 "이건 그룹 전체 효과"라는 게 바로 드러남.

    개별 is_MCI / is_Dementia 행은 "SCD 대비" 값(SCD 자체 정보는 안 담김)이고,
    '{group} (joint)' 행은 SCD를 포함한 변수 전체 효과 - 서로 다른 걸 재는
    지표라는 점에 유의해서 같이 보면 됨 (지난 논의 참고).

    group_labels : 통합하고 싶은 grouped 변수 이름 리스트, 예) ['1st_cognitive_status']
                   (컬럼 1개짜리인 AGE, education_year, APOE_e4_carrier는 이미
                   imp_summary에 개별 feature로 들어있어 중복 추가 불필요)

    반환: combined DataFrame (feature, importance_mean, importance_std, importance_type),
          importance_mean 내림차순 정렬. grouped_importance_long이 비어있으면
          개별 feature만 담긴 표를 그대로 반환 (에러 없이 동작).
    """
    combined = imp_summary[['feature', 'importance_mean', 'importance_std']].copy()
    combined['importance_type'] = 'individual'

    if len(grouped_importance_long) > 0:
        for group_label in group_labels:
            grp = grouped_importance_long[grouped_importance_long['group'] == group_label]
            if len(grp) == 0:
                continue
            mean_val = grp['grouped_permutation_importance_mean'].mean()
            std_val = grp['grouped_permutation_importance_mean'].std()
            combined = pd.concat([combined, pd.DataFrame({
                'feature': [f'{group_label} (joint)'],
                'importance_mean': [mean_val],
                'importance_std': [std_val],
                'importance_type': ['group (joint)'],
            })], ignore_index=True)

    return combined.sort_values('importance_mean', ascending=False).reset_index(drop=True)


def plot_combined_importance(combined_table, title='Combined Feature Importance (개별 + Group joint)', top_n=15):
    """
    report_combined_importance()가 반환한 표를 그대로 받아 막대그래프로 표시.
    importance_type == 'group (joint)'인 행만 다른 색(주황)으로 강조 -
    범례로도 "주황 = 여러 컬럼을 묶은 그룹 전체 효과"라는 걸 표시함.
    """
    if len(combined_table) == 0:
        print(f"[{title}] 데이터가 비어있어 plot을 그릴 수 없습니다.")
        return None

    plot_df = combined_table.sort_values('importance_mean', ascending=False).head(top_n).iloc[::-1]
    xerr = plot_df['importance_std'].fillna(0)
    colors = ['darkorange' if t == 'group (joint)' else 'steelblue' for t in plot_df['importance_type']]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(plot_df) + 1)))
    bars = ax.barh(plot_df['feature'], plot_df['importance_mean'],
                   xerr=xerr, capsize=3, alpha=0.85, color=colors, ecolor='black')
    ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=8)
    ax.set_xlabel('Permutation Importance (AUC drop)')
    ax.set_title(title)
    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color='steelblue', label='개별 feature'),
        Patch(color='darkorange', label='Group (joint) - 더미 묶음 전체 효과'),
    ], loc='lower right', fontsize=8)

    fig.tight_layout()
    return fig


# ============================================================
# 6. SHAP summary plot (RF 전용)
# ============================================================

def plot_shap_summary(shap_result, title='SHAP Summary (Test set)', max_display=15):
    """
    modeling.compute_shap_values()의 결과를 받아 SHAP summary plot(beeswarm)을 그림.

    beeswarm plot 읽는 법 (처음 보시면):
    - y축: feature, importance 큰 순서로 위에서부터 정렬
    - x축: SHAP value (그 feature가 이 사람의 예측을 양성 쪽(+)/음성 쪽(-)으로
      얼마나 밀었는지)
    - 점 하나 = 피험자 한 명. 색은 그 사람의 실제 feature 값(빨강=높음, 파랑=낮음)
    - 예: delayed_free_recall 행에서 파란 점들이 오른쪽(+)에 몰려 있으면
      "delayed_free_recall이 낮을수록 amyloid 양성 쪽으로 예측이 밀린다"는 뜻
    """
    import shap

    shap.summary_plot(
        shap_result['shap_values'], shap_result['X_scaled'],
        feature_names=shap_result['features'], max_display=max_display, show=False,
    )
    fig = plt.gcf()
    fig.axes[0].set_title(title)
    fig.tight_layout()
    return fig


# ============================================================
# 7. 성능 추이 시각화 (M1~M6(or M5) 모델별 F1/AUC 추이)
# ============================================================

def plot_performance_trend(summary_df, group_name, model_type):
    """
    summary_df(run_experiments.run_all_experiments 반환값)에서 특정 Group x
    model_type의 M1~M6(or M5) 성능 추이를 F1/AUC 각각 다른 색 꺾은선으로 표시.
    """
    sub = summary_df[
        (summary_df['Group'] == group_name) & (summary_df['model_type'] == model_type)
    ].copy()

    if len(sub) == 0:
        print(f"[{group_name} / {model_type}] 해당하는 데이터가 없습니다.")
        return None

    sub['model_order'] = sub['Model'].str.extract(r'M(\d+)').astype(int)
    sub = sub.sort_values('model_order')
    model_sequence = ' - '.join(sub['Model'].tolist())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(sub['Model'], sub['cv_f1_mean'], yerr=sub['cv_f1_std'],
                marker='o', capsize=4, color='steelblue', label='CV F1')
    ax.errorbar(sub['Model'], sub['cv_auc_mean'], yerr=sub['cv_auc_std'],
                marker='s', capsize=4, color='darkorange', label='CV AUC')
    ax.plot(sub['Model'], sub['test_auc'], marker='^', color='firebrick',
            linestyle='--', label='Test AUC')

    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title(f'{group_name} / {model_type} / {model_sequence}')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig