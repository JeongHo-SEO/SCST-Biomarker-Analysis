"""
reporting.py
------------
학습된 모델/CV 결과를 "사람이 읽을 수 있는 표/그래프"로 정리하는 함수 전용.
모델 학습은 modeling.py, 유의성 검정은 stats_tests.py 참고.

이 모듈의 두 축:
    [성능]  전부 5-fold CV의 out-of-fold(OOF) 예측에서만 나옴.
            최종 refit 모델을 자기 학습 데이터에 다시 적용한 apparent 성능은
            항상 낙관적으로 편향되므로 아예 계산하지 않음.
    [해석]  OR / 95% CI / p-value / 수식 / Top-K는 전부 최종 refit 모델 1개에서
            나옴. fold별 계수를 평균내지 않음.

--- 이전 버전에서 삭제된 것 ------------------------------------------------
  report_rf_importance, plot_rf_importance, plot_grouped_importance,
  report_combined_importance, plot_combined_importance, plot_shap_summary,
  report_test_performance(hold-out test set 폐지), fold 기반 report_lr_odds_ratio
----------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    brier_score_loss, classification_report, f1_score, roc_auc_score, roc_curve,
)

import config as cfg


# ============================================================
# 1. CV 성능 리포트 (성능 숫자의 유일한 출처)
# ============================================================

def report_cv_performance(fold_results, threshold=0.5, verbose=True):
    """
    fold별 F1 / AUC + 전체 fold OOF 합산(pooled) classification_report 출력.

    threshold : modeling.youden_threshold()가 pooled OOF에서 계산한 값.

    반환: fold별 metric DataFrame
    """
    if len(fold_results) == 0:
        if verbose:
            print("=== 모든 fold가 적합 실패로 제외됨 - 결과 없음 ===")
        return pd.DataFrame(columns=['fold', 'f1', 'auc'])

    all_y_true, all_y_prob, fold_metrics = [], [], []

    for res in fold_results:
        y_true, y_prob = res['y_true'], res['y_prob']
        y_pred = (y_prob >= threshold).astype(int)

        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = np.nan  # fold에 클래스가 하나만 있는 경우

        fold_metrics.append({'fold': res['fold'], 'f1': f1_score(y_true, y_pred), 'auc': auc})
        all_y_true.extend(y_true)
        all_y_prob.extend(y_prob)

    metrics_df = pd.DataFrame(fold_metrics)

    if verbose:
        print("=== Fold별 F1 / AUC (out-of-fold) ===")
        print(metrics_df.to_string(index=False))
        print(f"\nF1  mean ± std : {metrics_df['f1'].mean():.3f} ± {metrics_df['f1'].std():.3f}")
        print(f"AUC mean ± std : {metrics_df['auc'].mean():.3f} ± {metrics_df['auc'].std():.3f}")

        all_y_pred = (np.array(all_y_prob) >= threshold).astype(int)
        print(f"\n=== Pooled OOF classification_report (threshold={threshold:.3f}) ===")
        print(classification_report(all_y_true, all_y_pred, digits=3))

    return metrics_df


def plot_roc_curve(fold_results, title='ROC Curve (CV folds)'):
    """fold별 OOF ROC curve를 한 그래프에 overlay. Youden threshold 근거 확인용."""
    if len(fold_results) == 0:
        print(f"[{title}] 모든 fold가 실패해 ROC curve를 그릴 수 없습니다.")
        return None

    fig, ax = plt.subplots(figsize=(6, 6))
    for res in fold_results:
        fpr, tpr, _ = roc_curve(res['y_true'], res['y_prob'])
        auc = roc_auc_score(res['y_true'], res['y_prob'])
        ax.plot(fpr, tpr, alpha=0.7, label=f"Fold {res['fold']} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    return fig


def plot_auc_comparison(perf_records, title='CV AUC Comparison'):
    """
    Model별 CV AUC(mean ± std) 막대그래프.
    perf_records: [{'label': 'M1_All', 'auc': 0.78, 'std': 0.03}, ...]
    """
    df = pd.DataFrame(perf_records)
    if len(df) == 0:
        print(f"[{title}] 비교할 결과가 없습니다.")
        return None

    yerr = df['std'].fillna(0) if 'std' in df.columns else None

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(df) + 2), 4.5))
    bars = ax.bar(df['label'], df['auc'], yerr=yerr, capsize=4,
                  color='steelblue', alpha=0.85, ecolor='black')
    ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    ax.set_ylabel('CV AUC (out-of-fold)')
    ax.set_ylim(0.5, 1.0)
    ax.set_title(title)
    ax.axhline(0.5, color='gray', linewidth=0.8, linestyle='--')
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
    fig.tight_layout()
    return fig


# ============================================================
# 2. 최종 모델 Odds Ratio 표 (+ Top-K 하이라이트)
# ============================================================

def report_final_or(fitted, top_k=cfg.TOP_K, alpha=cfg.TOP_K_ALPHA):
    """
    **최종 refit 모델 1개**에서 OR / 95% CI / p-value를 뽑음.
    스케일링을 하지 않았으므로 exp(beta)가 곧 원 스케일 Odds Ratio.

    Top-K 규칙 (config.py 참고):
        1) p < alpha 인 변수만 후보
        2) 후보를 |beta| = |log(OR)| 내림차순 정렬
        3) 상위 top_k개에 is_top_k = True

    |OR|이 아니라 |beta|로 정렬하는 이유: OR은 1 기준으로 비대칭이라 OR=0.5(강한
    음의 효과)가 OR=1.2(약한 양의 효과)보다 작아 보임. 이 데이터는 인지 z-score가
    높을수록 amyloid 음성이라 composite 대부분의 OR이 1보다 작으므로, |OR| 정렬은
    핵심 마커를 순위 밖으로 밀어냄.

    반환: DataFrame
        feature, beta, abs_beta, OR, CI_lower, CI_upper, p_value,
        is_significant, is_top_k, separation_flag
    """
    model = fitted['model']
    conf = model.conf_int()  # DataFrame (index=param name, columns=[0,1])

    records = []
    for feat in fitted['features']:
        beta = float(model.params[feat])
        lo, hi = float(conf.loc[feat, 0]), float(conf.loc[feat, 1])
        p = float(model.pvalues[feat])

        or_ = np.exp(beta)
        ci_lo, ci_hi = np.exp(lo), np.exp(hi)
        ci_ratio = ci_hi / ci_lo if ci_lo > 0 else np.inf

        records.append({
            'feature': feat,
            'beta': beta,
            'abs_beta': abs(beta),
            'OR': or_,
            'CI_lower': ci_lo,
            'CI_upper': ci_hi,
            'p_value': p,
            'is_significant': p < alpha,
            'separation_flag': bool(
                or_ >= cfg.SEPARATION_OR_MAX
                or or_ <= 1.0 / cfg.SEPARATION_OR_MAX
                or ci_ratio >= cfg.SEPARATION_CI_RATIO_MAX
            ),
        })

    or_df = pd.DataFrame(records)

    # Top-K: 유의한 것만 후보로 두고 |beta| 내림차순 상위 top_k
    or_df['is_top_k'] = False
    candidates = or_df[or_df['is_significant']].sort_values('abs_beta', ascending=False)
    or_df.loc[candidates.index[:top_k], 'is_top_k'] = True

    # 보기 좋게 |beta| 내림차순 정렬
    or_df = or_df.sort_values('abs_beta', ascending=False).reset_index(drop=True)

    # 절편도 참고용으로 마지막에 붙임 (Top-K 대상 아님)
    intercept_row = {
        'feature': '(Intercept)',
        'beta': float(model.params['const']),
        'abs_beta': np.nan,
        'OR': np.exp(float(model.params['const'])),
        'CI_lower': np.exp(float(conf.loc['const', 0])),
        'CI_upper': np.exp(float(conf.loc['const', 1])),
        'p_value': float(model.pvalues['const']),
        'is_significant': False,
        'separation_flag': False,
        'is_top_k': False,
    }
    return pd.concat([or_df, pd.DataFrame([intercept_row])], ignore_index=True)


def print_or_table(or_df, label='', top_k=cfg.TOP_K, alpha=cfg.TOP_K_ALPHA):
    """
    OR 표를 콘솔에 보기 좋게 출력. Top-K에는 ★, separation 의심에는 ⚠ 표시.
    """
    show = or_df.copy()
    show['mark'] = np.where(show['is_top_k'], '★', '')
    show.loc[show['separation_flag'], 'mark'] = show.loc[show['separation_flag'], 'mark'] + '⚠'

    show['OR (95% CI)'] = show.apply(
        lambda r: f"{r['OR']:.3f} ({r['CI_lower']:.3f}–{r['CI_upper']:.3f})", axis=1
    )
    show['p'] = show['p_value'].apply(lambda p: '<0.001' if p < 0.001 else f'{p:.3f}')

    out = show[['mark', 'feature', 'beta', 'OR (95% CI)', 'p']].rename(
        columns={'beta': 'β=log(OR)'}
    )

    print(f"\n=== {label} Odds Ratio (최종 refit 모델, n={len(or_df) - 1} features) ===")
    print(out.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print(f"\n★ Top-{top_k} : p < {alpha} 후보 중 |β| 상위 {top_k}개")
    print("  (다중비교 미보정 - 탐색적 해석)")

    if show['separation_flag'].any():
        flagged = show.loc[show['separation_flag'], 'feature'].tolist()
        print(f"\n⚠ separation 의심 (OR 폭발 또는 CI 과대): {flagged}")
        print("  표본 대비 파라미터가 많을 때 나타납니다. 해당 그룹은 composite 축소나")
        print("  Firth 보정을 검토하세요. 이 계수를 그대로 배포하면 위험합니다.")


# ============================================================
# 3. 최종 회귀식 출력
# ============================================================

def format_formula(fitted, or_df=None, max_line_width=None, decimals=4):
    """
    최종 모델의 logistic regression 수식을 사람이 읽을 수 있는 문자열로 만듦.
    HTML에도 그대로 표시할 수 있도록 JSON에 함께 실어 보냄.

    or_df를 주면 |β| 내림차순으로 항을 정렬함 (안 주면 features 순서 그대로).
    """
    model = fitted['model']
    intercept = float(model.params['const'])

    feats = list(fitted['features'])
    if or_df is not None:
        ordered = [f for f in or_df['feature'] if f in set(feats)]
        feats = ordered + [f for f in feats if f not in set(ordered)]

    lines = [f"logit(P) = {intercept:+.{decimals}f}"]
    for f in feats:
        b = float(model.params[f])
        lines.append(f"           {b:+.{decimals}f} * {f}")
    lines.append("")
    lines.append("P(amyloid = positive) = 1 / (1 + exp(-logit(P)))")
    return '\n'.join(lines)


# ============================================================
# 4. Calibration (노트북 전용 - JSON/HTML에는 절대 반영하지 않음)
# ============================================================
# Platt scaling이나 isotonic regression을 붙이면 배포 식이 sigmoid를 두 번
# 통과하는 구조가 되어 "학습된 식 그대로 이식"이라는 목표가 깨짐.
# 여기서는 OOF 예측이 잘 보정되어 있는지 확인만 함.

def report_calibration(y_true, y_prob, n_bins=cfg.CALIBRATION_N_BINS,
                       label='', plot=True, verbose=True):
    """
    pooled OOF 예측의 보정 상태를 확인.

    반환: dict(brier, calibration_intercept, calibration_slope, bin_table)

    calibration slope / intercept (Cox calibration):
        실제 y를 logit(p_hat)에 회귀시켰을 때의 계수/절편.
        - slope = 1, intercept = 0 이면 완벽
        - slope < 1 이면 예측이 과도하게 극단적(과적합의 전형적 신호)
        - intercept != 0 이면 전반적으로 확률이 높거나 낮게 치우침

    in-sample이 아니라 반드시 OOF로 계산해야 의미가 있음.
    (MLE로 적합한 LR은 학습 데이터에서 자동으로 잘 보정되어 보임)
    """
    import statsmodels.api as sm

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-9, 1 - 1e-9)

    brier = brier_score_loss(y_true, y_prob)

    logit_p = np.log(y_prob / (1 - y_prob))
    cal_model = sm.Logit(y_true, sm.add_constant(logit_p, has_constant='add')).fit(disp=0)
    cal_intercept, cal_slope = float(cal_model.params[0]), float(cal_model.params[1])

    # 분위수 기반 bin (동일 개수) - 확률이 한쪽에 몰려도 bin이 비지 않음
    bins = pd.qcut(y_prob, q=n_bins, duplicates='drop')
    bin_table = (
        pd.DataFrame({'y': y_true, 'p': y_prob, 'bin': bins})
        .groupby('bin', observed=True)
        .agg(n=('y', 'size'), mean_predicted=('p', 'mean'), observed_rate=('y', 'mean'))
        .reset_index(drop=True)
    )

    if verbose:
        print(f"\n=== {label} Calibration (pooled out-of-fold) ===")
        print(f"Brier score          : {brier:.4f}  (낮을수록 좋음)")
        print(f"Calibration intercept: {cal_intercept:+.3f}  (0에 가까울수록 좋음)")
        print(f"Calibration slope    : {cal_slope:.3f}  (1에 가까울수록 좋음)")
        print(f"평균 예측확률 {y_prob.mean():.3f} vs 실제 유병률 {y_true.mean():.3f}")
        print(bin_table.to_string(index=False, float_format=lambda x: f'{x:.3f}'))

    fig = None
    if plot:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='완벽한 보정')
        ax.plot(bin_table['mean_predicted'], bin_table['observed_rate'],
                marker='o', color='steelblue', label='관측')
        ax.set_xlabel('평균 예측확률')
        ax.set_ylabel('실제 양성 비율')
        ax.set_title(f'{label} Reliability Diagram (OOF)')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()

    return {
        'brier': float(brier),
        'calibration_intercept': cal_intercept,
        'calibration_slope': cal_slope,
        'bin_table': bin_table,
        'fig': fig,
    }