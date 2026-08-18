"""
run_experiments.py
------------------
config.py에 정의된 Group x Model 조합을 실제로 순회하며 실행하는 오케스트레이션 파일.
전처리는 data_prep, 학습은 modeling, 검정은 stats_tests, 리포트는 reporting에
위임하고, 여기서는 "어떤 순서로 무엇을 호출하는지"만 담당.

한 Group x Model 조합 안에서 일어나는 일:
    1) run_cv_pipeline()   : 5-fold CV -> OOF 예측 (성능 추정)
    2) youden_threshold()  : pooled OOF에서 threshold 1회 계산
    3) report_cv_performance() : fold별 F1/AUC + pooled classification report
    4) fit_final_lr()      : **전체 표본**으로 최종 모델 1회 refit
    5) report_final_or()   : OR / 95% CI / p-value / Top-K
    6) format_formula()    : 최종 회귀식 문자열
    7) lr_joint_test()     : (더미 2개 이상인 covariate가 있으면) LRT
    8) report_calibration(): OOF 기준 보정 확인 (노트북 전용)

사용 예 (노트북에서):
    import data_prep, run_experiments as run

    df, covariate_column_map = data_prep.preprocess(df_composite_raw)
    summary_df, detail = run.run_all_experiments(df, covariate_column_map,
                                                 target_folder='amyloid_beta_prob_pipeline',
                                                 result_folder='889_final')
    run.print_full_report(detail)
    run.export_all_models_json(detail, '../amyloid_beta_prob_pipeline/models.json')
"""

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
import data_prep
import modeling
import reporting
import stats_tests


# ============================================================
# 1. Model별 feature 리스트 생성
# ============================================================

def get_features_for_model(group_name, model_idx, covariate_column_map):
    """
    GROUPS[group_name]의 covariate_steps를 앞에서부터 model_idx개만 누적해서
    실제 X 컬럼 리스트(composite scores + 누적 covariate)를 만듦.

    model_idx=0 -> M1 (baseline, composite만)
    model_idx=k -> covariate_steps[:k] 까지 포함

    반환: model_name, feature_cols, covariate_included(원래 이름 리스트)
    """
    group_cfg = cfg.GROUPS[group_name]
    steps = group_cfg['covariate_steps']
    model_names = group_cfg['model_names']

    if model_idx > len(steps):
        raise ValueError(f"{group_name}: model_idx={model_idx}가 step 수({len(steps)})를 초과")

    model_name = model_names[model_idx]

    covariate_included = []
    for step in steps[:model_idx]:
        covariate_included.extend(step)

    feature_cols = list(cfg.COMPOSITE_SCORES)
    for cov in covariate_included:
        cols = covariate_column_map.get(cov, [])
        if len(cols) == 0:
            raise KeyError(f"{cov}가 covariate_column_map에 없음 - preprocess() 먼저 실행했는지 확인")
        feature_cols.extend(cols)

    return model_name, feature_cols, covariate_included


def build_group_map(covariate_list, covariate_column_map):
    """
    covariate 원래 이름 리스트 중 "더미화되어 2개 이상 컬럼으로 나뉜 것"만 골라
    LRT 대상 group_map 생성.

    컬럼 1개짜리(AGE, education_year, APOE_e4_carrier, sex 이진)는 Wald test와
    결과가 사실상 같으므로 제외. 현재 이 조건에 걸리는 건 1st_cognitive_status뿐.

    반환: 예) {'1st_cognitive_status': ['is_MCI', 'is_Dementia']}
    """
    return {
        cov: cols
        for cov in covariate_list
        if len(cols := covariate_column_map.get(cov, [])) >= 2
    }


# ============================================================
# 2. 한 Group x Model 조합 실행
# ============================================================

def _run_one_model(df_g, feature_cols, model_name, group_map, verbose=True):
    """한 조합에 대해 CV -> threshold -> 최종 refit -> OR/수식/LRT/보정까지 수행."""
    X = df_g[feature_cols]
    y = df_g['y_amyloid']

    # --- 1) 성능 추정: 5-fold CV (OOF) ---
    fold_results = modeling.run_cv_pipeline(X, y, features=feature_cols, verbose=verbose)

    if len(fold_results) > 0:
        oof_y, oof_p = modeling.pooled_predictions(fold_results)
        threshold, roc_df = modeling.youden_threshold(oof_y, oof_p)
    else:
        oof_y, oof_p, threshold, roc_df = None, None, 0.5, pd.DataFrame()

    cv_metrics = reporting.report_cv_performance(fold_results, threshold=threshold, verbose=False)

    # --- 2) 해석/배포: 전체 표본 최종 refit 1회 ---
    fitted = modeling.fit_final_lr(X, y, features=feature_cols, label=model_name, verbose=verbose)
    or_df = reporting.report_final_or(fitted)
    formula = reporting.format_formula(fitted, or_df=or_df)
    intercept, coef = modeling.extract_coefficients(fitted)

    result = {
        'model_name': model_name,
        'feature_cols': feature_cols,
        'fold_results': fold_results,
        'cv_metrics': cv_metrics,
        'oof_y_true': oof_y,
        'oof_y_prob': oof_p,
        'threshold': threshold,
        'roc_df': roc_df,
        'fitted': fitted,
        'or_df': or_df,
        'formula': formula,
        'intercept': intercept,
        'coef': coef,
        'n': fitted['n'],
        'n_events': fitted['n_events'],
    }

    # --- 3) LRT (더미 2개 이상인 covariate가 있을 때만) ---
    if len(group_map) > 0:
        result['lrt'] = stats_tests.lr_joint_test(
            X, y, fitted['features'], group_map, label=model_name, verbose=verbose
        )

    # --- 4) Calibration (OOF 기준, 노트북 확인용) ---
    if oof_y is not None:
        cal = reporting.report_calibration(oof_y, oof_p, label=model_name,
                                           plot=False, verbose=False)
        cal.pop('fig', None)
        result['calibration'] = cal

    return result


# ============================================================
# 3. 전체 실험 실행
# ============================================================

def _resolve_result_dir(target_folder, result_folder):
    """
    결과 저장 경로를 '{target_folder}/results/{result_folder}'로 조립.

    target_folder가 절대경로면 그대로 사용하고, 상대경로면 '../{target_folder}'로
    해석함 (노트북이 형제 폴더에 있는 기존 사용 방식과 호환).

    ** 노트북에서는 절대경로를 넘기는 쪽을 권장 **
    상대경로는 os.getcwd() 기준이라, 커널을 다른 위치에서 띄우거나 중간에
    디렉터리를 옮기면 결과가 엉뚱한 곳에 저장됨. 예:
        amyloid_dir = (Path.cwd().parent / 'amyloid_beta_prob_pipeline').resolve()
        run.run_all_experiments(..., target_folder=amyloid_dir, result_folder='889_final')
    """
    base = Path(target_folder)
    if not base.is_absolute():
        base = Path('..') / base
    return str(base / 'results' / result_folder)


def run_all_experiments(df, covariate_column_map, target_folder, result_folder,
                        verbose=True, auto_save=True):
    """
    GROUPS에 정의된 모든 Group x Model 조합을 실행.

    df                   : data_prep.preprocess()의 결과 (전체 N=889, split 없음)
    covariate_column_map : data_prep.preprocess()의 결과
    target_folder        : 결과 저장 상위 폴더 (예: 'amyloid_beta_prob_pipeline')
    result_folder        : 하위 폴더 (예: '889_final')
                           -> '../{target_folder}/results/{result_folder}'

    반환:
        summary_df    : 조합별 요약 1행씩
        detail_results: detail_results[group][model]에 전체 결과 보관
    """
    result_dir = _resolve_result_dir(target_folder, result_folder)
    if auto_save:
        os.makedirs(result_dir, exist_ok=True)

    summary_records = []
    detail_results = {}

    for group_name, group_cfg in cfg.GROUPS.items():
        df_g = data_prep.filter_group(df, group_cfg['filter'])
        n_pos = int(df_g['y_amyloid'].sum())
        n_neg = len(df_g) - n_pos

        if verbose:
            print(f"\n{'=' * 66}")
            print(f"Group: {group_name}  (n={len(df_g)}, 양성={n_pos}, 음성={n_neg})")
            print('=' * 66)

        detail_results[group_name] = {}

        for model_idx in range(len(group_cfg['model_names'])):
            model_name, feature_cols, covariate_included = get_features_for_model(
                group_name, model_idx, covariate_column_map
            )
            group_map = build_group_map(covariate_included, covariate_column_map)

            # EPV(events per variable): 소수 클래스 기준
            n_minority = min(n_pos, n_neg)
            epv = n_minority / len(feature_cols)

            if verbose:
                print(f"\n--- {model_name} "
                      f"(n_features={len(feature_cols)}, covariates={covariate_included}) ---")
                warn = '  <-- 낮음(권장 10 이상), 계수 해석에 주의' if epv < 10 else ''
                print(f"    EPV = {n_minority}/{len(feature_cols)} = {epv:.1f}{warn}")

            res = _run_one_model(df_g, feature_cols, model_name, group_map, verbose=verbose)
            res.update({'group': group_name, 'epv': epv,
                        'covariate_included': covariate_included})
            detail_results[group_name][model_name] = res

            cv = res['cv_metrics']
            summary_records.append({
                'Group': group_name,
                'Model': model_name,
                'n': res['n'],
                'n_events': res['n_events'],
                'n_features': len(feature_cols),
                'EPV': round(epv, 1),
                'threshold': res['threshold'],
                'cv_auc_mean': cv['auc'].mean() if len(cv) > 0 else np.nan,
                'cv_auc_std': cv['auc'].std() if len(cv) > 0 else np.nan,
                'cv_f1_mean': cv['f1'].mean() if len(cv) > 0 else np.nan,
                'cv_f1_std': cv['f1'].std() if len(cv) > 0 else np.nan,
                'brier': res.get('calibration', {}).get('brier', np.nan),
                'cal_slope': res.get('calibration', {}).get('calibration_slope', np.nan),
                'n_separation_flags': int(res['or_df']['separation_flag'].sum()),
            })

    summary_df = pd.DataFrame(summary_records)

    if auto_save:
        save_results(summary_df, detail_results, target_folder, result_folder)

    return summary_df, detail_results


# ============================================================
# 4. Model 간 비교 (DeLong, pooled OOF 기준)
# ============================================================

def compare_models_delong(detail_results, group_name, model_a=None, model_b=None):
    """
    같은 Group 안의 두 모델을 pooled OOF 예측으로 DeLong 비교.
    기본값은 첫 모델(M1, composite만) vs 마지막 모델(full covariate).

    "covariate를 추가한 것이 통계적으로 유의미한 판별력 향상을 주는가"에 대한 답.
    """
    models = detail_results[group_name]
    names = list(models.keys())
    model_a = model_a or names[0]
    model_b = model_b or names[-1]

    a, b = models[model_a], models[model_b]
    return stats_tests.delong_test(
        a['oof_y_true'], a['oof_y_prob'], b['oof_y_prob'],
        label_1=model_a, label_2=model_b,
    )


def compare_all_steps_delong(detail_results, group_name):
    """한 Group의 인접 모델 쌍(M1 vs M2, M2 vs M3, ...)을 순차 비교한 표."""
    names = list(detail_results[group_name].keys())
    rows = []
    for a, b in zip(names[:-1], names[1:]):
        r = compare_models_delong(detail_results, group_name, a, b)
        rows.append({
            'Group': group_name, 'from': a, 'to': b,
            'auc_from': r[f'auc_{a}'], 'auc_to': r[f'auc_{b}'],
            'auc_diff': -r['auc_diff'],  # to - from 방향으로 뒤집어 표시
            'p_value': r['p_value'],
        })
    return pd.DataFrame(rows)


# ============================================================
# 5. HTML 배포용 JSON export
# ============================================================
# 이 JSON이 Python과 HTML 사이의 유일한 연결고리.
# HTML은 이 파일만 읽어서 아래 3줄로 확률을 계산함:
#     let logit = m.intercept;
#     for (const [k, v] of Object.entries(m.coefficients)) logit += v * x[k];
#     const p = 1 / (1 + Math.exp(-logit));
#
# StandardScaler를 쓰지 않았기 때문에 scaler의 mean_/scale_을 실을 필요가 없음.
# Calibration 보정(Platt/isotonic)도 일부러 넣지 않음 - 넣으면 sigmoid를 두 번
# 통과하는 구조가 되어 "학습된 식 그대로 이식"이 깨짐.

def _encoding_spec(covariate_column_map):
    """HTML에서 입력값을 0/1로 바꿀 때 쓸 if-then 규칙."""
    sex_cols = covariate_column_map.get('sex', [])
    sex_col = sex_cols[0] if sex_cols else None
    sex_positive = sex_col.split('sex_', 1)[1] if sex_col else None

    return {
        'sex': {
            'column': sex_col,
            'rule': f"{sex_col} = 1 if sex == '{sex_positive}' else 0" if sex_col else None,
        },
        'APOE': {
            'column': 'APOE_e4_carrier',
            'rule': "APOE_e4_carrier = 1 if genotype contains 'E4' else 0",
        },
        cfg.COGNITIVE_STATUS_COL: {
            'columns': [f'is_{c}' for c in cfg.COGNITIVE_STATUS_CATEGORIES],
            'rule': (f"is_MCI = 1 if status == 'MCI' else 0; "
                     f"is_Dementia = 1 if status == 'Dementia' else 0; "
                     f"reference = '{cfg.COGNITIVE_STATUS_REFERENCE}' (둘 다 0)"),
        },
        'composite_scores': {
            'columns': list(cfg.COMPOSITE_SCORES),
            'rule': 'demographic-stratified z-score를 그대로 입력 (추가 표준화 없음)',
        },
    }


def build_model_json(res):
    """
    detail_results의 항목 하나를 HTML 배포용 dict로 변환.

    encoding(입력값 -> 0/1 변환 규칙)은 모델마다 동일하므로 여기 넣지 않고
    payload 최상위에 한 번만 둠 (11개 모델에 복사하면 규칙 수정 시 11군데를
    고쳐야 하고 "어느 것이 진짜인가" 문제가 생김).
    """
    or_df = res['or_df']
    or_records = []
    for _, r in or_df.iterrows():
        if r['feature'] == '(Intercept)':
            continue
        or_records.append({
            'feature': r['feature'],
            'beta': round(float(r['beta']), 6),
            'OR': round(float(r['OR']), 4),
            'ci_lower': round(float(r['CI_lower']), 4),
            'ci_upper': round(float(r['CI_upper']), 4),
            'p_value': float(f"{r['p_value']:.6g}"),
            'is_significant': bool(r['is_significant']),
            'top_k': bool(r['is_top_k']),
        })

    cv = res['cv_metrics']

    return {
        'model_name': res['model_name'],
        'group': res['group'],
        'requires': {
            'cognitive_status': cfg.COGNITIVE_STATUS_COL in res['covariate_included']
                                 or res['group'] in ('SCD_MCI', 'Dementia'),
            'apoe': 'APOE' in res['covariate_included'],
            'demographics': 'AGE' in res['covariate_included'],
        },
        'intercept': round(float(res['intercept']), 6),
        'coefficients': {k: round(float(v), 6) for k, v in res['coef'].items()},
        'features': list(res['fitted']['features']),
        'formula': res['formula'],
        'or_table': or_records,
        'top_k_rule': f"p < {cfg.TOP_K_ALPHA} 후보 중 |beta| 상위 {cfg.TOP_K}개",
        'n': int(res['n']),
        'n_events': int(res['n_events']),
        'prevalence': round(res['n_events'] / res['n'], 4),
        'cv_auc_mean': round(float(cv['auc'].mean()), 4) if len(cv) > 0 else None,
        'cv_auc_std': round(float(cv['auc'].std()), 4) if len(cv) > 0 else None,
        'threshold_youden': round(float(res['threshold']), 4),
    }


def export_all_models_json(detail_results, covariate_column_map, out_path,
                           models_to_export=None, verbose=True):
    """
    모든(또는 지정한) 모델을 하나의 JSON 파일로 저장.

    models_to_export : ['M2_All', 'M2_SCD_MCI', ...] 형태로 담을 모델을 고를 수 있는
                       **Python 쪽 인자**. None이면 전부 저장(기본).
                       전체 11개를 다 담아도 116KB 수준이라 용량/속도 문제가 없고,
                       HTML에서 모델을 바꿔가며 비교하는 것도 가능해지므로 기본값 권장.

    JSON 구조:
        {
          "meta": {...},
          "application_rule": {...},   # cognitive_status/APOE 유무 -> 모델 선택
          "encoding": {...},           # 모든 모델 공통 (여기 한 번만 존재)
          "models": {"M2_All": {...}, "M4_SCD_MCI": {...}, ...}
        }
    """
    models = {}
    for group_name, group_models in detail_results.items():
        for model_name, res in group_models.items():
            if models_to_export is not None and model_name not in models_to_export:
                continue
            models[model_name] = build_model_json(res)

    payload = {
        'meta': {
            'target': 'P(1st_amyloid_status == 양성)',
            'algorithm': 'Logistic Regression (no regularization, no scaling)',
            'note': ('계수는 원 스케일입니다. 입력값을 그대로 곱해 더한 뒤 sigmoid를 '
                     '적용하면 됩니다. 별도의 표준화나 calibration 보정이 필요 없습니다.'),
            'disclaimer': '연구용 참고 도구이며 진단 목적이 아닙니다.',
            'random_state': cfg.RANDOM_STATE,
            'n_cv_splits': cfg.N_CV_SPLITS,
        },
        'application_rule': {
            'cognitive_status=Unknown, APOE=Unknown': 'M2_All',
            'cognitive_status=Known,   APOE=Unknown': 'M2_SCD_MCI / M2_Dementia',
            'cognitive_status=Unknown, APOE=Known': ('인지기능검사 -> 혈액검사 순서이므로 '
                                                      'cognitive_status 입력을 먼저 요청'),
            'cognitive_status=Known,   APOE=Known': 'M4_SCD_MCI / M3_Dementia',
        },
        # 모든 모델이 공유하는 입력 인코딩 규칙 (모델별로 복사하지 않음)
        'encoding': _encoding_spec(covariate_column_map),
        'models': models,
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"JSON 저장 완료: {out_path}  (모델 {len(models)}개)")
    return payload


# ============================================================
# 6. Python <-> JS 일치 검증용 golden set
# ============================================================

def make_golden_set(detail_results, df, covariate_column_map, model_name,
                    out_path, n=20, random_state=cfg.RANDOM_STATE):
    """
    JSON 계수를 HTML에 옮긴 뒤, JS 계산 결과가 Python과 일치하는지 대조할
    "정답표"를 CSV로 만듦.

    계수를 손으로 옮기는 과정에서 인코딩/반올림/컬럼 누락으로 값이 어긋나기 쉬운데,
    이 표가 있으면 배포 전에 확실히 잡을 수 있음.

    저장 컬럼: 모델 입력 feature 전부 + python_prob
    """
    res = None
    for group_models in detail_results.values():
        if model_name in group_models:
            res = group_models[model_name]
            break
    if res is None:
        raise KeyError(f"{model_name}을 detail_results에서 찾을 수 없습니다.")

    df_g = data_prep.filter_group(df, cfg.GROUPS[res['group']]['filter'])
    sample = df_g.sample(n=min(n, len(df_g)), random_state=random_state)

    feats = res['fitted']['features']
    probs = modeling.predict_proba(res['fitted'], sample)

    out = sample[feats].copy()
    out.insert(0, 'model_name', model_name)
    out['python_prob'] = probs

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    out.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"golden set 저장 완료: {out_path}  (n={len(out)})")
    print("  -> HTML에 같은 값을 넣고 python_prob과 소수점 6자리까지 일치하는지 확인하세요.")
    return out


# ============================================================
# 7. 결과 저장 / 로드
# ============================================================

def save_results(summary_df, detail_results, target_folder, result_folder):
    """summary_df -> csv, detail_results -> pickle."""
    result_dir = _resolve_result_dir(target_folder, result_folder)
    os.makedirs(result_dir, exist_ok=True)

    summary_path = os.path.join(result_dir, 'summary.csv')
    detail_path = os.path.join(result_dir, 'detail_results.pkl')

    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    with open(detail_path, 'wb') as f:
        pickle.dump(detail_results, f)

    print(f"\n저장 완료: {summary_path}")
    print(f"저장 완료: {detail_path}")
    return result_dir


def load_results(target_folder, result_folder):
    """save_results()로 저장된 결과를 다시 불러옴."""
    result_dir = _resolve_result_dir(target_folder, result_folder)
    summary_df = pd.read_csv(os.path.join(result_dir, 'summary.csv'))
    with open(os.path.join(result_dir, 'detail_results.pkl'), 'rb') as f:
        detail_results = pickle.load(f)
    return summary_df, detail_results


# ============================================================
# 8. 전체 리포트 출력
# ============================================================

def print_full_report(detail_results, show_plots=True):
    """
    Group -> Model 순서로 순회하며
    CV 성능 -> ROC -> OR 표(Top-K 하이라이트) -> 회귀식 -> LRT -> calibration 출력.
    마지막에 Group 단위로 CV AUC 비교 막대그래프와 DeLong 표를 그림.
    """
    import matplotlib.pyplot as plt

    for group_name, models in detail_results.items():
        print(f"\n{'#' * 70}\n# Group: {group_name}\n{'#' * 70}")
        auc_records = []

        for model_name, res in models.items():
            print(f"\n{'=' * 70}\n[{group_name}] {model_name}  "
                  f"(n={res['n']}, events={res['n_events']}, "
                  f"features={len(res['feature_cols'])}, EPV={res['epv']:.1f})\n{'=' * 70}")

            reporting.report_cv_performance(res['fold_results'],
                                            threshold=res['threshold'], verbose=True)

            if show_plots and len(res['fold_results']) > 0:
                reporting.plot_roc_curve(res['fold_results'],
                                         title=f'{group_name} - {model_name} - OOF ROC')
                plt.show()

            reporting.print_or_table(res['or_df'], label=f'{group_name}/{model_name}')

            print(f"\n--- {model_name} 최종 회귀식 ---")
            print(res['formula'])

            if 'lrt' in res and len(res['lrt']) > 0:
                print(f"\n--- {model_name} Likelihood Ratio Test (확증적) ---")
                print(res['lrt'].to_string(index=False))

            if 'calibration' in res:
                cal = res['calibration']
                print(f"\n--- {model_name} Calibration (OOF) ---")
                print(f"Brier={cal['brier']:.4f}, "
                      f"intercept={cal['calibration_intercept']:+.3f}, "
                      f"slope={cal['calibration_slope']:.3f}")

            cv = res['cv_metrics']
            auc_records.append({
                'label': model_name,
                'auc': cv['auc'].mean() if len(cv) > 0 else np.nan,
                'std': cv['auc'].std() if len(cv) > 0 else np.nan,
            })

        if show_plots:
            reporting.plot_auc_comparison(auc_records, title=f'{group_name} - CV AUC (OOF)')
            plt.show()

        print(f"\n--- {group_name} 인접 Model DeLong 비교 (pooled OOF) ---")
        print(compare_all_steps_delong(detail_results, group_name)
              .to_string(index=False, float_format=lambda x: f'{x:.4f}'))