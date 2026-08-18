"""
run_experiments.py
--------------------
config.py에 정의된 Group x Model 조합을 실제로 순회하며 실행하는 오케스트레이션 파일.
전처리는 data_prep, 모델 학습은 modeling, 통계검정은 stats_tests, 리포트는
reporting에 위임하고, 여기서는 "어떤 순서로 무엇을 호출하는지"만 담당.

사용 예 (노트북에서):
    import data_prep, modeling, stats_tests, reporting
    import run_experiments as run
    import config as cfg

    df, covariate_column_map = data_prep.preprocess(df_composite_raw)
    df_train, df_test = data_prep.split_train_test(df)

    summary_df, detail_results = run.run_all_experiments(
        df_train, df_test, covariate_column_map
    )
    run.print_full_report(detail_results)

한 Group x Model x model_type 조합 안에서 실제로 일어나는 일 (1.1 -> 1.2 순서):
    1) run_cv_pipeline()        : train(80%) 5-fold CV -> Model 비교용 성능
       - LR: pooled 예측값으로 youden_threshold() 계산
       - RF: 미리 tune_rf_hyperparams()로 하이퍼파라미터 탐색 후 CV에 사용
    2) fit_final_lr/rf()        : train(80%) 전체로 최종 모델 1개 refit
    3) predict_proba() x 2      : train 성능(refit, 참고용) / test 성능(최종 평가) 각각 계산
    4) report_lr_odds_ratio / report_rf_importance : 해석용 부가 리포트
    5) (group_map 있으면) lr_joint_test / grouped_permutation_importance
"""

import os
import pickle

import numpy as np
import pandas as pd

import config as cfg
import modeling
import reporting
import stats_tests


# ============================================================
# 1. Model별 feature 리스트 생성
# ============================================================

def get_features_for_model(group_name, model_idx, covariate_column_map):
    """
    GROUPS[group_name]의 covariate_order를 앞에서부터 model_idx개만 슬라이싱해서
    실제 X 컬럼 리스트(composite scores + 누적 covariate)를 만듦.

    model_idx=0 -> baseline (covariate 없음)
    model_idx=k -> covariate_order[:k] 까지 포함

    반환: model_name, feature_cols, covariate_included(원래 이름 리스트,
          build_group_map에 그대로 넘기면 됨)
    """
    group_cfg = cfg.GROUPS[group_name]
    covariate_order = group_cfg['covariate_order']
    model_names = group_cfg['model_names']

    if model_idx > len(covariate_order):
        raise ValueError(
            f"{group_name}: model_idx={model_idx}가 covariate_order 길이"
            f"({len(covariate_order)})를 초과함"
        )

    model_name = model_names[model_idx]
    covariate_included = covariate_order[:model_idx]

    feature_cols = list(cfg.COMPOSITE_SCORES)
    for cov in covariate_included:
        cols = covariate_column_map.get(cov, [])
        if len(cols) == 0:
            raise KeyError(f"{cov}가 covariate_column_map에 없음 - preprocess() 먼저 실행했는지 확인")
        feature_cols.extend(cols)

    return model_name, feature_cols, covariate_included


def build_group_map(covariate_list, covariate_column_map):
    """
    covariate 원래 이름 리스트를 받아 "더미화되어 2개 이상 컬럼으로 나뉜 것"만
    골라 group_map 생성 (grouped LRT / grouped permutation importance 대상).

    AGE, education_year, APOE_e4_carrier처럼 컬럼 1개짜리는 제외
    (grouped 검정은 "여러 컬럼을 묶어서" 보는 것이라 컬럼 1개면 개별 검정과
    결과가 동일해져 의미가 없음 - APOE가 carrier 이진값 1개로 바뀌면서
    더 이상 grouped 대상이 아니게 됨. 현재 이 조건에 걸리는 건
    1st_cognitive_status(is_MCI, is_Dementia) 뿐).

    반환: group_map, 예) {'1st_cognitive_status': ['is_MCI', 'is_Dementia']}
    """
    group_map = {}
    for cov in covariate_list:
        cols = covariate_column_map.get(cov, [])
        if len(cols) >= 2:
            group_map[cov] = cols
    return group_map


# ============================================================
# 2. 한 Group x Model x model_type 조합 실행
# ============================================================

def _run_one_combination(model_type, X_train, y_train, X_test, y_test,
                          model_name, group_map, use_lasso=True):
    """
    한 조합(예: All / M4 / LR)에 대해 CV -> (RF면 튜닝) -> 최종 refit ->
    train/test 평가까지 전부 수행하고 결과를 dict로 반환.
    """
    result = {}

    if model_type == 'LR':
        fold_results = modeling.run_cv_pipeline(X_train, y_train, model_type='LR', use_lasso=use_lasso)
        cv_metrics_raw = reporting.report_cv_performance(fold_results, threshold=0.5, verbose=False)

        if len(fold_results) > 0:
            pooled_y_true, pooled_y_prob = modeling.pooled_predictions(fold_results)
            threshold, roc_df = modeling.youden_threshold(pooled_y_true, pooled_y_prob)
        else:
            threshold, roc_df = 0.5, pd.DataFrame()

        # threshold 확정 후 CV metric을 그 threshold 기준으로 다시 집계 (F1 등)
        cv_metrics = reporting.report_cv_performance(fold_results, threshold=threshold, verbose=False)

        final_features = modeling.lasso_select_features(X_train, y_train) if use_lasso else list(X_train.columns)
        final_model = modeling.fit_final_lr(X_train, y_train, final_features)

        or_summary, or_long = reporting.report_lr_odds_ratio(fold_results)
        result.update({'or_summary': or_summary, 'or_long': or_long, 'roc_df': roc_df})

        if len(group_map) > 0 and len(fold_results) > 0:
            lrt_long, lrt_summary = stats_tests.lr_joint_test(X_train, y_train, group_map, use_lasso=use_lasso)
            result.update({'lrt_long': lrt_long, 'lrt_summary': lrt_summary})

    else:  # RF
        # 튜닝은 LASSO로 한 번 골라둔 feature 위에서 수행 (전체 컬럼으로 튜닝하면
        # CV의 fold별 LASSO 선택과 탐색 대상이 어긋날 수 있어, 대표 feature set을 고정)
        tuning_features = modeling.lasso_select_features(X_train, y_train) if use_lasso else list(X_train.columns)
        best_params, _search = modeling.tune_rf_hyperparams(X_train[tuning_features], y_train)

        fold_results = modeling.run_cv_pipeline(
            X_train, y_train, model_type='RF', use_lasso=use_lasso,
            rf_params=best_params, group_map=group_map if len(group_map) > 0 else None,
        )
        threshold = 0.5  # RF는 Youden's Index 대상 아님 (LR만 threshold 최적화)
        cv_metrics = reporting.report_cv_performance(fold_results, threshold=threshold, verbose=False)

        final_features = tuning_features
        final_model = modeling.fit_final_rf(X_train, y_train, final_features, rf_params=best_params)

        imp_summary, imp_long = reporting.report_rf_importance(fold_results)
        result.update({'imp_summary': imp_summary, 'imp_long': imp_long, 'rf_best_params': best_params})

        grouped_importance_long = pd.DataFrame()
        if len(group_map) > 0:
            grouped_list = [fr['grouped_perm_importance'].assign(fold=fr['fold'])
                             for fr in fold_results if 'grouped_perm_importance' in fr]
            if len(grouped_list) > 0:
                grouped_importance_long = pd.concat(grouped_list, ignore_index=True)
                result['grouped_importance_long'] = grouped_importance_long

        # 개별 + 그룹(joint) importance를 하나의 표로 통합 (group_map이 없으면 개별만 담김)
        result['combined_importance'] = reporting.report_combined_importance(
            imp_summary, grouped_importance_long, group_labels=list(group_map.keys())
        )

    # --- 최종 모델로 train(refit 확인용) / test(최종 평가) 성능 ---
    train_prob = modeling.predict_proba(final_model, X_train[final_features])
    test_prob = modeling.predict_proba(final_model, X_test[final_features])

    train_perf = reporting.report_test_performance(
        y_train, train_prob, threshold, label=f'{model_name}_train_refit', verbose=False
    )
    test_perf = reporting.report_test_performance(
        y_test, test_prob, threshold, label=f'{model_name}_test', verbose=False
    )

    result.update({
        'fold_results': fold_results, 'cv_metrics': cv_metrics,
        'threshold': threshold, 'final_model': final_model, 'final_features': final_features,
        'train_perf': train_perf, 'test_perf': test_perf,
        'train_prob': train_prob, 'test_prob': test_prob,
        'y_train': y_train.values, 'y_test': y_test.values,  # 나중에 classification_report 재출력용
    })

    # --- SHAP (RF 전용, 최종 모델을 test set에 적용 - 1.2 단계) ---
    if model_type == 'RF':
        try:
            result['shap_result'] = modeling.compute_shap_values(final_model, X_test)
        except ImportError:
            print("  [SHAP] 'shap' 패키지가 설치되어 있지 않아 SHAP 계산을 건너뜁니다 "
                  "(pip install shap --break-system-packages).")

    return result


# ============================================================
# 3. 전체 실험 실행
# ============================================================

def _resolve_result_dir(target_folder, result_folder):
    """최종 저장 경로를 '../{target_folder}/results/{result_folder}' 형태로 조립."""
    return os.path.join('..', target_folder, 'results', result_folder)


def run_all_experiments(df_train, df_test, covariate_column_map, target_folder, result_folder,
                         model_types=None, verbose=True, auto_save=True, checkpoint=True):
    """
    GROUPS에 정의된 모든 Group x Model 조합을 model_types(기본: LR, RF)에 대해 전부 실행.

    df_train, df_test : data_prep.split_train_test()의 결과
    covariate_column_map : data_prep.preprocess()의 결과
    target_folder : 결과를 저장할 상위 프로젝트 폴더명 (예: 'amyloid_beta_diagnosis').
                    최종 저장 경로는 '../{target_folder}/results/{result_folder}'.
    result_folder : 그 안의 하위 폴더명 (예: '889_feature_selection_v2'). 실수로 결과가
                    엉뚱한 폴더에 저장/덮어써지는 걸 막기 위해 둘 다 매번 명시적으로 지정해야 함.

    반환:
        summary_df     : 조합별 요약 1행씩 (Group, Model, model_type, n_features,
                          threshold, cv_f1_mean/std, cv_auc_mean/std,
                          train_auc, test_auc, test_f1)
        detail_results  : 중첩 dict, detail_results[group][model][model_type]에
                           fold_results/최종모델/OR·importance·LRT 테이블까지 전부 보관
                           (필요할 때 노트북에서 reporting.* 함수 재호출 가능)
    """
    model_types = model_types or cfg.MODEL_TYPES

    if checkpoint or auto_save:
        result_dir = _resolve_result_dir(target_folder, result_folder)
        os.makedirs(result_dir, exist_ok=True)

    summary_records = []
    detail_results = {}

    for group_name, group_cfg in cfg.GROUPS.items():
        if verbose:
            print(f"\n{'='*60}\nGroup: {group_name}\n{'='*60}")

        if group_cfg['filter'] is None:
            df_train_g, df_test_g = df_train, df_test
        else:
            filter_col, filter_val = group_cfg['filter']
            df_train_g = df_train[df_train[filter_col] == filter_val].copy()
            df_test_g = df_test[df_test[filter_col] == filter_val].copy()

        detail_results[group_name] = {}

        for model_idx in range(len(group_cfg['model_names'])):
            model_name, feature_cols, covariate_included = get_features_for_model(
                group_name, model_idx, covariate_column_map
            )
            group_map = build_group_map(covariate_included, covariate_column_map)

            X_train = df_train_g[feature_cols]
            y_train = df_train_g['y_amyloid']
            X_test = df_test_g[feature_cols]
            y_test = df_test_g['y_amyloid']

            if verbose:
                print(f"\n--- {model_name} (n_features={len(feature_cols)}, "
                      f"covariates={covariate_included}, n_train={len(X_train)}, n_test={len(X_test)}) ---")

            detail_results[group_name][model_name] = {}

            for model_type in model_types:
                if verbose:
                    print(f"  [{model_type}] 학습 중...")

                result_entry = _run_one_combination(
                    model_type, X_train, y_train, X_test, y_test, model_name, group_map,
                )
                detail_results[group_name][model_name][model_type] = result_entry

                cv = result_entry['cv_metrics']
                summary_records.append({
                    'Group': group_name, 'Model': model_name, 'model_type': model_type,
                    'n_features': len(feature_cols), 'n_folds_used': len(result_entry['fold_results']),
                    'threshold': result_entry['threshold'],
                    'cv_f1_mean': cv['f1'].mean() if len(cv) > 0 else np.nan,
                    'cv_f1_std': cv['f1'].std() if len(cv) > 0 else np.nan,
                    'cv_auc_mean': cv['auc'].mean() if len(cv) > 0 else np.nan,
                    'cv_auc_std': cv['auc'].std() if len(cv) > 0 else np.nan,
                    'train_auc': result_entry['train_perf']['auc'],
                    'test_auc': result_entry['test_perf']['auc'],
                    'test_f1': result_entry['test_perf']['f1'],
                })

        if checkpoint:
            save_results(pd.DataFrame(summary_records), detail_results,
                         result_folder=result_folder, target_folder=target_folder)
            if verbose:
                print(f"\n[checkpoint] '{group_name}' 그룹까지 결과 저장 완료 -> {result_dir}")

    summary_df = pd.DataFrame(summary_records)

    if auto_save:
        save_results(summary_df, detail_results,
                     result_folder=result_folder, target_folder=target_folder)
        print(f"\n최종 결과가 '{result_dir}'에 저장되었습니다.")

    return summary_df, detail_results


# ============================================================
# 4. Baseline vs Full-covariate 모델 DeLong 비교 (선택적 후속 분석)
# ============================================================

def compare_baseline_vs_full(detail_results, group_name, model_type,
                              df_test, covariate_column_map):
    """
    한 Group의 baseline 모델(M1)과 full-covariate 모델(마지막 M)의 test set AUC를
    DeLong test로 비교. "covariate를 전부 추가하는 게 통계적으로 유의미한 성능
    변화를 주는가"에 대한 답.

    같은 test set(df_test_g, 같은 피험자 순서)에 대해 두 모델의 예측확률을
    다시 계산해서 비교 - detail_results에 이미 저장된 test_prob을 재사용하되,
    두 모델의 test set 피험자 구성이 같아야 하므로(같은 Group) 그 부분만 확인.

    반환: stats_tests.delong_test()의 결과 dict
    """
    model_names = list(detail_results[group_name].keys())
    baseline_name, full_name = model_names[0], model_names[-1]

    baseline_entry = detail_results[group_name][baseline_name][model_type]
    full_entry = detail_results[group_name][full_name][model_type]

    # 두 모델 모두 같은 Group의 test set을 썼으므로 y_true는 동일해야 함
    filter_ = cfg.GROUPS[group_name]['filter']
    df_test_g = df_test if filter_ is None else df_test[df_test[filter_[0]] == filter_[1]]
    y_test = df_test_g['y_amyloid'].values

    return stats_tests.delong_test(
        y_test, baseline_entry['test_prob'], full_entry['test_prob'],
        label_1=baseline_name, label_2=full_name,
    )


# ============================================================
# 5. 결과 저장 / 로드
# ============================================================

def save_results(summary_df, detail_results, target_folder, result_folder):
    """summary_df -> csv, detail_results -> pickle 저장. '../{target_folder}/results/{result_folder}'에 저장됨."""
    result_dir = _resolve_result_dir(target_folder, result_folder)
    os.makedirs(result_dir, exist_ok=True)

    summary_path = os.path.join(result_dir, 'summary.csv')
    detail_path = os.path.join(result_dir, 'detail_results.pkl')

    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    with open(detail_path, 'wb') as f:
        pickle.dump(detail_results, f)

    print(f"저장 완료: {summary_path}")
    print(f"저장 완료: {detail_path}")
    return result_dir


def load_results(target_folder, result_folder):
    """
    save_results()로 저장된 결과를 다시 불러옴.
    target_folder, result_folder : save_results에 넘겼던 것과 동일한 값을 넘기면
                                   같은 '../{target_folder}/results/{result_folder}' 경로를 찾음.
    """
    result_dir = _resolve_result_dir(target_folder, result_folder)
    summary_df = pd.read_csv(os.path.join(result_dir, 'summary.csv'))
    with open(os.path.join(result_dir, 'detail_results.pkl'), 'rb') as f:
        detail_results = pickle.load(f)
    return summary_df, detail_results


# ============================================================
# 6. 전체 결과 리포트 (성능 + OR/LRT + importance plot 전부 출력)
# ============================================================

def print_full_report(detail_results, model_types=('LR', 'RF')):
    """
    detail_results를 Group -> model_type -> Model 순서로 순회하며 출력.
    (같은 model_type끼리 M1~M6를 쭉 이어서 보여줘서 성능 추이 비교가 쉬움)

    각 Model마다: CV 성능표 -> (LR이면 ROC curve, Youden threshold) ->
    train/test classification_report 전체 -> (LR이면 OR/LRT, RF면 importance/SHAP)
    을 출력한 뒤, Group x model_type 단위로 M1~M6 전체의 train/test AUC
    비교 막대그래프를 마지막에 한 번 그림.
    """
    import matplotlib.pyplot as plt

    for group_name, models in detail_results.items():
        print(f"\n{'#'*70}\n# Group: {group_name}\n{'#'*70}")
        model_names = list(models.keys())

        for model_type in model_types:
            print(f"\n{'='*70}\n[{group_name}] model_type = {model_type}\n{'='*70}")

            train_perf_records, test_perf_records = [], []

            for model_name in model_names:
                res = models[model_name].get(model_type)
                if res is None:
                    continue

                print(f"\n--- {group_name}/{model_name} [{model_type}] CV 성능 (threshold={res['threshold']:.3f}) ---")
                print(res['cv_metrics'])

                if model_type == 'LR':
                    # ROC curve (모든 threshold) - Youden's Index 근거 확인용, CV 단계 전용
                    reporting.plot_roc_curve(
                        res['fold_results'], title=f'{group_name} - {model_name} - CV ROC Curve'
                    )
                    plt.show()

                # train(refit)/test 성능 - classification_report 전체 출력
                reporting.report_test_performance(
                    res['y_train'], res['train_prob'], res['threshold'],
                    label=f'{group_name}/{model_name} [{model_type}] train(refit)', verbose=True,
                )
                reporting.report_test_performance(
                    res['y_test'], res['test_prob'], res['threshold'],
                    label=f'{group_name}/{model_name} [{model_type}] test', verbose=True,
                )
                train_perf_records.append({'label': model_name, 'auc': res['train_perf']['auc']})
                test_perf_records.append({'label': model_name, 'auc': res['test_perf']['auc']})

                if model_type == 'LR':
                    print(f"\n--- {group_name}/{model_name} [LR] 개별 OR table (Wald test) ---")
                    print(res['or_summary'])
                    if 'lrt_summary' in res:
                        print(f"\n--- {group_name}/{model_name} [LR] 그룹 LRT p-value ---")
                        print(res['lrt_summary'])

                elif model_type == 'RF':
                    print(f"\n--- {group_name}/{model_name} [RF] best_params: {res['rf_best_params']} ---")

                    # 개별 + 그룹(joint) importance 통합 버전만 report (표 + 그래프)
                    # importance_type 컬럼으로 어떤 행이 group(joint)인지 표에서 바로 확인 가능
                    print(f"\n--- {group_name}/{model_name} [RF] Feature Importance "
                          f"(개별 + Group joint 통합, importance_type 컬럼 참고) ---")
                    print(res['combined_importance'].to_string(index=False))

                    reporting.plot_combined_importance(
                        res['combined_importance'],
                        title=f'{group_name} - {model_name} - Feature Importance (개별 + Group joint)',
                    )
                    plt.show()

                    if 'shap_result' in res:
                        reporting.plot_shap_summary(
                            res['shap_result'], title=f'{group_name} - {model_name} - SHAP (Test set)'
                        )
                        plt.show()

            # Group x model_type 단위 - M1~M6 전체의 train/test AUC 비교 (1.1.2 / 1.2 요구사항)
            reporting.plot_auc_comparison(
                train_perf_records, title=f'{group_name} / {model_type} - Train(refit) AUC Comparison'
            )
            plt.show()
            reporting.plot_auc_comparison(
                test_perf_records, title=f'{group_name} / {model_type} - Test AUC Comparison'
            )
            plt.show()