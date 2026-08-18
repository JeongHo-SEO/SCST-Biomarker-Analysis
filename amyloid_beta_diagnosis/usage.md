# 실행방법

```python
import data_prep, modeling, stats_tests, reporting
import run_experiments as run
import config as cfg

### preparing
df, covariate_column_map = data_prep.preprocess(df_composite)
df_train, df_test = data_prep.split_train_test(df)

### training
summary_df, detail_results = run.run_all_experiments(
    df_train, df_test, covariate_column_map,
    target_folder="amyloid_beta_diagnosis",
    result_folder="889_feature_selection_v2",
)

### 저장위치: ../target_folder/results/result_folder



### report
run.print_full_report(detail_results)
```

