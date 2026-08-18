# SCST(Seoul Cognitive Status Test) Biomarker Analysis

> [!Note]
> - Program: [CUop](https://cuop.kaist.ac.kr/) 과학기술특성화대학 회사연계 프로그램
> - Company: [(주)뷰브레인헬스케어](https://beaubrain.bio/)
> - Department: 기업부설연구소 솔루션개발 인지팀
> - Position: Intern
> - Period: June 23, 2026 - August 19, 2026

## Goals

- [X] Three Cognitive Status Diagnosis based on [SCST (Seoul Cognitive Status Test))](https://beaubrain.bio/publication/scst/)
- [X] Binary Amyloid Beta Status Diagnosis based on [SCST (Seoul Cognitive Status Test))](https://beaubrain.bio/publication/scst/)
- [X] Pipeline: Raw SCST dataset $\to$ P (Amyloid Beta == Positive | New Patient )
- [ ] Search NEW important features (biomarkers)
  - Compared to SNSB (Seoul Neuropsychological Screening Battery), SCST is Tablet-Based Digital Cognitive Test so that there are lots of features from the log.
  - However, since the detailed features have been reported at the mid of this test, available parients are far tiny.
  - That's why I could not find NEW biomarkers. I summarized additional features for future works.

## Model Structure

> [!Important]
> - Logistic Regression with Odds Ratio, CI and p-value
> - Random Forest with SHAP values
> - To analysis various models with each covariate features, I added features for each step.

### Group 1: `1st_cognitive_status` == All (covariate, SCD + MCI + Dementia)

| Model | Covariate (demo features) | composite scores |
| :---: | :-----------------------: | :--------------: |
| M1 (baseline) | &nbsp; | O |
| M2 | `1st_cognitive_status` | O |
| M3 | `1st_cognitive_status` + `sex` | O |
| M4 | `1st_cognitive_status` + `sex` + `AGE` | O |
| M5 | `1st_cognitive_status` + `sex` + `AGE` + `education_year` | O |
| M6 | `1st_cognitive_status` + `sex` + `AGE` + `education_year` + `APOE` | O |

### Group 2: `1st_cognitive_status` == SCD

| Model | Covariate (demo features) | composite scores |
| :---: | :-----------------------: | :--------------: |
| M1_SCD (baseline) | &nbsp; | O |
| M2_SCD | `sex` | O |
| M3_SCD | `sex` + `AGE` | O |
| M4_SCD | `sex` + `AGE` + `education_year` | O |
| M5_SCD | `sex` + `AGE` + `education_year` + `APOE` | O |

### Group 3: `1st_cognitive_status` == MCI (important)

| Model | Covariate (demo features) | composite scores |
| :---: | :-----------------------: | :--------------: |
| M1_MCI (baseline) | &nbsp; | O |
| M2_MCI | `sex` | O |
| M3_MCI | `sex` + `AGE` | O |
| M4_MCI | `sex` + `AGE` + `education_year` | O |
| M5_MCI | `sex` + `AGE` + `education_year` + `APOE` | O |

### Group 4: `1st_cognitive_status` == Dementia

| Model | Covariate (demo features) | composite scores |
| :---: | :-----------------------: | :--------------: |
| M1_Dementia (baseline) | &nbsp; | O |
| M2_Dementia | `sex` | O |
| M3_Dementia | `sex` + `AGE` | O |
| M4_Dementia | `sex` + `AGE` + `education_year` | O |
| M5_Dementia | `sex` + `AGE` + `education_year` + `APOE` | O |

In the Goal 3, I shorten the M2-M4 for demographic features.

### Group 1: `1st_cognitive_status` == All (covariate, SCD + MCI + Dementia)

| Model | Covariate (demo features) | composite scores |
| :---: | :-----------------------: | :--------------: |
| M1_All | &nbsp; | O |
| M2_All | `1st_cognitive_status` | O |
| M3_All | `1st_cognitive_status` + `sex` + `AGE` + `education_year` | O |
| M4_All | `1st_cognitive_status` + `sex` + `AGE` + `education_year` + `APOE` | O |

### Group 2: `1st_cognitive_status` == SCD + MCI

| Model | Covariate (demo features) | composite scores |
| :---: | :-----------------------: | :--------------: |
| M1_SCD_MCI | &nbsp; | O |
| M2_SCD_MCI | `sex` + `AGE` + `education_year` | O |
| M3_SCD_MCI | `sex` + `AGE` + `education_year` + `1st_cognitive_status` | O |
| M4_SCD_MCI | `sex` + `AGE` + `education_year` + `1st_cognitive_status` + `APOE` | O |

### Group 3: `1st_cognitive_status` == Dementia

| Model | Covariate (demo features) | composite scores |
| :---: | :-----------------------: | :--------------: |
| M1_Dementia | &nbsp; | O |
| M2_Dementia | `sex` + `AGE` + `education_year` | O |
| M3_Dementia | `sex` + `AGE` + `education_year` + `APOE` | O |

---

### Goal 1. Three Cognitive Status Diagnosis



### Goal 2. Binary Amyloid Beta Status Diagnosis

### Goal 3. Pipeline: Raw SCST data $\to$ Probability of Amyloid Beta Status

ppt 4가지 케이스 표

GIF 만들어서 삽입

html <- github pages
