# Phase 6 — Final XGBoost Freeze and Temporal Evaluation 2025

## 1. Why XGBoost was selected in Phase 5

The primary model-selection metric was pre-specified as Average Precision. Phase 5 Validation AP was 0.002351 for Logistic Regression, 0.010527 for Random Forest, and 0.021905 for XGBoost. XGBoost Compact20 was therefore selected and frozen before Evaluation 2025 was accessed. LR and RF were not reconsidered using Evaluation results.

## 2. Frozen Compact20 architecture

The final estimator uses exactly 20 raw predictors: `Rain_t`, `RainSum_3h`, `RainSum_6h`, `RH_t`, `RHMin_6h`, `Temp_t`, `TempMax_6h`, `Hour_sin`, `Hour_cos`, `Month_sin`, `Month_cos`, `Elevation_m`, `SUHI_5K`, `BuiltUp_Binary`, `d2m`, `u10`, `v10`, `q_850`, `u_850`, and `v_850`. No predictor was added, removed, engineered, selected, or transformed beyond the frozen architecture.

## 3. Final training population: May–October 2020–2024

The final refit combined Train 2020–2023 and Validation 2024: 655,145 rows, 386 positives, 654,759 negatives, and 40 stations. Evaluation 2025 was excluded from fitting.

## 4. Final hyperparameters

The frozen model used `objective=binary:logistic`, `eval_metric=aucpr`, `tree_method=hist`, `n_estimators=53`, `learning_rate=0.05`, `max_depth=4`, `min_child_weight=5`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0`, `reg_alpha=0.0`, `random_state=154499`, and `n_jobs=-1`. No Evaluation `eval_set` and no early stopping were used.

## 5. Why final boosting rounds equal 53

Phase 5 early stopping reported zero-based `best_iteration=52` on Validation 2024. The final fixed estimator count is therefore 52 + 1 = 53 boosting rounds. Early stopping was not rerun during the final refit.

## 6. Final scale_pos_weight

The frozen rule was reapplied to the combined development population: 654,759 negatives / 386 positives = 1696.266839378238. This data-derived update is not hyperparameter tuning.

## 7. Model freeze preceded Evaluation scoring

`Final_Model_Freeze_v1.md` was written first. The 53-round final model was then fitted and saved as `XGB_Final_Compact20_v1.json`. The evaluation script verified the manifest, fitted-model hash, round count, feature schema, and file chronology before loading Evaluation 2025.

## 8. Evaluation 2025 population

Evaluation contains 121,296 May–October 2025 rows, 95 positives, 121,201 negatives, and 38 stations. All 20 predictors are complete and finite. Keys are unique and the target lead is exactly one hour.

## 9. Evaluation Average Precision

Final Evaluation AP is **0.00598009**. AP / PR-AUC is the primary temporal-evaluation metric because Def5 is extremely imbalanced.

## 10. Evaluation ROC AUC

Final Evaluation ROC AUC is **0.86402726**. ROC AUC is secondary.

## 11. Evaluation prevalence

Evaluation prevalence is 0.00078321 (0.07832%). AP / prevalence is 7.635379. This ratio is a baseline-relative comparison and does not mean “times more accurate.”

## 12. Threshold-0.5 confusion matrix

Threshold 0.5 is descriptive only. No operational threshold was selected.

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 97,008 | 24,193 |
| Actual 1 | 24 | 71 |

## 13. Threshold-0.5 precision, recall, specificity, and F1

At threshold 0.5, precision is 0.00292615, recall is 0.74736842, specificity is 0.80038944, and F1 is 0.00582947. These values describe one fixed operating point and were not used to alter the model.

## 14. PR curve interpretation

The PR curve shows precision-recall behavior for the frozen final XGBoost across possible score cutoffs and includes the Evaluation prevalence baseline. The curve is descriptive; no cutoff was selected from it.

## 15. ROC curve interpretation

The ROC curve shows sensitivity versus false-positive rate across possible score cutoffs for the frozen final estimator. It is a secondary summary and was not used for post-Evaluation tuning.

## 16. Limitations

- Def5 events remain extremely rare, so apparently small false-positive rates can correspond to many false-positive station-hours.
- The model score is not a calibrated probability.
- Threshold 0.5 is diagnostic and is not an operational warning threshold.
- Phase 5 Validation metrics came from the Train-only development model with Validation early stopping. Phase 6 Evaluation metrics come from a different fitted estimator refit on combined Train + Validation data. They are not a strict paired performance-change test.
- Temporal Evaluation 2025 assesses one future season and does not by itself establish spatial transferability.

## 17. No Evaluation-based tuning occurred

Evaluation 2025 did not inform model selection, feature selection, hyperparameters, boosting rounds, class-weighting rule, or threshold. Whatever the Evaluation result, no model redesign, threshold search, calibration, or post-hoc rescue was performed.

## 18. Readiness for Phase 7 SHAP

Phase 6 is complete and the frozen final model plus Evaluation score table are ready for a separately authorized Phase 7 explanation workflow. No SHAP, feature importance, PDP, spatial SHAP, station SHAP, spatial analysis, or spatial prediction was performed in Phase 6.

## Validation-versus-Evaluation context

Historical Phase 5 XGBoost Validation metrics were AP 0.021905 and ROC AUC 0.819619. They came from the Train-only development model with Validation early stopping. The Phase 6 metrics come from the final 53-round refit trained on combined Train + Validation. The two results are not from the identical fitted estimator and must not be treated as a strict paired performance-change test. No tuning was performed in response to their difference.

## Actual-vs-Predict output

`Evaluation2025_Actual_vs_Predicted_v1.parquet` stores every Evaluation row with the final model score, threshold-0.5 prediction, and TP/FP/TN/FN label. `Evaluation2025_Actual_vs_Predict_Sample_v1.csv` contains 20 rows selected reproducibly within available outcome groups using seed 154499, without selecting examples based on attractive scores.

## STATUS

PASS
