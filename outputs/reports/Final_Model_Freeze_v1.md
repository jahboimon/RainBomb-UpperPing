# Final Model Freeze v1

Freeze timestamp (UTC): `2026-09-08T03:54:09.171955+00:00`

**FINAL MODEL:** XGBoost Compact20

**DEVELOPMENT DATA:** May–October 2020–2024

**FINAL EVALUATION:** May–October 2025

**NUMBER OF RAW MODEL INPUTS:** 20

**FINAL BOOSTING ROUNDS:** 53

**THRESHOLD:** No operational threshold selected.

Threshold 0.5 is diagnostic only.

## Frozen scientific design

Predictors at station-hour t forecast `Target_Def5_tPlus1` at t+1 hour. The final development fit combines the frozen Train 2020–2023 and Validation 2024 May–October rows. Evaluation 2025 is excluded from fitting.

## Frozen predictors

1. `Rain_t`
2. `RainSum_3h`
3. `RainSum_6h`
4. `RH_t`
5. `RHMin_6h`
6. `Temp_t`
7. `TempMax_6h`
8. `Hour_sin`
9. `Hour_cos`
10. `Month_sin`
11. `Month_cos`
12. `Elevation_m`
13. `SUHI_5K`
14. `BuiltUp_Binary`
15. `d2m`
16. `u10`
17. `v10`
18. `q_850`
19. `u_850`
20. `v_850`

## Frozen hyperparameters

- `objective = binary:logistic`
- `eval_metric = aucpr`
- `tree_method = hist`
- `n_estimators = 53`
- `learning_rate = 0.05`
- `max_depth = 4`
- `min_child_weight = 5`
- `subsample = 0.8`
- `colsample_bytree = 0.8`
- `reg_lambda = 1.0`
- `reg_alpha = 0.0`
- `random_state = 154499`
- `n_jobs = -1`

No early stopping is used in the final refit. The 53 rounds equal Phase 5's zero-based `best_iteration = 52` plus one.

## Final class weighting

The frozen rule is `scale_pos_weight = development negatives / development positives`. For 654,759 negatives and 386 positives, the derived value is `1696.266839378238`. This applies the frozen rule to the combined development population and is not tuning.

## Population and provenance

- Development rows: 655,145
- Development positives: 386
- Development negatives: 654,759
- Development stations: 40
- Season: May–October
- Model-ready input SHA256: `d77c69a86f1b1251e09f37451a59f68e0a25c7f9307542b549b17e8b2e7d8a83`
- Random seed: 154499
- No threshold optimization
- No Evaluation use in model fitting

## Evaluation independence declaration

Evaluation 2025 has not informed:

- model selection
- feature selection
- hyperparameters
- number of boosting rounds
- class weighting rule
- threshold

This manifest is written before the final fitted model and before Evaluation 2025 is loaded or scored.
