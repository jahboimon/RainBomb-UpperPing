# Phase 7 — SHAP / Explainable AI for Final XGBoost

## 1. Frozen final model being explained

This phase explains the frozen **XGBoost Compact20** model saved at `03_OUTPUT/06_Final_Model_Evaluation/XGB_Final_Compact20_v1.json`. The saved model contains 53 boosting rounds and uses the frozen 20 predictors in their original order. It was loaded read-only. No retraining, hyperparameter change, threshold optimization, feature selection, or model variant was performed.

## 2. Evaluation population

TreeSHAP contributions were computed for all 121,296 Evaluation rows from May–October 2025: 95 actual Def5 positives across 38 stations. Threshold-0.5 outcomes remain TP=71, FP=24,193, TN=97,008, and FN=24.

## 3. SHAP method and output scale

Native XGBoost TreeSHAP was computed with `Booster.predict(DMatrix, pred_contribs=True)`. Each feature contribution and `SHAP_Bias` explains the **raw model margin / log-odds**. Individual SHAP values are not percentage-point probability changes or percentage risk increases. Positive SHAP values contribute to a higher raw model score; negative values contribute to a lower raw model score. These are model attributions, not causal effects.

## 4. Reconstruction validation

For every row, the 20 feature contributions plus `SHAP_Bias` were summed and compared with XGBoost's raw margin. Applying a sigmoid to the reconstructed margin was then compared with the frozen `XGB_Final_Score`.

- Maximum absolute raw-margin error: **1.49041443365e-06**
- Maximum absolute model-score error: **1.64687537663e-07**
- Target tolerance: **1e-6**
- Result: **PASS_WITH_CAVEATS**. The raw-margin discrepancy is limited to float32 summation rounding; the sigmoid-reconstructed frozen score remains within 1e-6.

## 5. Global feature ranking

Across all Evaluation station-hours, **u_850** had the largest mean absolute attribution (0.492165 on the raw-margin scale). The complete 20-feature ranking is stored in `Global_SHAP_Importance_v1.csv` and plotted in `Global_SHAP_MeanAbs_v1.png`.

| Rank | Feature | Mean |SHAP| | Median |SHAP| | Mean signed SHAP |
|---:|---|---:|---:|---:|
| 1 | u_850 | 0.492165 | 0.240483 | -0.242309 |
| 2 | Hour_sin | 0.403583 | 0.415909 | -0.128048 |
| 3 | q_850 | 0.361586 | 0.259751 | -0.209299 |
| 4 | Rain_t | 0.231432 | 0.215819 | -0.138445 |
| 5 | u10 | 0.142928 | 0.136760 | -0.031809 |
| 6 | Temp_t | 0.123840 | 0.096274 | -0.078879 |
| 7 | RainSum_3h | 0.106988 | 0.050769 | -0.062377 |
| 8 | RH_t | 0.088887 | 0.056197 | -0.052086 |
| 9 | d2m | 0.088171 | 0.043027 | -0.051440 |
| 10 | TempMax_6h | 0.075543 | 0.061237 | -0.050953 |

## 6. Positive-event feature ranking

Because Def5 is rare, all 95 actual Def5 rows were also summarized separately. **Hour_sin** ranked first for positive-event mean absolute attribution (0.384622). Differences from the all-row ranking describe how the frozen model behaved on observed Def5 rows; they do not establish causation.

| Rank | Feature | Mean |SHAP| | Median |SHAP| | Mean signed SHAP |
|---:|---|---:|---:|---:|
| 1 | Hour_sin | 0.384622 | 0.417834 | 0.201872 |
| 2 | Rain_t | 0.366147 | 0.201498 | 0.126851 |
| 3 | u_850 | 0.282327 | 0.216899 | 0.070665 |
| 4 | q_850 | 0.268757 | 0.232999 | 0.035264 |
| 5 | u10 | 0.099994 | 0.099074 | 0.011911 |
| 6 | RainSum_3h | 0.091767 | 0.038475 | 0.057178 |
| 7 | Temp_t | 0.090972 | 0.077833 | -0.002054 |
| 8 | RH_t | 0.068162 | 0.040709 | 0.004748 |
| 9 | TempMax_6h | 0.053291 | 0.036966 | -0.006493 |
| 10 | d2m | 0.053141 | 0.036357 | 0.018500 |

## 7. Beeswarm interpretation

The overall beeswarm uses a reproducible simple random sample of 5,000 Evaluation rows (`random_state=154499`) and contains 8 positives; positives were not oversampled. The positive-event beeswarm uses all 95 actual Def5 rows. Horizontal position is signed contribution to the raw margin, while color shows low-to-high predictor value within each feature. The plots show association with higher or lower model output, not physical effects.

## 8. TP / FP / FN attribution patterns

Mean absolute attributions by frozen threshold-0.5 outcome are descriptive:

- **TP (71 rows):** Rain_t (0.4179), Hour_sin (0.3723), q_850 (0.2502), u_850 (0.2170), RainSum_3h (0.1037).
- **FP (24,193 rows):** Hour_sin (0.3985), Rain_t (0.2754), q_850 (0.2234), u_850 (0.2205), u10 (0.1147).
- **TN (97,008 rows):** u_850 (0.5601), Hour_sin (0.4049), q_850 (0.3961), Rain_t (0.2203), u10 (0.1500).
- **FN (24 rows):** u_850 (0.4757), Hour_sin (0.4210), q_850 (0.3236), Rain_t (0.2131), u10 (0.1293).

Representative local examples were selected mechanically, never for scientific attractiveness:

- **TP:** station TB0007, target 2025-06-12 18:00:00, score 0.698543; selected as the row closest to the TP median score (0.698543).
- **FP:** station DOCI, target 2025-06-12 17:00:00, score 0.637938; selected as the row closest to the FP median score (0.637938).
- **FN:** station CHM004, target 2025-09-13 18:00:00, score 0.370983; selected as the row closest to the FN median score (0.373678).

The outcome-specific tables and local plots should be read as differences in model use among prediction outcomes. They do not explain why rainfall occurred.

## 9. Station-level attribution summary

Station-feature summaries cover all 38 Evaluation stations. Each station's dominant SHAP feature is the feature with the highest station-level mean |SHAP|. Dominant-feature frequencies are: u_850 (24 stations), Hour_sin (8 stations), q_850 (6 stations). This label identifies the frozen model's largest average attribution at a station, not a dominant physical cause. The long and summary tables are ready as point-station inputs for Phase 8 after human review.

## 10. RainSum_3h definitional caution

Def5 includes an antecedent-rainfall criterion. Therefore, any high attribution for `RainSum_3h` may partly reflect its definitional relationship with the operational target. Its SHAP importance is not independent evidence of a physical causal mechanism.

## 11. BuiltUp and SUHI interpretation caution

`BuiltUp_Binary` is a static land-cover context variable. Its attribution does not establish a causal relationship between urbanization and the operational target. `SUHI_5K` is `LST_Median - Local_Rural_LST_Median_5km` in °C: a local surface-temperature anomaly relative to rural background, not absolute urban temperature. ERA5 variables likewise show how the model used atmospheric fields and do not provide causal meteorological attribution.

## 12. Limitations

- SHAP describes the fitted model, not the data-generating process and not causation.
- Native contributions are on the raw-margin/log-odds scale; they are not calibrated probability changes.
- Correlated predictors can share or redistribute attribution, so rankings should not be read as unique physical importance.
- Global summaries are dominated by negative station-hours; the 95-positive summary is more event-focused but has a small sample.
- Station-level summaries reflect unequal local event counts and this single May–October 2025 evaluation period.
- Threshold 0.5 remains diagnostic; no operating threshold was selected.

## 13. Readiness for Phase 8

Phase 7 passes the material reconstruction and completeness checks with the documented float32 raw-margin rounding caveat. The outputs are ready for **human review**. Phase 8 has not been started; its point-station spatial analysis remains gated on explicit approval of this Phase 7 result.

## STATUS

PASS_WITH_CAVEATS
