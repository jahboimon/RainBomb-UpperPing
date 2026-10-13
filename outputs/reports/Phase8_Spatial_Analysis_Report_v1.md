# Phase 8 — Final Spatial Analysis and Geographic Interpretation

## 1. Spatial datasets used

The analysis reused frozen sources read-only:

- Final Evaluation: `03_OUTPUT/06_Final_Model_Evaluation/Evaluation2025_Actual_vs_Predicted_v1.parquet`
- Phase 7 SHAP rows and station summaries: `03_OUTPUT/07_SHAP_XAI/`
- Model-ready static predictors: `03_OUTPUT/05_Model_Comparison/ModelReady_Seasonal_Compact20_v1.parquet`
- Protected station metadata and coordinates: `D:/STUDY_GEO/2569_1/154499/Workplace_Code/Hourly_rain/RESETWORKPLACE/02_processed/Station_Metadata_Cleaned.csv`
- Context boundary: `D:/STUDY_GEO/2569_1/154499/Workplace_Code/Station_spatial_diagnostics_v1/selected_study_area_boundary_v1.geojson`. This valid WGS 84 polygon is the project-preserved, repaired dissolve of `Study_area/Study_area.shp` (20 source features, original CRS EPSG:32647).
- Frozen long-term station rates: `D:/STUDY_GEO/2569_1/154499/Workplace_Code/Def5_Station_Dynamics_v1.csv`

No source was modified.

## 2. Station coverage

Evaluation 2025 contains 121,296 May–October station-hours, 95 actual Def5 rows, and 38 mapped stations. Frozen metadata covers all 38 Evaluation stations. The boundary covers all 40 protected station coordinates. Evaluation maps show 38 active Evaluation stations; the long-term context map shows all 40 stations present in the frozen 2012–2025 summary.

## 3. Observed Evaluation 2025 Def5 spatial pattern

**Observation:** the highest station Def5 rate was 258.02 per 100,000 Evaluation rows at THCG. The lowest was 0.00 at FOP016, FOP017, MNDM, OMKI, SBRS, VLGE04, VLGE25. These are station-hour observations with station-specific denominators, not estimates for unsampled locations.

## 4. Station model-score pattern

**Model behavior:** the highest median score was 0.314290 at MNDM, and the lowest was 0.151989 at HOTT. Q95 ranged from 0.615166 at OMKI to 0.748461 at VLGE11. These are XGBoost model scores, not calibrated probabilities.

## 5. TP / FP / FN and recall spatial pattern

At the frozen diagnostic threshold 0.5, station counts reconcile to TP=71, FP=24,193, TN=97,008, and FN=24. Recall is estimable at 31 stations and not estimable at 7 stations with no actual positives. Among estimable stations, recall ranged from 0.000 at CHKG, TB0006, WGHG to 1.000 at BNHG, BTHI, CHM002, CHM005, DOCI, DSKT, FOP018, FOP052, HOTT, LEEE, LMPN, SKPG, and 1 more. A station with no actual positive is shown as **not estimable**, never as recall zero.

## 6. False-positive spatial pattern

**Model behavior:** station false-positive rate ranged from 0.1129 at OMKI to 0.3031 at MNDM. These rates describe threshold-0.5 behavior among negative Evaluation rows; the threshold remains diagnostic.

## 7. Station SHAP patterns

**Model attribution:** the heatmap displays station-level mean |SHAP| for the global top ten predictors: `u_850`, `Hour_sin`, `q_850`, `Rain_t`, `u10`, `Temp_t`, `RainSum_3h`, `RH_t`, `d2m`, `TempMax_6h`. Values remain on the raw-margin/log-odds attribution scale. Spatial contrast in this heatmap describes how the frozen model used predictors across sampled stations.

## 8. Dominant SHAP features

The dominant station feature is the predictor with the largest station-level mean |SHAP|. Frequencies were `u_850` at 24 stations, `Hour_sin` at 8 stations, `q_850` at 6 stations. This category is not a physical-driver label and does not establish a causal mechanism at a station.

## 9. Built-up vs Non-built-up descriptive comparison

The frozen mapping is `BuiltUp_Binary=1` for Built-up and `0` for Non-built-up. Pooled Evaluation summaries are:

- **Non-built-up:** 21 stations, 62,994 rows, 46 Def5 rows (73.02 per 100,000), median/Q95 model scores 0.2239/0.7147, recall 0.783, false-positive rate 0.201, mean/median SUHI_5K 1.642/1.552 °C.
- **Built-up:** 17 stations, 58,302 rows, 49 Def5 rows (84.05 per 100,000), median/Q95 model scores 0.2251/0.7163, recall 0.714, false-positive rate 0.198, mean/median SUHI_5K 3.216/3.213 °C.

This comparison is descriptive. No inferential significance test was run, and spatial association does not establish a causal relationship between land-cover context and the target. `SUHI_5K` is the local LST anomaly relative to a 5 km rural background, not absolute urban temperature.

## 10. Long-term Def5 context

The frozen `Def5_Station_Dynamics_v1.csv` was available and reused without rebuilding climatology. It contains 1,123 events and 3,554,919 evaluable station-hours across 40 stations during 2012–2025. The highest long-term rate was 51.72 per 100,000 evaluable station-hours at MNDM; the lowest was 9.94 at OMKI. This all-period characterization is contextual and is distinct from the May–October Evaluation 2025 performance maps.

## 11. Representative event cases

Rows were selected mechanically as the score closest to the median within TP, FP, and FN. They were not chosen for scientific attractiveness.

- **TP:** TB0007, target 2025-06-12 18:00:00, actual 1, prediction 1, model score 0.698543. Top signed contributions: Hour_sin (+0.5955), q_850 (+0.2763), u10 (-0.2122), u_850 (+0.1735), Rain_t (-0.1530).
- **FP:** DOCI, target 2025-06-12 17:00:00, actual 0, prediction 1, model score 0.637938. Top signed contributions: Hour_sin (+0.5481), RH_t (-0.3080), q_850 (+0.2102), u_850 (+0.1571), Rain_t (-0.0722).
- **FN:** CHM004, target 2025-09-13 18:00:00, actual 1, prediction 0, model score 0.370983. Top signed contributions: q_850 (-0.8907), Hour_sin (+0.4104), Rain_t (-0.2129), u_850 (+0.1922), u10 (-0.1725).

## 12. Geographic interpretation

- **Observation** refers only to Def5 counts and denominator-adjusted rates at sampled stations.
- **Model behavior** refers to frozen XGBoost scores and threshold-0.5 errors at sampled stations.
- **Model attribution** refers to SHAP contributions to the frozen model's raw margin.

These three evidence types are kept separate. Maps show station-level values only and do not represent conditions between stations or outside the network.

## 13. Limitations

- No interpolation was performed.
- No gridded risk surface was created.
- All maps show station-level point values with an existing basin polygon used only as context.
- Station values do not represent unsampled areas.
- SHAP attribution is not causation.
- Threshold 0.5 remains diagnostic and was not optimized.
- Model scores are not calibrated probabilities.
- Recall is undefined at stations without actual positives; these stations remain visibly not estimable.
- Rare station-level event counts make rates and recall unstable, especially over one Evaluation season.
- The long-term context spans a different period and seasonal scope from Evaluation 2025 and is not a direct performance comparator.
- Phase 7 carried a small native float32 raw-margin reconstruction rounding caveat; Phase 8 reused its frozen SHAP values without alteration.

## STATUS

PASS_WITH_CAVEATS
