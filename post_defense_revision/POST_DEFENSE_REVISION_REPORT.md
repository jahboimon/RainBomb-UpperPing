# Rain Bomb post-defense scientific revision report

## Scope and freeze protection

This package implements R1–R5 as a separate scientific revision. It reads the canonical repository at the location recorded in `config.json` and writes only beneath `post_defense_revision/`. The original folder is a Git repository, but this ChatGPT project mirror is not; therefore separation is implemented as a new workspace rather than a branch. SHA-256 hashes were calculated before and after the run for canonical rainfall CSV/Parquet, frozen P99, official Def5 target, station metadata, baseline cleaned rainfall, and ERA5 canonical. All hashes remained identical. No model was trained, no frozen output was overwritten, and 2025 was not used for threshold selection or tuning.

## R1 — Zero versus missing: PASS with a zero-run caveat

Direct inspection of 4,393,680 canonical station-hours found 3,499,499 numeric zeros, 679,973 missing values, and 214,208 positive values. In the canonical CSV all 679,973 missing observations are blank fields. None of the declared telemetry sentinels (`-999`, `999999`, `9999`, `-`) remains in the rainfall column, and none was counted as zero.

There are no duplicate station–datetime keys and no absent timestamps within any represented station-year span. Some station-years consist entirely of missing rainfall; this is a real availability limitation, not zero rain. The Def5 audit found 3,554,919 rows with exact hourly continuity and finite rainfall at E−3…E+3. Recomputing Def5 from those rows produced 1,123 events and an exact key match to the official target.

Long numeric-zero runs exist (maximum exceeds 3,000 hours at some stations). Under the documented source convention these remain observed no-rain, and the exact Def5 reproduction confirms they were not created by the final target builder. Still, a long zero run alone cannot prove sensor validity. The thesis should disclose this as a limitation and, if raw quality flags become available, crosscheck the longest runs. Until then their physical status is `UNCERTAIN`, not missing by assumption.

Key evidence: `R1_zero_missing/R1_zero_missing_summary.csv`, `R1_station_year_audit.csv`, and `R1_zero_run_audit.csv`.

## R2 — P99 robustness

Annual station-specific P99 was calculated from wet hours (`Rainfall > 0`) for 2012–2024H1, using the same linear quantile convention as the frozen threshold. The table contains 446 station-years across all 40 stations. Across stations, the median coefficient of variation of annual P99 is 0.328 (IQR 0.261–0.460). Across station-years, the median absolute difference from frozen P99 is 3.473 mm/h; the 10th–90th percentile of the annual/frozen ratio is 0.539–1.271. Thirty station-years have fewer than 100 wet hours and therefore yield particularly unstable annual P99 estimates.

The variability is scientifically meaningful, but it does not justify changing the target. A yearly threshold would change the event definition from year to year and make interannual counts harder to compare. The frozen 2012–2024H1 station P99 remains the primary threshold; annual P99 is robustness evidence and should be shown with the low-sample caveat.

## R3 — Event-definition sensitivity

| Scenario | Events | Added vs S0 | Removed vs S0 | Jaccard with S0 |
|---|---:|---:|---:|---:|
| S0 Def5 | 1,123 | 0 | 0 | 1.000 |
| Antecedent ≤10 mm | 1,289 | 166 | 0 | 0.871 |
| Antecedent ≤20 mm | 1,429 | 306 | 0 | 0.786 |
| Post rain below station heavy threshold | 1,128 | 5 | 0 | 0.996 |
| Post restriction removed, finite window retained | 1,185 | 62 | 0 | 0.948 |
| Station wet-hour P99.5 intensity | 717 | 0 | 406 | 0.638 |
| Station wet-hour P99.9 intensity | 170 | 0 | 953 | 0.151 |

The grid is defensible rather than arbitrary: antecedent limits are nested multiples of the frozen 5-mm criterion; post-event variants use the station heavy-rain threshold or remove only the intensity rule while retaining a finite window; stronger intensity uses development-period station percentiles. Month, hour, station, event lists, and representative seven-hour hydrographs are supplied for every family. These results show that antecedent choice has a larger effect than a modest post-event relaxation, while extreme percentile tightening changes the event population substantially. S0 remains the baseline.

## R4 — Spatial interpolation

The interpolated variable is the development-period Def5 event rate per 100,000 evaluable station-hours. It is continuous and denominator-aware; no categorical SHAP label is interpolated. Forty station locations were projected to UTM 47N. Leave-one-station-out CV gave:

| Method | MAE | RMSE | Bias |
|---|---:|---:|---:|
| IDW | 6.210 | 7.733 | 0.170 |
| Ordinary kriging | 6.002 | 7.603 | 0.163 |

Ordinary kriging performs slightly better, but the margin is small. The fitted exponential variogram has an approximate 55.4-km range and should be viewed as a pragmatic surface model with only 40 points. The map masks 3,717 grid cells outside the station convex hull instead of extrapolating; even within the hull, the 90th-percentile nearest-station distance is 24.88 km. The map must be captioned as an **estimated long-term event-rate surface**, not an observed rainfall field. Sparse/high-terrain zones remain high-risk for interpretation.

## R5 — Atmospheric interpretation and predictors

The 850-hPa variables are defensible as grid-scale lower-tropospheric moisture and wind context for monsoon and terrain-flow interaction, but they do not resolve station-scale valleys or ridges. The highest project station elevation is about 1,180 m, so `valid_850` and the terrain limitation must remain explicit. The current ERA5 canonical does not include the main requested candidates. Ranked future candidates are TCWV, CAPE, multi-level vertical velocity, total-column cloud liquid/ice, profile cloud water/humidity, and cloud cover. None is promoted here. See `R5_atmospheric_predictor_review.md`.

## Revised scientific freeze

### REQUIRED revision

- Add the R1 missing/zero evidence and exact seven-hour eligibility result to methods/QC.
- State that rates use evaluable station-hours and disclose fully missing station-years.
- Add R2 annual-P99 robustness figure/table without replacing frozen P99.
- Add the R3 scenario comparison and state explicitly that S0 remains official.
- Replace any categorical SHAP interpolation with the continuous Def5-rate surface; show station points, mask outside-hull areas, and add the estimated-surface warning.
- Strengthen the 850-hPa terrain/resolution limitation.

### Sensitivity only

- Annual P99 distributions.
- Antecedent, sustained/post-event, and stronger-intensity scenario grid.
- IDW-versus-kriging comparison.

### Discussion only

- Long zero-run ambiguity where no independent raw flag proves sensor status.
- Physical interpretation of 850-hPa moisture/wind.

### Future work

- Independent raw telemetry quality-flag review for longest zero runs.
- Versioned ERA5 extraction and development-only ablation for TCWV, CAPE, omega, and cloud variables.
- Denser gauges or terrain-aware spatial models before operational use of the surface.

## Thesis change list

1. **Data/QC section:** insert R1 summary and missing-code statement; distinguish structural availability, missing observation, and observed zero.
2. **Event definition section:** retain Def5 formula; add R2 and R3 as robustness/sensitivity subsections.
3. **Results:** add annual/frozen P99 plot, scenario comparison table, representative hydrograph, station-rate map, and LOSO CV table.
4. **Discussion:** add zero-run uncertainty, station coverage/extrapolation risk, and 850-hPa terrain caveat.
5. **Future work:** add ranked atmospheric candidates; do not report them as tested predictors.

Scientific revisions are now frozen at the evidence/decision level. The thesis itself has intentionally not been edited.
