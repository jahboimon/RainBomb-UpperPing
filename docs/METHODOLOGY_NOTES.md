# Methodology Notes

## Def5

Def5 is frozen and is not recalculated in this repository. At the event hour `E`:

1. `Rain(E)` is at least the frozen station wet-hour P99.
2. `Rain(E)` is at least 20 mm/h.
3. rainfall over `E−1:E−3` totals at most 5 mm.
4. each of `E+1:E+3` remains below 20 mm/h (`Post3_Max < 20`).
5. required values are finite and nonmissing; missing is never replaced with zero.

The forecast table uses information at `t` to predict Def5 at `t+1`.

## Model design

Compact20 is recorded in `config/features.yaml`. The season and roles are recorded in `config/splits.yaml`. XGBoost was selected using Validation 2024 average precision, refitted on Train+Validation, and evaluated once on 2025.

The score is not described as a calibrated probability. Threshold 0.5 is diagnostic, not an operational warning threshold.

## SHAP and spatial interpretation

SHAP describes model attribution, not causality. Weighted KDE layers in the web explorer show spatial concentration of station attribution; they are not rainfall intensity, storm extent, risk, or a continuous physical field.

ERA5 timestamps were converted UTC→ICT before joining. Station-source timezone remains unverified; stored datetimes must not be converted again by assumption.
