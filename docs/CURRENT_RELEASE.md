# Current Scientific Release

The current scientific release is the **RainBomb Final Analysis v2 hourly Compact20 chain**, not the older model workflow retained in `Workplace_Code`.

## Frozen boundary

- Target: official Def5, 1,123 positive station-hours during 2012–2025
- Modeling unit: predictors at `t` forecast `Target_Def5_tPlus1` at `t+1 hour`
- Seasonal population: May–October 2020–2025
- Compact20 table: 776,441 rows
- Development: Train 2020–2023 + Validation 2024
- Final model: `XGB_Final_Compact20_v1.json`, 53 rounds
- Evaluation: 2025, 121,296 rows, 95 positives, 38 stations
- SHAP: native TreeSHAP for the frozen Evaluation table

## Authority order

1. `RELEASE_MANIFEST.yaml` hashes and the frozen artifacts it identifies
2. copied Phase 5–8 reports under `outputs/reports`
3. scripts under `src`
4. web/report derivatives

`XGB_Model_v1.json` from model comparison is a development model. Only `XGB_Final_Compact20_v1.json` is the final refitted model.

## Reproduction policy

Large data remain in read-only external workspaces. A new run must target the configured separate `runtime_root`. Never point `runtime_root` at `RainBomb_FinalAnalysis_v2` or `Workplace_Code`.
