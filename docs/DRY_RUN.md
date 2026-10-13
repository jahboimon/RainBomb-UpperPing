# Dry-run Copy and Rename Plan

Destination check before creation: `D:/STUDY_GEO/2569_1/154499/RainBomb_Research_GitHub` did not exist.

No destructive command is required. Operations are limited to creating new directories/files and copying selected source artifacts.

## Scientific scripts

| Source | Planned repository path |
|---|---|
| `RainBomb_FinalAnalysis_v2/02_PROCESS/01_DataPeriod_QC/01_data_period_qc_v1.py` | `src/01_qc/qc_data.py` |
| `.../02_Seasonal_Scope/01_define_seasonal_scope_v1.py` | `src/01_qc/define_seasonal_scope.py` |
| `.../03_Seasonal_Modeling_Universe/01_build_seasonal_modeling_universe_v1.py` | `src/04_model_ready/build_modeling_universe.py` |
| `.../04_Feature_Architecture/01_review_feature_architecture_v1.py` | `src/03_features/review_feature_architecture.py` |
| `.../05_Model_Comparison/01_build_modelready_v2.py` | `src/04_model_ready/build_model_table.py` |
| `.../05_Model_Comparison/02_train_compare_models_v1.py` | `src/05_model_compare/compare_models.py` |
| `.../06_Final_Model_Evaluation/01_freeze_refit_final_xgb_v1.py` | `src/06_final_model/train_final_xgb.py` |
| `.../06_Final_Model_Evaluation/02_evaluate_final_xgb_2025_v1.py` | `src/06_final_model/evaluate_2025.py` |
| `.../07_SHAP_XAI/01_compute_final_xgb_shap_v1.py` | `src/07_shap/compute_shap.py` |
| `.../07_SHAP_XAI/02_summarize_shap_v1.py` | `src/07_shap/summarize_shap.py` |
| `.../07_SHAP_XAI/03_shap_dependence_diagnostics_v1.py` | `src/07_shap/shap_dependence.py` |
| `.../08_Spatial_Analysis/01_build_station_spatial_summary_v1.py` | `src/08_spatial/build_spatial_summary.py` |
| `.../08_Spatial_Analysis/02_create_final_spatial_maps_v1.py` | `src/08_spatial/make_spatial_figures.py` |

The Def5 and upstream feature builders remain referenced by manifests/documentation rather than copied as runnable GitHub code in this release. Rebuilding canonicals/Def5 is explicitly outside the normal quick-start path.

## Frozen compact artifacts

- Copy final XGBoost JSON, final metrics/training summary, model comparison table, compact SHAP summary tables, and station spatial summary.
- Do not copy Evaluation/SHAP/model-ready Parquet tables; record original paths, hashes, and row counts in `RELEASE_MANIFEST.yaml`.

## Web

- Copy the current hand-authored static application files from `Workplace_Code_Redesign/dist` to `web/app`.
- Copy selected derived browser-display data (JSON, GeoJSON, PNG) to `web/data/derived`; exclude COG/TIFF and caches.
- Adjust only the copied web data paths from `./data/derived` to `../data/derived`.
- Generate new current screenshots after local validation.

## Explicitly excluded

Raw archives, full canonical tables, NetCDF, large Parquet datasets, virtual environments, old model workflows, backup archives, QA render caches, presentation XLSX/NDJSON, and secrets/API credentials.
