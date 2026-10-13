# File Rename and Copy Map

All sources remained in place. “Path refactor” means only the copied file now reads paths through `config/paths*.yaml`.

## Scientific code

| Original | Repository copy | Change |
|---|---|---|
| `RainBomb_FinalAnalysis_v2/02_PROCESS/01_DataPeriod_QC/01_data_period_qc_v1.py` | `src/01_qc/qc_data.py` | renamed + path refactor |
| `.../02_Seasonal_Scope/01_define_seasonal_scope_v1.py` | `src/01_qc/define_seasonal_scope.py` | renamed + path refactor |
| `.../03_Seasonal_Modeling_Universe/01_build_seasonal_modeling_universe_v1.py` | `src/04_model_ready/build_modeling_universe.py` | renamed + path refactor |
| `.../04_Feature_Architecture/01_review_feature_architecture_v1.py` | `src/03_features/review_feature_architecture.py` | renamed + path refactor |
| `.../05_Model_Comparison/01_build_modelready_v2.py` | `src/04_model_ready/build_model_table.py` | renamed + path refactor |
| `.../05_Model_Comparison/02_train_compare_models_v1.py` | `src/05_model_compare/compare_models.py` | renamed + path refactor |
| `.../06_Final_Model_Evaluation/01_freeze_refit_final_xgb_v1.py` | `src/06_final_model/train_final_xgb.py` | renamed + path refactor |
| `.../06_Final_Model_Evaluation/02_evaluate_final_xgb_2025_v1.py` | `src/06_final_model/evaluate_2025.py` | renamed + path refactor; overwrite guard preserved |
| `.../07_SHAP_XAI/01_compute_final_xgb_shap_v1.py` | `src/07_shap/compute_shap.py` | renamed + path refactor |
| `.../07_SHAP_XAI/02_summarize_shap_v1.py` | `src/07_shap/summarize_shap.py` | renamed + path refactor |
| `.../07_SHAP_XAI/03_shap_dependence_diagnostics_v1.py` | `src/07_shap/shap_dependence.py` | renamed + path refactor |
| `.../08_Spatial_Analysis/01_build_station_spatial_summary_v1.py` | `src/08_spatial/build_spatial_summary.py` | renamed + path refactor |
| `.../08_Spatial_Analysis/02_create_final_spatial_maps_v1.py` | `src/08_spatial/make_spatial_figures.py` | renamed + path refactor |

## Compact frozen outputs

- Final model → `outputs/models/XGB_Final_Compact20_v1.json`
- Evaluation metrics, training summary, model comparison, global/station SHAP summaries, station spatial summary → `outputs/tables/`
- Phase 6–8 reports and codebase audit → `outputs/reports/`

Large Parquet artifacts were not copied; their exact original paths and hashes are in `RELEASE_MANIFEST.yaml`.

## Web

| Original | Repository copy |
|---|---|
| `Workplace_Code_Redesign/dist/index.html` | `web/app/index.html` |
| `dist/app.js` | `web/app/app.js` |
| `dist/maps.js` | `web/app/maps.js` |
| `dist/styles.css` | `web/app/styles.css` |
| `dist/interpretation.js` | `web/app/interpretation.js` |
| `dist/interpretation-ui.js` | `web/app/interpretation-ui.js` |
| `dist/xgb-runtime.js` | `web/app/xgb-runtime.js` |
| `dist/data/derived/**/*.{json,geojson,png}` | `web/data/derived/` preserving relative names |
| `scripts/build_web_data.py` | `web/scripts/build_web_data.py` |
| `scripts/build_web_extension.py` | `web/scripts/build_web_extension.py` |

The copied web app changes only its relative data/PNG base path to match the clean repository layout. Copied derivative builders change only path discovery: external locations now come from `config/paths.yaml`, and outputs target `web/data/derived`.
