# Pipeline and Run Order

## Normal release verification

The recommended action is to verify hashes and tests, not to rebuild frozen upstream products or rerun Evaluation 2025.

```powershell
python -m pytest tests
```

## Full scientific order for a separately authorized clean runtime

1. `src/01_qc/qc_data.py`
2. `src/01_qc/define_seasonal_scope.py`
3. `src/04_model_ready/build_modeling_universe.py`
4. `src/03_features/review_feature_architecture.py`
5. `src/04_model_ready/build_model_table.py`
6. `src/05_model_compare/compare_models.py`
7. `src/06_final_model/train_final_xgb.py`
8. `src/06_final_model/evaluate_2025.py` — frozen one-time step; do not rerun without explicit methodological authorization
9. `src/07_shap/compute_shap.py`
10. `src/07_shap/summarize_shap.py`
11. `src/07_shap/shap_dependence.py`
12. `src/08_spatial/build_spatial_summary.py`
13. `src/08_spatial/make_spatial_figures.py`

## Upstream lineage

Canonical rainfall/RH/temperature, frozen Def5/t+1 target, static GIS predictors, and ERA5 station-hour data are external read-only inputs. Their builders are not part of the routine GitHub run because this release does not rebuild upstream canonicals. See `data/README.md` and `archive_notes/LEGACY_WORKFLOWS.md`.

## Leakage controls preserved

- temporal role is assigned from `Target_Datetime`
- Evaluation 2025 is excluded from model selection and fitting
- final model parameters and 53 rounds are frozen before Evaluation is loaded
- SHAP is computed only after final evaluation and does not refit the model
