# Rain Bomb / ฝนตกหนักเฉพาะแห่งในลุ่มน้ำปิงตอนบน

Research code, frozen-release provenance, and the RainBomb Atlas web explorer for a station-hour study of isolated or abrupt heavy rainfall in the Upper Ping River Basin.

## Research task

Predict whether a frozen **Def5** event occurs at station-hour `t+1` using information available at `t`. Def5 identifies a heavy rainfall pulse that exceeds the frozen station P99 and 20 mm/h floor, has low antecedent 3-hour rainfall, and does not remain heavy during the following three hours. The exact definition is frozen; see [methodology notes](docs/METHODOLOGY_NOTES.md).

- Season: May–October 2020–2025
- Train: 2020–2023
- Validation: 2024
- Evaluation: 2025
- Predictors: frozen Compact20
- Compared models: Logistic Regression, Random Forest, XGBoost
- Final model: XGBoost, 53 boosting rounds
- Interpretation: native TreeSHAP/SHAP; attribution, not causality

## Pipeline

```text
Frozen canonical inputs → Def5/t+1 target → seasonal universe → Compact20
→ LR/RF/XGB validation comparison → frozen final XGBoost
→ one-time Evaluation 2025 → SHAP → spatial summaries/figures → Atlas web explorer
```

The large source datasets are intentionally **not stored in this repository**. Paths and SHA-256 values are recorded in [RELEASE_MANIFEST.yaml](RELEASE_MANIFEST.yaml) and [data/README.md](data/README.md).

## Quick start

1. Copy `config/paths.example.yaml` to `config/paths.yaml` and set read-only upstream paths plus a separate runtime output directory.
2. Create an environment using `environment/requirements.txt`; do not upgrade the frozen legacy environment in place.
3. Run non-destructive checks:

```powershell
python -m pytest tests
```

4. Consult [PIPELINE.md](docs/PIPELINE.md) before running scientific scripts. Evaluation 2025 is frozen and must not be rerun merely to reproduce the repository presentation.

Run the web explorer:

```powershell
python -m http.server 4173 --directory web
```

Open `http://127.0.0.1:4173/app/`.

## Reproducibility scope

This repository can verify the final code structure, provenance, frozen artifact
identities, release counts, temporal splits, and leakage safeguards. Downstream
analyses can be rerun in a separate runtime after the external read-only paths in
`config/paths.yaml` are configured. The bundled RainBomb Atlas can be served
directly with the local command above.

To keep the repository suitable for GitHub, it does not contain raw station
archives, large canonical datasets, the full Evaluation or SHAP Parquet tables,
or NetCDF, TIFF/COG, GRIB, and source geodatabase files. Their frozen identities
and original locations are recorded in `RELEASE_MANIFEST.yaml` and the data
manifests; absence from Git is intentional and does not authorize rebuilding them.

## Repository roles

- `src/`: copied final-analysis scripts with configurable paths; scientific calculations preserved
- `outputs/`: compact frozen artifacts suitable for Git
- `data/`: manifests only, not raw/canonical data
- `web/`: downstream visualization and inspection tool; never a scientific upstream
- `archive_notes/`: legacy classifications and unresolved items

See [CURRENT_RELEASE.md](docs/CURRENT_RELEASE.md) for the authoritative release boundary.
