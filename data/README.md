# External Data and Provenance

Large datasets are not committed. Configure their read-only locations in `config/paths.yaml`.

| Dataset | Role | Stored in Git? |
|---|---|---|
| canonical hourly rainfall 2012–2025 | dynamic station-hour denominator and rain features | no; manifest/hash only |
| canonical RH and temperature | predictors | no |
| frozen Def5 event table | official event metadata and count | no; manifest/hash only |
| frozen t+1 target | evaluable station-hour target | no |
| ERA5 station-hour canonical | six final atmospheric predictors | no |
| seasonal modeling universe | final common population | no |
| Compact20 model table | final feature table | no |
| Evaluation 2025 predictions | frozen one-time evaluation | no; metrics are bundled |
| Evaluation SHAP | frozen attribution table | no; compact summaries are bundled |
| web display derivatives | downstream visualization only | selected JSON/GeoJSON/PNG bundled under `web/data` |

Artifact paths, hashes, row counts, and statuses are in `RELEASE_MANIFEST.yaml` and `data/manifests/artifact_catalog.csv`.
Checksums for the compact files actually bundled in this repository are listed in
`data/checksums/SHA256SUMS.txt`.
