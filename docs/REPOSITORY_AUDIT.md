# Repository Audit

Audit date: 2026-10-13 (Asia/Bangkok)

## Result

The new repository is isolated from all legacy workspaces. No original file was
moved, deleted, overwritten, or used as a write target. No scientific pipeline,
model, Def5 rule, split, feature set, frozen Evaluation 2025 result, or SHAP result
was recomputed.

Final cleanup removed redundant numeric filename prefixes from all 13 scientific
scripts. Stage ordering remains encoded by the `src/01_*` through `src/09_*`
directories; script content was not edited during this rename. Documentation,
tests, source-hash mappings, and bundled checksums were updated to the new paths.

## Automated checks

- Python compilation: PASS for `src/` and `web/scripts/`.
- Repository assertions: PASS, 9 assertions across 4 test files. The active
  migration environment lacked pytest, so the plain test functions were executed
  directly with Python; `pytest` remains the documented normal runner.
- Frozen bundled hashes: PASS for every artifact with a `repo_path` in
  `RELEASE_MANIFEST.yaml`.
- Leakage guards: PASS; final fitting reads Train + Validation only, records that
  Evaluation was not used, and the one-time Evaluation overwrite guard remains.
- Web smoke test: PASS at `http://127.0.0.1:4174/app/#map`; data loading completed,
  and the Map, Events, XAI, and Scenario routes each became the expected visible
  page. The browser recorded no warning/error messages. Existing validated page
  captures remain bundled under `web/screenshots`.
- Large/forbidden-content scan: 0 Parquet, TIFF, NetCDF, GRIB, key/PEM, virtual
  environment, or node_modules files; 0 files above 10 MiB.
- Candidate-secret scan: no matches outside provenance/manifest documentation.
- Absolute-path scan: no machine-specific `D:/...` path remains in runnable copied
  code; external locations resolve through `config/paths.yaml`.

## Provenance controls

- `RELEASE_MANIFEST.yaml` records the frozen chain, original paths, SHA-256 values,
  and known row counts.
- `data/manifests/artifact_catalog.csv` is the tabular catalog.
- `data/manifests/source_script_hashes.csv` pairs all 13 original scientific scripts
  with their migrated copies and documents the configuration-only path refactor.
- `data/checksums/SHA256SUMS.txt` covers bundled release artifacts and web files.

## UNCERTAIN

1. The legacy web application is located in a directory named `dist`, but the
   workspace contains no alternate framework source directory or package/build
   manifest. The legacy audit identifies those plain files as the current working
   application. They are migrated to `web/app` as operational source, while this
   naming ambiguity remains documented.
2. Upstream canonical/Def5 builders remain in legacy workspaces. Their final status
   beyond the frozen artifacts and audit lineage was not re-adjudicated here; they
   are deliberately excluded from the normal repository run.
3. Historical station-source timezone is not independently verified. Stored
   datetimes remain unchanged; the web documentation must not infer a timezone.

## Deliberately excluded

Raw/canonical archives, large Parquet model tables, full Evaluation/SHAP tables,
NetCDF/GRIB, source GDB/layer packages, TIFF/COG rasters, virtual environments,
node_modules, caches, render-QA products, backup presentations, historical model
experiments, secrets, and legacy copies. Exact frozen artifacts remain referenced
by path and checksum rather than duplicated.

## Residual operational notes

- Leaflet and OpenStreetMap assets require network access during normal viewing.
- Rebuilding web derivatives requires the optional packages in
  `web/requirements.txt`, GDAL command-line tools, and external read-only source
  paths. The bundled explorer itself needs only a local static HTTP server.
- Python bytecode caches created during compilation are ignored by `.gitignore`
  and are not repository content.
