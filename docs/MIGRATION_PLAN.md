# Migration Plan

Status: approved repository build plan; no legacy mutation permitted.

## Purpose

Create a clean, GitHub-ready research repository that documents and packages the current Rain Bomb scientific workflow and the downstream Atlas web explorer without changing scientific logic or frozen results.

## Source-of-truth policy

- Scientific analysis chain: `RainBomb_FinalAnalysis_v2/02_PROCESS/01–08` and its frozen outputs.
- Upstream canonical/provenance: selected files under `Workplace_Code` and `Workplace_GIS`; referenced by manifest, not copied as large data.
- Web explorer: latest working static application in `Workplace_Code_Redesign/dist`; copied into `web/app` as source files, with derived display data separated under `web/data`.
- Legacy directories remain read-only and in place.

## Non-negotiable safeguards

1. No delete, move, rename, overwrite, reset, refit, retune, or rerun against legacy workspaces.
2. No change to Def5, Compact20 predictors, temporal split, model parameters, Evaluation 2025, SHAP values, or frozen outputs.
3. No raw/canonical large datasets in Git.
4. Evaluation 2025 artifacts are represented by hashes/manifests and small final artifacts only; evaluation is not rerun.
5. All copied/renamed files are recorded in `docs/FILE_RENAME_MAP.md`.
6. Every uncertain classification is marked `UNCERTAIN`.

## Build sequence

1. Create this migration plan first.
2. Produce a dry-run source → destination map and verify the destination did not exist.
3. Create repository directories and documentation.
4. Copy selected scripts into new names; refactor only path resolution to use `config/paths.yaml` or `config/paths.example.yaml`.
5. Copy only compact frozen outputs needed for verification and demonstration; record all large artifacts by path/hash instead.
6. Package the current static web source and selected web-display derivatives separately from scientific code.
7. Add non-mutating tests for counts, splits, leakage guards, model rounds, and hashes.
8. Generate release/data manifests and environment reports from the existing environments without installing or upgrading packages.
9. Audit the new repository for hard-coded legacy paths, large files, secrets, missing mappings, syntax errors, and accidental source writes.
10. Provide Git commands but do not initialize or push unless separately authorized.

## Planned exclusions

- Raw meteorological archives, ERA5 NetCDF, canonical full-size tables, intermediate Parquet datasets, virtual environments, caches, render-QA directories, notebooks/experiments outside the frozen hourly Compact20 chain, and legacy duplicate model workflows.
- Presentation-only XLSX/inspect NDJSON files and temporary Codex artifacts.

## Rollback

The new repository is additive and isolated. Rollback means removing only `RainBomb_Research_GitHub` after explicit user approval; no legacy artifact requires restoration because none is modified.
