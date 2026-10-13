# RainBomb Atlas Web Explorer

This is a **downstream visualization and inspection tool only**. It must never be used as scientific upstream input or as authority over frozen model/Def5 artifacts.

## Current source and authority note

The latest working version was copied from `Workplace_Code_Redesign/dist` on 13 October 2026. The source entry point is `web/app/index.html`; it is a plain static HTML/CSS/JavaScript application using Leaflet 1.9.4 from a public CDN. There is no backend, database, package manifest, framework source tree, or bundling step.

`UNCERTAIN`: the legacy directory is named `dist`, which normally implies a generated build artifact, but the audit found no alternative HTML/CSS/JavaScript source authority. The files in that directory are the hand-maintained operational application identified by the legacy web audit. They were therefore migrated to the neutral `web/app` name without treating legacy caches or render products as sources.

## Data

The app reads selected browser derivatives under `web/data/derived`: station/event GeoJSON, model/SHAP summaries, timeline JSON, frozen browser model, validation rows, and display PNG rasters. COG/TIFF, raw sources, full model-ready data, and full SHAP Parquet are excluded and remain referenced by scientific manifests.

The read-only derivative builders are preserved in `web/scripts`. They load external paths from `config/paths.yaml`, verify frozen hashes where defined, and write only to `web/data/derived`. Their observed Python dependencies are recorded in `web/requirements.txt`; the first builder additionally requires the external GDAL command-line tools. Rebuilding web derivatives is optional and does not change scientific outputs.

## Run locally

From repository root:

```powershell
python -m http.server 4173 --directory web
```

Open `http://127.0.0.1:4173/app/`. OpenStreetMap basemap tiles and Leaflet CDN assets require internet access.

`web/app` is the repository's operational source; `web/dist`, caches, and render-QA outputs are ignored. Four validated page captures are retained in `web/screenshots`.
