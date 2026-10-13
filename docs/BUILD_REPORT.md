# Migration Build Report

## Repository created

`D:/STUDY_GEO/2569_1/154499/RainBomb_Research_GitHub`

The release contains the concise project README, methodology/pipeline/release
documentation, configurable paths, frozen artifact manifests, 13 renamed final-chain
scripts, compact final artifacts, environment records, non-mutating tests, legacy
classification notes, and the RainBomb Atlas downstream explorer with four current
screenshots.

## Top-level tree

```text
RainBomb_Research_GitHub/
├── README.md
├── RELEASE_MANIFEST.yaml
├── .gitignore
├── archive_notes/
├── config/
├── data/
│   ├── checksums/
│   └── manifests/
├── docs/
├── environment/
├── outputs/
│   ├── diagnostics/
│   ├── figures/
│   ├── models/
│   ├── reports/
│   └── tables/
├── src/01_qc … src/09_figures
├── tests/
└── web/
    ├── app/
    ├── assets/
    ├── data/derived/
    ├── public/
    ├── screenshots/
    └── scripts/
```

The complete old-to-new mapping is in `docs/FILE_RENAME_MAP.md`; uncertainties,
exclusions, and validation evidence are in `docs/REPOSITORY_AUDIT.md`.

## Git commands (not executed)

```powershell
cd D:\STUDY_GEO\2569_1\154499\RainBomb_Research_GitHub
git init
git add .
git status
git commit -m "Create reproducible Rain Bomb research release"
git branch -M main
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

The final two commands prepare and perform a push. Do not run them until the remote
URL and permission to publish have been explicitly confirmed.
