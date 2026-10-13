# Target construction boundary

Def5 target construction is an upstream, frozen dependency. This release does not
copy or rerun that workflow because the approved scientific target must not be
redefined. Its canonical artifact, checksum, provenance, and row count are listed
in `data/manifests/artifact_catalog.csv` and `RELEASE_MANIFEST.yaml`. The frozen
definition and its role in the t → t+1 task are documented in
`docs/METHODOLOGY_NOTES.md` and the run boundary is documented in
`docs/CURRENT_RELEASE.md`.

The runtime pipeline consumes that frozen target read-only through
`config/paths.yaml` (created locally from `config/paths.example.yaml`).
