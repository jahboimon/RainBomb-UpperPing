# Rain Bomb post-defense scientific revision

This workspace is separate from the frozen workflow. It reads the canonical/legacy repository at the path recorded in `config.json` and writes only beneath this directory.

## Run order

1. `python run_revision.py --canonical-root <path-to-Workplace_Code>`
2. Review `R1_zero_missing/` before accepting later sections.
3. Review R2–R4 tables and figures.
4. Read `R5_atmospheric_predictor_review.md`.
5. Read `POST_DEFENSE_REVISION_REPORT.md` and `REVISION_DECISION_LOG.md`.
6. Run `python verify_revision.py` for a reproducibility/immutability check.

The run is descriptive/sensitivity work only. It does not train or tune a model, does not use 2025 for threshold selection, and never writes to the canonical repository.
