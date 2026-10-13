# Legacy Workflows

- Def1–Def3 and most Def4 outputs are historical decision evidence, not the current target.
- `RainBomb_Def4_2012_2025_v1` remains a verification dependency because the frozen P99 and prior event keys are used by the Def5 build.
- Earlier `*_v1` feature/master tables may be reproduction dependencies for current `v2/v3` builders; they are not safe deletion candidates.
- The older `Workplace_Code` LR/RF/XGB → ablation → pre-test lock → Test 2025 → SHAP chain is an official historical workflow, but it is distinct from the current FinalAnalysis v2 Compact20 release.
- Parallel daily, next-day, station-week, radar, controlled-improvement, and physical-feature experiments are `UNCERTAIN / experimental` unless a later decision record promotes them.
- No legacy file was moved or deleted while creating this repository.
