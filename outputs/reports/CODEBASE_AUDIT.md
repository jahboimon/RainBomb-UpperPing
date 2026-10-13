# Rain Bomb Codebase Audit

วันที่สำรวจ: 13 ตุลาคม 2026  
สถานะ: **read-only audit** — ไม่ได้ย้าย แก้ หรือลบ source, script, scientific output หรือ methodology ใด ๆ; ไฟล์นี้เป็น output ใหม่เพียงไฟล์เดียว

## 1. ขอบเขตและข้อสรุปหลัก

สำรวจโครงสร้างระดับโครงการใต้ `D:/STUDY_GEO/2569_1/154499` และลงรายละเอียดสายงานที่มีหลักฐานว่าใช้งานจริงใน:

- `Workplace_Code` — legacy/upstream workspace, frozen canonicals, Def5, feature builders, ERA5 และสาย model เดิม
- `RainBomb_FinalAnalysis_v2` — clean final-analysis workspace ที่อ่าน frozen inputs จาก `Workplace_Code`
- `Workplace_Code_Redesign` — web/report derivative workspace; ไม่ใช่แหล่ง scientific truth
- `Workplace_GIS` — static/spatial inputs ภายนอก code workspace

ขนาดโดยประมาณที่พบ:

- `RainBomb_FinalAnalysis_v2`: 1,362 ไฟล์, 1.23 GB, script-like files 204 ไฟล์ (รวม auxiliary/experimental branches)
- `Workplace_Code`: 72,508 ไฟล์, 15.16 GB, script-like files 13,646 ไฟล์; ตัวเลขสูงมากเพราะรวม environments, archives, generated artifacts และ historical branches

**ข้อสรุป:** สำหรับผลวิจัยปัจจุบัน ให้ถือ `RainBomb_FinalAnalysis_v2/02_PROCESS/01–08` และ `03_OUTPUT/01–08` เป็นสายวิเคราะห์หลัก ส่วน `Workplace_Code` เป็น upstream frozen-data lineage ที่ยังจำเป็นและห้ามลบ/ย้าย. `Workplace_Code/PROJECT_VIEW_v2` เป็น index/shortcut view ไม่ใช่สำเนาข้อมูล canonical.

## 2. แหล่ง canonical/final ที่มีหลักฐานชัด

| Stage | Canonical/final artifact | สถานะจากหลักฐาน |
|---|---|---|
| Station metadata | `Workplace_Code/Hourly_rain/RESETWORKPLACE/02_processed/Station_Metadata_Cleaned.csv` | CURRENT / PROTECTED / DO NOT MOVE |
| Rainfall canonical | `Workplace_Code/Rainfall_Canonical_2012_2025_Final_v1/PROCESS_Hourly_Rain_Canonical_2012_2025_v1.parquet` | CURRENT / CANONICAL; 2012–2025 |
| Rainfall QC | `.../PROCESS_Hourly_Rain_Canonical_2012_2025_StationYear_QC_v1.csv` | frozen QC companion |
| RH canonical | `Workplace_Code/Humidity_Canonical_Final_v4/PROCESS_Hourly_Humidity_Canonical_v4.csv` | CURRENT / CANONICAL; recommended by final report |
| Temperature canonical | `Workplace_Code/Temperature_Canonical_Final_v1/PROCESS_Hourly_Temperature_Canonical_v1.csv` | CURRENT / CANONICAL |
| Pressure canonical | `Workplace_Code/Pressure_Canonical_Final_v1/PROCESS_Hourly_Pressure_Canonical_v1.csv` | CURRENT / CANONICAL; excluded from final Compact20 primary model |
| Static predictors | `Workplace_GIS/LST/Station_Static_Predictors_v1.csv` | CURRENT external input; upstream GIS provenance incomplete/fragmented |
| Def5 definition | `Workplace_Code/Def5_Final_Target_2012_2025_v1/01_Def5_Final_Freeze/Def5_FINAL_DEFINITION_v1.md` + JSON/provenance | FINAL / FROZEN |
| Def5 events | `.../02_Def5_Extended_Target/RainBomb_Target_Def5_2012_2025_v1.parquet` | FINAL / FROZEN TARGET; 1,123 station-hours |
| Def5 t+1 base | `.../04_Def5_Plus1h_Target/ML_Base_StationHour_Target_Def5_t1_2012_2025_v1.parquet` | frozen evaluable target base |
| Dynamic feature tables | rainfall `v2`, humidity `v3`, temperature `v2`, pressure `v2` folders under `Workplace_Code` | CURRENT feature builders/outputs |
| Non-ERA5 master bridge | `Workplace_Code/ML_StationHour_Master_Candidate_2012_2025_v2/ML_StationHour_Master_Candidate_2012_2025_v2.parquet` | CURRENT PREDICTOR BRIDGE; **not current target authority** |
| ERA5 canonical | `Workplace_Code/PhaseB_Atmospheric/ERA5/05_StationHour_Canonical/ERA5_StationHour_2013_2025_v1.parquet` | CURRENT / CANONICAL |
| Frozen Def5+ERA5 source | `.../06_Def5_ModelReady/PhaseB_Def5_ERA5_ModelReady_v1.parquet` | upstream source used by FinalAnalysis Phase 3 |
| Final seasonal universe | `RainBomb_FinalAnalysis_v2/03_OUTPUT/03_Seasonal_Modeling_Universe/Seasonal_Modeling_Universe_v1.parquet` | canonical population for final analysis |
| Final Compact20 table | `RainBomb_FinalAnalysis_v2/03_OUTPUT/05_Model_Comparison/ModelReady_Seasonal_Compact20_v1.parquet` | final model-ready table; SHA256 frozen in project docs |
| Development models | `LR_Model_v1.joblib`, `RF_Model_v1.joblib`, `XGB_Model_v1.json` in `03_OUTPUT/05_Model_Comparison` | model-comparison artifacts only |
| Final model | `03_OUTPUT/06_Final_Model_Evaluation/XGB_Final_Compact20_v1.json` | **final frozen XGBoost**, 53 rounds; distinct from Phase 5 XGB model |
| Final evaluation | `.../Evaluation2025_Actual_vs_Predicted_v1.parquet` + metrics/report | untouched Evaluation 2025 output; 121,296 rows, 95 positives |
| Final SHAP | `03_OUTPUT/07_SHAP_XAI/Evaluation2025_SHAP_v1.parquet` | native TreeSHAP for final frozen model |
| Final SHAP summaries | `Global_SHAP_Importance_v1.csv`, `Station_SHAP_Long_v1.csv`, `Station_SHAP_Summary_v1.csv`, `Local_SHAP_Examples_v1.csv` | derived explanatory outputs |
| Spatial summaries/figures | `03_OUTPUT/08_Spatial_Analysis/*_v1` | final-analysis derivatives of Evaluation/SHAP; not new target/model |

## 3. Workflow จริงและลำดับ dependency

```text
raw station archives / protected cleaned baseline
    → rainfall, RH, temperature canonicals (+ pressure sensitivity canonical)
    → lag/window feature tables
    → non-ERA5 station-hour master bridge
    → frozen Def5 definition + Def5 event target + t+1 target/splits
    → ERA5 station-hour canonical
    → Def5 + ERA5 model-ready source
    → FinalAnalysis seasonal scope/universe (May–October 2020–2025)
    → Compact20 model-ready table
    → LR/RF/XGB development comparison on Train/Validation
    → freeze/refit final XGB on 2020–2024
    → one-time Evaluation 2025
    → native TreeSHAP + summaries/dependence diagnostics
    → station spatial summaries/maps
    → diagnostics and report/thesis/web derivatives
```

### Recommended run order for the current final-analysis chain

1. **Do not rebuild upstream by default.** Verify hashes/provenance of frozen `Workplace_Code` inputs first.
2. `02_PROCESS/01_DataPeriod_QC/01_data_period_qc_v1.py`
3. `02_PROCESS/02_Seasonal_Scope/01_define_seasonal_scope_v1.py`
4. `02_PROCESS/03_Seasonal_Modeling_Universe/01_build_seasonal_modeling_universe_v1.py`
5. `02_PROCESS/04_Feature_Architecture/01_review_feature_architecture_v1.py`
6. `02_PROCESS/05_Model_Comparison/01_build_modelready_v2.py`
7. `02_PROCESS/05_Model_Comparison/02_train_compare_models_v1.py`
8. `02_PROCESS/06_Final_Model_Evaluation/01_freeze_refit_final_xgb_v1.py`
9. `02_PROCESS/06_Final_Model_Evaluation/02_evaluate_final_xgb_2025_v1.py` — script ตั้งใจ refuse หาก Evaluation outputs มีอยู่แล้ว; **ไม่ควรรันซ้ำบนชุด final เดิม**
10. `02_PROCESS/07_SHAP_XAI/01_compute_final_xgb_shap_v1.py`
11. `02_PROCESS/07_SHAP_XAI/02_summarize_shap_v1.py`
12. `02_PROCESS/07_SHAP_XAI/03_shap_dependence_diagnostics_v1.py`
13. `02_PROCESS/08_Spatial_Analysis/01_build_station_spatial_summary_v1.py`
14. `02_PROCESS/08_Spatial_Analysis/02_create_final_spatial_maps_v1.py`
15. Phase 9 diagnostics/figures เฉพาะเมื่อจำเป็น; เป็น downstream audits ไม่ใช่ input สำหรับเลือกหรือ refit final model

## 4. Input/output ของ script หลัก

| Script | Main inputs | Main outputs |
|---|---|---|
| Legacy `01_finalize_rainfall_canonical_2012_2025_v1.py` | protected cleaned rainfall, 2024H2–2025 extension candidate, station metadata, duplicate audit | canonical rainfall CSV/Parquet, station-year QC, audit, provenance |
| Legacy Def5 `01_build_def5_final_target_2012_2025_v1.py` | rainfall canonical, frozen P99/Def4 verification sources, station metadata, non-ERA5 master bridge, definition audits/decision log | frozen Def5 definition, event table, Def4-vs-Def5 comparisons, t+1 base, temporal splits, QC/provenance |
| Legacy rain/RH/temp/pressure feature builders | respective canonical + earlier baseline feature table used for reproduction checks | versioned t0–t6 feature Parquet + audit/provenance |
| Legacy `01_build_ml_stationhour_master_candidate_2012_2025_v2.py` | old/base target bridge, dynamic feature tables, static GIS table, old master for reproduction check | non-ERA5 master v2 + coverage tables/audit/provenance |
| ERA5 `ERA5_Build_StationHour_2013_2025_v1.py` | downloaded ERA5 NetCDF archive + station locations | ERA5 station-hour canonical + QC/report/provenance |
| ERA5 `01_build_phaseb_def5_era5_modelready_v1.py` | ERA5 canonical + frozen Def5 Train/Validation/Test tables + station-year QC | `PhaseB_Def5_ERA5_ModelReady_v1.parquet` + QC/report/provenance |
| Phase 1 QC | rainfall/RH/temp canonicals, station metadata, Def5 events/QC/t+1 base | annual/station-year/period QC CSVs, figure, report, frozen-input note |
| Phase 2 scope | Def5 event table + t+1 evaluable base | monthly/candidate-window/season-year tables, figure, report |
| Phase 3 universe | PhaseB Def5+ERA5 table, frozen t+1/events, station metadata, Phase 2 counts | seasonal universe Parquet + summary/monthly/station-year/missingness CSVs + report |
| Phase 4 architecture | Phase 3 universe/summary, feature datatype audit, LULC XML, static station table | architecture/redundancy/LULC diagnostic tables + report; no model fit |
| Phase 5 build | Phase 3 universe | Compact20 model-ready Parquet; maps LULC class 9 → `BuiltUp_Binary=1` |
| Phase 5 compare | Compact20 table restricted to Train 2020–2023 + Validation 2024 | LR/RF joblib, development XGB JSON, validation predictions/metrics, comparison plots/report |
| Phase 6 freeze/refit | Compact20 Train+Validation + Phase 5 comparison decision | freeze manifest, final XGB JSON, training summary |
| Phase 6 evaluate | final model/manifest/summary + Compact20 Evaluation rows | 2025 predictions, metrics, curves/confusion matrix, report |
| Phase 7 SHAP compute | final model, freeze manifest, training summary, model-ready Evaluation, final scores | row-level `Evaluation2025_SHAP_v1.parquet` |
| Phase 7 summarize | SHAP table + final model | global/positive/outcome/station/local CSVs, beeswarms/bar charts/local figures/report |
| Phase 7 dependence | frozen SHAP table + global importance + final model hashes | feature dependence figures/tables/report |
| Phase 8 station summary | final evaluation, SHAP, Compact20 Evaluation, station metadata | station evaluation, built/nonbuilt, case-study CSVs |
| Phase 8 maps | Phase 8 summaries, station SHAP summaries, metadata, boundary, long-term Def5 context | final maps, heatmap, case figure, report |

## 5. ไฟล์ซ้ำ เก่า หรือ obsolete

### ยืนยันจากเอกสารว่า historical/obsolete สำหรับ current target

- Def1–Def3 outputs และ Def3 exploratory/audit folders: historical evidence; ไม่ใช่ current target.
- `RainBomb_Def4_2012_2025_v1`: **historical definition + current verification dependency**; ยังห้ามลบ เพราะ Def5 builder ใช้ frozen P99/Def4 keys ตรวจ reproduction.
- `ML_Base_Target_t1_v1`, `ML_StationHour_Master_Candidate_v1`, `Rainfall_Features_t0_t6_v1`, `Humidity_Features_t0_t6_v2`, `Temperature_Features_t0_t6_v1`, `Pressure_Features_t0_t6_v1`: รุ่นก่อนหน้าที่ถูก v2/v3 supersede แต่บางตัวถูก current builders อ่านเพื่อ baseline reproduction. จึงควรติดป้าย `archive_dependency` ไม่ใช่ลบทิ้ง.
- Legacy model chain `ML_Logistic_Baseline_v1`, `ML_RandomForest_Baseline_v1`, `ML_XGBoost_Baseline_v1`, ablation/pre-test/final-test/SHAP folders เป็น official ของ workflow รุ่นก่อน แต่ **ไม่ควรปะปน** กับ final-analysis-v2 models/results.

### Generated/helper files ที่ไม่ควรเป็น scientific authority

- `03_OUTPUT/**/Check_*.py`, `check_parquet.py`: ad-hoc verification helpers อยู่ใน output tree; ไม่ใช่ canonical process scripts.
- `*.inspect.ndjson`, generated XLSX presentation tables, render-QA folders, slide assets, web `dist/`: report/presentation derivatives; ห้ามย้อนใช้เป็น model input.
- `PROJECT_VIEW` และ `PROJECT_VIEW_v2`: navigation/index/shortcut layers; อย่าใช้ shortcut existence เป็นหลักฐานว่า artifact canonical โดยลำพัง.
- `.venv*`, `_runtime_deps`, `__pycache__`, `.codex_tmp*`, render QA: environment/cache; ไม่ใช่ research outputs.

### `UNCERTAIN`

- `BackUP_499`, `Workplace_Code.zip`, `TMD_DATA.zip` และ archives อื่นอาจเป็น exact copies หรือ historical snapshots แต่ audit นี้ไม่ได้ hash ทุกไฟล์ขนาดใหญ่ จึง **ห้ามถือว่าซ้ำและห้ามลบ**.
- `RainBomb_FinalAnalysis_v2` มี parallel branches เช่น daily/next-day/station-week/radar/controlled-improvement. ไม่พบหลักฐานว่าถูก promote เข้าสู่ final hourly Compact20 chain; จัดเป็น `UNCERTAIN / parallel experiment` จนมี decision record.
- `03_OUTPUT/08_Spatial_Analysis/S5_Geographic_Interpretation` มี scripts อยู่ใต้ output tree และ provenance ของตนเอง; เป็น downstream extension ไม่ใช่ส่วนบังคับของ core 01–08 run order.
- เอกสาร `00_PROJECT_DESIGN/README.md` ระบุสถานะ “ready for Phase 7” และว่า SHAP ยังไม่ทำ แต่ filesystem มี Phase 7/8 outputs แล้ว; README จึงล้าสมัยบางส่วน.
- `Workplace_Code/AGENTS.md` ยังประกาศ Def3 frozen และห้าม hard-code drive letters ขณะที่ current official register/Def5 workspaceและ FinalAnalysis scriptsใช้ Def5 และ absolute `D:` paths. ต้องรักษาเป็น historical governance evidence แต่ไม่ควรใช้ตัดสิน current methodology โดยไม่มีการแก้ governance document อย่างเป็นทางการ.
- Legacy official split (Train 2013–2024H1, Validation 2024H2, Test 2025) ต่างจาก final-analysis-v2 split (Train 2020–2023, Validation 2024, Evaluation 2025). ทั้งสองถูกต้องใน workflow คนละรุ่น; ชื่อ `Train/Validation/Test` อย่างเดียวไม่พอ ต้องผูกกับ workspace/version.

## 6. จุดสับสนและความเสี่ยงใช้ไฟล์ผิด

1. **สอง workflow เป็น official คนละเวลา:** `Workplace_Code` official register ยังชี้ legacy model chain แต่ final results ปัจจุบันอยู่ `RainBomb_FinalAnalysis_v2`. ควรประกาศ “current analysis release” เพียงจุดเดียว.
2. **ชื่อ model ซ้ำความหมาย:** `XGB_Model_v1.json` คือ development model; `XGB_Final_Compact20_v1.json` เท่านั้นคือ final refit model.
3. **target bridge circular-looking lineage:** Def5 builder อ่าน non-ERA5 master v2 ซึ่งสืบจาก target bridge รุ่นเก่า แล้วลบ/แทน target เป็น Def5. ถูกอธิบายใน docs แต่โครงสร้างเสี่ยงให้ผู้ใช้หยิบ `Target_t1` เก่ามาใช้ผิด.
4. **split semantics เปลี่ยนระหว่าง workspace:** legacy calendar split กับ final May–October 2020–2025 splitมีชื่อ role คล้ายกัน.
5. **absolute paths:** FinalAnalysis หลาย script hard-code `D:/...`; ย้ายเครื่อง/drive แล้วพัง แม้ผลปัจจุบันยัง reproducible บนเครื่องนี้.
6. **environment ไม่ freeze รวมศูนย์:** มี `.venv`, `.venv_era5`, `.venv_xgb` และ bundled runtime deps แต่ไม่มี environment file เดียวที่ยืนยัน entire final chain.
7. **scripts ใน output folders:** ทำให้แยก source code กับ generated results ยาก และอาจถูกรวมใน archive ผลลัพธ์โดยไม่ตั้งใจ.
8. **report/web derivatives อ่าน frozen files โดยตรง:** ถูกต้องสำหรับการนำเสนอ แต่ต้องห้าม derivative กลับมาเป็น scientific upstream.
9. **static GIS provenance:** `Station_Static_Predictors_v1.csv` เป็น current external input แต่ official registerยอมรับว่า upstream GIS provenance ยังควรชัดขึ้น.
10. **timezone:** ERA5 UTC→ICT ถูกบันทึก แต่ station-source timezone ยัง unverified; ห้ามแปลงเพิ่มโดยสมมติเอง.
11. **missingness:** canonicals/features ตั้งใจรักษา missing; ห้ามเติม 0 หรือ impute โดยไม่มี approval.
12. **rerun final evaluation:** Phase 6 evaluation scriptออกแบบให้ score ครั้งเดียวและ refuse overwrite. การลบ outputs เพื่อรันใหม่จะทำลาย audit intent.

## 7. Dependencies

Core final-analysis Python dependenciesที่เห็นจาก imports:

- `pandas`, `numpy`, `pyarrow`
- `matplotlib`
- `scikit-learn`, `joblib`
- `xgboost` (fallback ไป `Workplace_Code/.venv_xgb/Lib/site-packages`)
- `geopandas` สำหรับ spatial outputs

Upstream/optional dependencies:

- `polars` ใน Def5/master builds
- `xarray`, NetCDF stack, `cdsapi` สำหรับ ERA5 retrieval/extraction
- `requests`, `psutil`, Excel writer/reader dependenciesใน QC/report steps
- Earth Engine `ee` ปรากฏใน feasibility/smoke-test branch แต่ไม่ใช่ dependency ของ final Compact20 chain

`UNCERTAIN`: ไม่พบ lockfile/environment specification เดียวที่ครอบคลุมเวอร์ชันทั้งหมดของ core chain. ก่อนทำ reproducibility release ควร export package versionsจาก environment ที่ใช้สร้าง frozen outputs โดยไม่ reinstall ทับ environment เดิม.

## 8. Folder structure ที่เสนอสำหรับ final research codebase

ข้อเสนอนี้เป็นแบบจำลองเท่านั้น — **ยังไม่ refactor และยังไม่ย้ายไฟล์**.

```text
rainbomb-research/
├─ README.md
├─ CITATION.cff
├─ environment/
│  ├─ requirements-lock.txt
│  └─ environment-report.txt
├─ config/
│  ├─ paths.example.yaml
│  ├─ feature_sets.yaml
│  └─ temporal_splits.yaml
├─ data/
│  ├─ raw_external/              # immutable; manifest/links only
│  ├─ protected_canonical/       # rain/RH/temp/ERA5/static, checksums
│  ├─ frozen_targets/            # Def5 definition/events/t+1
│  ├─ intermediate/              # feature/master/universe
│  └─ manifests/                 # SHA256 + provenance
├─ src/
│  ├─ 01_qc/
│  ├─ 02_target/
│  ├─ 03_features/
│  ├─ 04_model_ready/
│  ├─ 05_model_comparison/
│  ├─ 06_final_evaluation/
│  ├─ 07_xai/
│  ├─ 08_spatial/
│  └─ 09_figures/
├─ outputs/
│  ├─ release_v1/
│  │  ├─ tables/
│  │  ├─ models/
│  │  ├─ figures/
│  │  ├─ reports/
│  │  └─ RELEASE_MANIFEST.json
│  └─ diagnostics/
├─ archive/
│  ├─ legacy_def1_def4/
│  ├─ superseded_features/
│  └─ prior_model_workflows/
├─ web/                          # derivative-only; never upstream
└─ tests/
   ├─ test_keys_and_counts.py
   ├─ test_no_leakage.py
   ├─ test_frozen_hashes.py
   └─ test_model_reproduction.py
```

## 9. ข้อเสนอขั้นถัดไปโดยยังไม่ refactor

1. สร้าง `CURRENT_RELEASE.md` หนึ่งหน้า ระบุว่า final scientific release คือ workspace/version ใด พร้อม SHA256 ของ target, Compact20 table, final model, evaluation และ SHAP.
2. ทำ machine-readable `pipeline_manifest.yaml` ที่ระบุ `script → inputs → outputs → hashes → environment`.
3. แยกสถานะ artifact เป็น `canonical`, `frozen`, `bridge`, `derived`, `diagnostic`, `historical`, `uncertain` โดยไม่ย้ายไฟล์ก่อน.
4. ตรวจ exact-duplicate ด้วย SHA256 เป็นงานแยก ก่อนพิจารณา archive/delete ใด ๆ.
5. ทำ dry-run plan สำหรับ relative/configurable paths; ห้ามแก้ frozen scripts จนกว่าจะมี copy/version ใหม่และ regression tests.

## 10. Evidence used

- `RainBomb_FinalAnalysis_v2/00_PROJECT_DESIGN/README.md`
- `RainBomb_FinalAnalysis_v2/01_INPUT_REFERENCES/FROZEN_INPUTS.md`
- phase scriptsใน `RainBomb_FinalAnalysis_v2/02_PROCESS/01–09`
- output inventoriesใน `RainBomb_FinalAnalysis_v2/03_OUTPUT`
- `Workplace_Code/00_PROJECT_CONTROL/CURRENT_CORE_FILES.md`
- `Workplace_Code/PROJECT_VIEW_v2/00_Project_Docs/Official_Artifact_Register_v1.csv`
- `Workplace_Code/PROJECT_VIEW_v2/00_Project_Docs/RainBomb_Pipeline_Map_v1.md`
- canonical/feature/Def5/ERA5 builder scriptsและ provenance/report filesที่อ้างถึงข้างต้น

การจัดชั้นทั้งหมดอาศัยเอกสารสถานะ, code path, explicit hashes/assertions และ actual output inventory. รายการที่หลักฐานไม่พอถูกทำเครื่องหมาย `UNCERTAIN` และไม่มีการเดาให้เป็น final.
