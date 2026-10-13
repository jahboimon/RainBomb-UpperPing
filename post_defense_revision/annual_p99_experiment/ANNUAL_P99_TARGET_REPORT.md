# Annual-P99 Target Sensitivity Experiment

## Executive result

งานนี้สร้าง target ใหม่และฝึก XGBoost Compact20 ใน workspace แยก โดยไม่แก้ frozen Def5, final model หรือ Evaluation2025 artifacts เดิม ผลรวมชี้ว่า **Annual-P99 เหมาะเป็น sensitivity analysis แต่ยังไม่ควรแทน main target**: event set เปลี่ยนชัดเจน (Jaccard 0.829) ขณะที่ผลโมเดล 2025 ไม่ดีขึ้นอย่างสม่ำเสมอ และนิยาม annual ใช้ข้อมูลทั้งปีเดียวกับ event จึงเป็น retrospective label ไม่ใช่ threshold ที่รู้ล่วงหน้าใน realtime

## Target construction

- ใช้ canonical hourly rainfall ปี 2012–2025 แยกตาม station-year
- P99 คำนวณจาก `Rainfall > 0` ด้วย linear quantile
- event hour E ต้องผ่านครบ: `Rain(E) >= annual P99`, `Rain(E) >= 20 mm/h`, antecedent 3-hour sum `<= 5 mm`, post-3-hour maximum `< 20 mm/h`
- E−3 ถึง E+3 ต้องเป็นสถานีเดียวกัน ต่อเนื่องตรงชั่วโมง และ rainfall ทุกค่าต้อง finite
- missing ไม่ถูกแทนด้วย zero
- frozen Def5 ถูก recompute ได้ 1,123 events และ key ตรง official event table ทุกแถวก่อนทำ comparison

ได้ annual thresholds 484 station-years จาก 40 สถานี (2012–2025) และ Annual-P99 events 1,006 เหตุการณ์ เทียบกับ frozen 1,123 เหตุการณ์:

| Definition | Events | Overlap with frozen | Added | Removed | Jaccard |
|---|---:|---:|---:|---:|---:|
| Frozen P99 | 1,123 | 1,123 | 0 | 0 | 1.000 |
| Annual P99 | 1,006 | 965 | 41 | 158 | 0.829 |
| Rolling prior-years P99 | 1,070 | 1,029 | 41 | 94 | 0.884 |

Annual/frozen P99 ratio มี P10/median/P90 = 0.563/0.937/1.260 จึงไม่ได้เป็นเพียงการขยับ threshold แบบคงที่ Annual target ตัด frozen events ออกมากกว่าเพิ่มใหม่ และจำนวน event รายปีแกว่งตั้งแต่ 30 ถึง 128

## Sample-size caveat

มี 17 station-years ที่มี wet hours น้อยกว่า 100 (ค่าต่ำสุด 1 ชั่วโมง; median ของทุก station-years = 438 ชั่วโมง) P99 ของกลุ่มนี้ไม่เสถียรอย่างมาก ตัวอย่างเช่น CHKG-2021 มี 21 wet hours และ P99 2.28 mm/h เทียบ frozen 14.556; FOP015-2023 มี 47 wet hours และ P99 0.20 mm/h เทียบ frozen 21.172 แม้เงื่อนไขขั้นต่ำ 20 mm/h ช่วยกัน threshold ต่ำผิดปกติบางส่วน แต่ไม่ได้แก้ปัญหาความไม่แน่นอนของ percentile estimate

รายชื่อและค่าทั้งหมดอยู่ใน `annual_p99_thresholds.csv` โดยมี flag `Low_Sample_LT100`

## Compact20 model experiment

Population และ predictors ตรงกับ frozen Compact20 ทุกแถว ยกเว้นเปลี่ยน label เป็น Annual-P99 ที่ `Target_Datetime=t+1` เท่านั้น:

| Role | Rows | Positives | Prevalence |
|---|---:|---:|---:|
| Train 2020–2023 | 525,474 | 260 | 0.0495% |
| Validation 2024 | 129,671 | 72 | 0.0555% |
| Evaluation 2025 | 121,296 | 85 | 0.0701% |

ใช้ XGBoost settings เดิม (`aucpr`, hist, learning rate 0.05, depth 4, min child weight 5, subsample/column sample 0.8, regularization และ seed เดิม) Train-only `scale_pos_weight=2020.054` Validation 2024 เลือก best iteration แบบ zero-based 103 หรือ 104 boosting rounds จากนั้น refit Train+Validation โดยคำนวณ development weight ใหม่เป็น 1972.328 และค่อย score 2025 เพียงหลัง model ถูก freeze

Validation 2024: AP 0.03569, ROC AUC 0.81620. ค่านี้ใช้เลือกจำนวนรอบเท่านั้น ไม่มี threshold/hyperparameter search เพิ่ม

### Evaluation 2025 side-by-side

| Metric | Frozen Def5 baseline | Annual-P99 |
|---|---:|---:|
| Positives | 95 | 85 |
| Prevalence | 0.0783% | 0.0701% |
| AP | 0.005980 | 0.004771 |
| ROC AUC | 0.864027 | 0.871719 |
| TP / FP / TN / FN | 71 / 24,193 / 97,008 / 24 | 63 / 19,131 / 102,080 / 22 |
| Precision | 0.002926 | 0.003282 |
| Recall | 0.7474 | 0.7412 |
| Specificity | 0.8004 | 0.8422 |
| F1 | 0.005829 | 0.006536 |

Annual target ลด AP 0.001209 และ recall 0.0062 แต่ ROC AUC เพิ่ม 0.00769, precision เพิ่ม 0.000356, specificity เพิ่ม 0.0418 และ false positives ลด 5,062 แถว ผลจึงเป็น trade-off ไม่ใช่การชนะ baseline ชัดเจน โดยเฉพาะ AP ซึ่งเป็น primary rare-event ranking metric ลดลง

ใน May–Oct 2025 มี Annual-P99 events ทั้งหมด 104 เหตุการณ์; 85 เหตุการณ์ (81.7%) อยู่ใน model-evaluable Compact20 universe ที่ threshold 0.5 ตรวจพบ 63 เหตุการณ์ คิดเป็น 74.1% ของ evaluable events หรือ 60.6% ของ Annual-P99 events ทั้งหมด Threshold 0.5 เป็น diagnostic operating point ไม่ใช่ operational threshold ที่ tune แล้ว

## Retrospective versus operational interpretation

**Annual-P99 เป็น retrospective definition:** threshold ของปี y คำนวณจาก wet hours ทั้งปี y รวมถึงข้อมูลที่เกิดหลัง event hour ดังนั้นไม่สามารถอ้างว่าเป็น threshold ที่ทราบล่วงหน้าในการพยากรณ์ realtime ได้ แม้การ split/train/evaluate model จะไม่ใช้ 2025 เพื่อ tune ก็ตาม label ของ 2025 ยังอาศัย rainfall ทั้งปี 2025 โดยนิยาม

เพื่อทดสอบ operational viability ได้สร้าง `Rolling_PriorYears_P99`: threshold ปี y ใช้ wet hours ทุกปีก่อน y เท่านั้น และต้องมีอย่างน้อย 100 wet hours ผลได้ 1,070 events และใกล้ frozen มากกว่า Annual-P99 (Jaccard 0.884 เทียบ 0.829) อย่างไรก็ตามมี 8 station-years ในช่วง model years ที่ยังไม่มี prior wet-hour history เพียงพอ ได้แก่ FOP015/016/017/018 ในปี 2020 และ FOP052/FOP053 ในปี 2022–2023 จึงต้องถือเป็น unavailable ไม่ใช่ non-event และยังไม่ควรนำ rolling variant ไป train model โดยไม่กำหนด cold-start policy ล่วงหน้า

## Governance and QA

- protected inputs 6 รายการมี SHA-256 ก่อนและหลังตรงกันทั้งหมด
- ไม่มีการเขียนไฟล์นอก `post_defense_revision/annual_p99_experiment/`
- official frozen event keys match 100%
- ไม่มี duplicate canonical station-hour keys
- 2025 ไม่ถูกใช้เลือกจำนวนรอบ, threshold หรือ hyperparameters
- artifacts เดิมของ frozen model และ Evaluation2025 ไม่ถูก overwrite

รายละเอียดแฮชอยู่ใน `protected_input_hashes.csv` และ machine-readable checks อยู่ใน `provenance_and_qa.json`

## Main artifacts

- `annual_p99_thresholds.csv`, `annual_p99_events.csv`
- annual counts by year/station/month/hour และ `annual_p99_added_removed_events.csv`
- `target_definition_comparison.csv`
- `representative_hydrographs.csv` และ figures
- `rolling_historical_p99_thresholds.csv`, `rolling_historical_p99_events.csv`
- `annual_p99_model_metrics.csv`, `annual_vs_frozen_model_metrics_2025.csv`
- `annual_p99_event_coverage_2025.csv`
- `annual_p99_model_pr_curve_2025.png`, `annual_p99_model_score_distribution_2025.png`
- isolated model-ready data, predictions และ model JSON
