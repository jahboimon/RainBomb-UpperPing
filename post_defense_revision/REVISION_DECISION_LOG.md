# Revision decision log

| ID | Evidence | Decision | Class | Status |
|---|---|---|---|---|
| D01 | Canonical sentinel scan: `-999`, `999999`, `9999`, `-` all 0; blank/NaN 679,973; numeric zero 3,499,499 | Retain zero as observed no-rain and missing as NaN. Add explicit text and R1 table to thesis methods/QC. | REQUIRED | ACCEPT |
| D02 | 3,554,919 finite, continuous 7-hour windows; S0 recomputation exactly matches all 1,123 official event keys | Def5 implementation is reproduced. Do not alter frozen target. | REQUIRED | ACCEPT |
| D03 | Several station-years are 100% missing but structurally present as hourly rows | Report dynamic availability and denominator as evaluable hours, not nominal hours. | REQUIRED | ACCEPT |
| D04 | Long observed-zero runs reach thousands of hours at some stations | Treat as a QC warning, not proof of missingness. Retain zeros under source semantics, disclose possible sensor/seasonal ambiguity, and prioritize independent raw-flag crosscheck if available. | REQUIRED | ACCEPT_WITH_CAVEAT |
| D05 | Annual station wet-hour P99: median station CV 0.328; median absolute annual–frozen difference 3.473 mm/h; 10th–90th ratio 0.539–1.271 | Keep frozen 2012–2024H1 P99 for comparability; add annual analysis as robustness evidence. | SENSITIVITY ONLY | ACCEPT |
| D06 | Antecedent relaxation adds 166/306 events; removing post restriction adds 62; station P99.5/P99.9 retains 717/170 events | S0 remains baseline. Do not promote a scenario without independent validation and a new freeze decision. | SENSITIVITY ONLY | ACCEPT |
| D07 | LOSO RMSE: IDW 7.733, ordinary kriging 7.603 events/100k evaluable h | Show kriging as primary estimated surface and IDW as baseline; difference is small, so avoid claims of decisive superiority. | REQUIRED FIGURE CAVEAT | ACCEPT |
| D08 | 3,717 grid cells lie outside station convex hull and are not interpolated; inside-hull P90 nearest-station distance is 24.88 km | Mask extrapolation and state coverage limits, especially sparse/high terrain. | REQUIRED | ACCEPT |
| D09 | 850-hPa moisture/wind is physically useful but coarse and close to high terrain; project elevation reaches 1,180 m | Retain as environmental context with `valid_850`; strengthen limitations paragraph. | DISCUSSION ONLY | ACCEPT |
| D10 | TCWV, CAPE, omega, and cloud variables are available in ERA5 but absent from current canonical | Rank candidates; do not add to frozen model. Any extraction/model test is future work with a new version and development-only selection. | FUTURE WORK | DEFER |
| D11 | No result here retrains a model or uses 2025 to select thresholds | Preserve 2025 holdout and frozen Evaluation 2025 artifacts. | REQUIRED GOVERNANCE | ACCEPT |

No scenario or predictor is promoted by this package. Any future promotion requires a separately signed decision entry, versioned data extraction, development/validation evidence, and a holdout-protection check.
