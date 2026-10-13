"""Review the frozen Phase 3 feature architecture without fitting a model.

Numerical redundancy diagnostics use Train rows only. LULC semantics are read
from the actual project legend metadata. The compact architecture and binary
LULC treatment are proposals for human approval, not implemented transforms.
"""

from __future__ import annotations
import sys as _sys
from pathlib import Path as _ConfigPath
_sys.path.insert(0, str(_ConfigPath(__file__).resolve().parents[1]))
from rainbomb_paths import load_paths as _load_paths
PATHS = _load_paths()

import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = PATHS.runtime_root
LEGACY = PATHS.upstream_root
GIS = PATHS.gis_root
OUT = ROOT / "03_OUTPUT" / "04_Feature_Architecture"

UNIVERSE = ROOT / "03_OUTPUT" / "03_Seasonal_Modeling_Universe" / "Seasonal_Modeling_Universe_v1.parquet"
PHASE3_SUMMARY = ROOT / "03_OUTPUT" / "03_Seasonal_Modeling_Universe" / "Seasonal_Modeling_Universe_Summary_v1.csv"
TYPE_AUDIT = LEGACY / "ML_Def5_Feature_DataType_Table_v1.csv"
LULC_XML = GIS / "LULC10.shp.xml"
LULC_STATION_SOURCE = GIS / "LST" / "Station_Static_Predictors_v1.csv"

RAIN = [
    "Rain_t", "Rain_t_1", "Rain_t_2", "Rain_t_3", "Rain_t_4", "Rain_t_5", "Rain_t_6",
    "RainSum_3h", "RainSum_6h", "RainMax_3h", "RainMax_6h",
]
RH = [
    "RH_t", "RH_t_1", "RH_t_2", "RH_t_3", "RH_t_4", "RH_t_5", "RH_t_6",
    "RHMean_3h", "RHMean_6h", "RHMin_3h", "RHMax_3h", "RHMin_6h", "RHMax_6h",
]
TEMP = [
    "Temp_t", "Temp_t_1", "Temp_t_2", "Temp_t_3", "Temp_t_4", "Temp_t_5", "Temp_t_6",
    "TempMean_3h", "TempMin_3h", "TempMax_3h", "TempMean_6h", "TempMin_6h", "TempMax_6h",
]
TIME = ["Hour_sin", "Hour_cos", "Month_sin", "Month_cos"]
TERRAIN = ["Elevation_m", "SUHI_5K"]
LULC = ["LULC_Class"]
ERA5 = ["d2m", "u10", "v10", "q_850", "u_850", "v_850"]
CANDIDATES = RAIN + RH + TEMP + TIME + TERRAIN + LULC + ERA5
NUMERIC = [feature for feature in CANDIDATES if feature != "LULC_Class"]
IDENTIFIERS = ["Station", "Datetime", "Target_Datetime", "Role"]
TARGET = ["Target_Def5_tPlus1"]

FAMILY_FEATURES = {
    "Rainfall": RAIN,
    "Relative Humidity": RH,
    "Temperature": TEMP,
    "Temporal / seasonality": TIME,
    "Terrain / urban environment": TERRAIN,
    "LULC": LULC,
    "ERA5 atmospheric environment": ERA5,
}
FEATURE_FAMILY = {feature: family for family, features in FAMILY_FEATURES.items() for feature in features}

PROPOSED_RETAINED = [
    "Rain_t", "RainSum_3h", "RainSum_6h",
    "RH_t", "RHMin_6h",
    "Temp_t", "TempMax_6h",
    "Hour_sin", "Hour_cos", "Month_sin", "Month_cos",
    "Elevation_m", "LULC_Class", "SUHI_5K",
    "d2m", "u10", "v10", "q_850", "u_850", "v_850",
]

LULC_LEGEND = {
    1: ("นาข้าว", "Rice paddy"),
    2: ("พืชไร่ พืชสวน และพื้นที่เกษตรกรรมอื่น ๆ", "Field crops, horticulture, and other agricultural land"),
    3: ("ไม้ผลและไม้ยืนต้น", "Fruit trees and perennial crops"),
    4: ("ไร่หมุนเวียน", "Shifting cultivation"),
    5: ("ปศุสัตว์และการเพาะเลี้ยงสัตว์น้ำ", "Livestock and aquaculture"),
    6: ("ป่าไม้", "Forest"),
    7: ("พุ่มไม้ ทุ่งหญ้า และพื้นที่ชุ่มน้ำ", "Shrubland, grassland, and wetland"),
    8: ("เหมือง พื้นที่เสื่อมโทรม และพื้นที่กำจัดขยะ", "Mining, degraded land, and waste-disposal area"),
    9: ("ชุมชนและสิ่งปลูกสร้าง", "Communities and built structures"),
    10: ("แหล่งน้ำ", "Water bodies"),
}


def temporal_form(feature: str) -> tuple[str, int | None, int | None]:
    lag_match = re.search(r"_t_(\d+)$", feature)
    if lag_match:
        return "lagged", int(lag_match.group(1)), None
    if feature.endswith("_t"):
        return "current", 0, None
    window_match = re.search(r"_(3|6)h$", feature)
    if window_match:
        return "rolling-summary", None, int(window_match.group(1))
    if feature in TIME:
        return "cyclic", 0, None
    if feature in TERRAIN + LULC:
        return "static", None, None
    if feature in ERA5:
        return "ERA5 current", 0, None
    return "not applicable", None, None


def interpretation(feature: str) -> str:
    if feature.startswith("Rain"):
        if "Sum" in feature:
            return "Recent accumulated station rainfall, representing antecedent wetness and continuing precipitation."
        if "Max" in feature:
            return "Maximum recent hourly station rainfall intensity within the stated window."
        return "Observed hourly station rainfall at the stated predictor-time lag."
    if feature.startswith("RH"):
        if "Mean" in feature:
            return "Mean recent near-surface relative humidity."
        if "Min" in feature:
            return "Minimum recent relative humidity, representing the drier bound of the recent moisture envelope."
        if "Max" in feature:
            return "Maximum recent relative humidity, representing the wetter bound of the recent moisture envelope."
        return "Near-surface relative humidity at the stated predictor-time lag."
    if feature.startswith("Temp"):
        if "Mean" in feature:
            return "Mean recent near-surface air temperature."
        if "Min" in feature:
            return "Minimum recent air temperature within the stated window."
        if "Max" in feature:
            return "Maximum recent air temperature, representing recent surface heating."
        return "Near-surface air temperature at the stated predictor-time lag."
    meanings = {
        "Hour_sin": "Sine component of local hour, preserving the diurnal cycle without a discontinuity at midnight.",
        "Hour_cos": "Cosine component paired with Hour_sin to represent the diurnal cycle.",
        "Month_sin": "Sine component of calendar month, representing position within the annual cycle.",
        "Month_cos": "Cosine component paired with Month_sin to represent the annual cycle.",
        "Elevation_m": "Station elevation, representing terrain and orographic setting.",
        "LULC_Class": "Nominal station land-use/land-cover class from the project's ten-class legend.",
        "SUHI_5K": "Static 5-km SUHI context around the station; exact upstream unit remains undocumented.",
        "d2m": "ERA5 2-metre dewpoint temperature, representing near-surface atmospheric moisture.",
        "u10": "ERA5 eastward 10-m wind component.",
        "v10": "ERA5 northward 10-m wind component.",
        "q_850": "ERA5 specific humidity at 850 hPa, representing lower-tropospheric moisture.",
        "u_850": "ERA5 eastward wind component at 850 hPa.",
        "v_850": "ERA5 northward wind component at 850 hPa.",
    }
    return meanings[feature]


def previous_role(feature: str) -> str:
    if feature in ERA5[:3]:
        return "SFC and FULL raw architecture"
    if feature in ERA5[3:]:
        return "H850 and FULL raw architecture"
    return "BASE, SFC, H850, and FULL raw architecture"


def proposal_reason(feature: str) -> tuple[str, str]:
    retained_reasons = {
        "Rain_t": ("Retain", "Current local rainfall directly represents ongoing precipitation at forecast issuance."),
        "RainSum_3h": ("Retain", "Compact short-window accumulation for recent storm persistence and antecedent wetness."),
        "RainSum_6h": ("Retain", "Longer recent accumulation adds storm-history context beyond the 3-hour window."),
        "RH_t": ("Retain", "Current station humidity is the most immediate local near-surface moisture measure."),
        "RHMin_6h": ("Retain", "Represents the recent drier bound and humidity evolution while avoiding the full lag sequence."),
        "Temp_t": ("Retain", "Current near-surface thermal state is directly interpretable at a one-hour horizon."),
        "TempMax_6h": ("Retain", "Represents recent surface heating while avoiding the full temperature lag sequence."),
        "Hour_sin": ("Retain", "First component required for continuous diurnal-cycle representation."),
        "Hour_cos": ("Retain", "Second component required with Hour_sin to locate hour uniquely on the cycle."),
        "Month_sin": ("Retain", "First component representing within-season progression."),
        "Month_cos": ("Retain", "Second component required with Month_sin for a complete cyclic month representation."),
        "Elevation_m": ("Retain", "Preserves terrain and orographic context."),
        "LULC_Class": ("Retain", "Preserves land-cover context; proposed preprocessing is a source-verified built-up binary."),
        "SUHI_5K": ("Retain", "Preserves the available local urban-thermal context, subject to its unit limitation."),
        "d2m": ("Retain", "Preserves near-surface atmospheric moisture from ERA5."),
        "u10": ("Retain", "Eastward surface-wind component is required with v10 to retain wind direction and magnitude."),
        "v10": ("Retain", "Northward surface-wind component is required with u10 to retain wind direction and magnitude."),
        "q_850": ("Retain", "Preserves lower-tropospheric moisture distinct from surface dewpoint."),
        "u_850": ("Retain", "Eastward 850-hPa wind is required with v_850 to retain lower-level flow."),
        "v_850": ("Retain", "Northward 850-hPa wind is required with u_850 to retain lower-level flow."),
    }
    if feature in retained_reasons:
        return retained_reasons[feature]
    if feature.startswith("Rain_t_"):
        return "Exclude", "Not retained because the exact hourly lag sequence is summarized by Rain_t, RainSum_3h, and RainSum_6h."
    if feature == "RainMax_3h":
        return "Exclude", "Not retained because its Train Spearman rank correlation with RainSum_3h is nearly one; RainSum_3h represents the short window."
    if feature == "RainMax_6h":
        return "Exclude", "Not retained because its Train Spearman rank correlation with RainSum_6h is nearly one; RainSum_6h represents the longer window."
    if feature.startswith("RH"):
        return "Exclude", "Not retained because current humidity and the retained 6-hour minimum provide a compact local moisture state and recent envelope."
    if feature.startswith("Temp"):
        return "Exclude", "Not retained because current temperature and the retained 6-hour maximum provide a compact thermal state and recent-heating summary."
    raise KeyError(feature)


def replacement(feature: str) -> str:
    if feature in PROPOSED_RETAINED:
        return "Retained directly"
    if feature.startswith("Rain_t_"):
        return "Rain_t; RainSum_3h; RainSum_6h"
    if feature == "RainMax_3h":
        return "RainSum_3h"
    if feature == "RainMax_6h":
        return "RainSum_6h"
    if feature.startswith("RH"):
        return "RH_t; RHMin_6h"
    if feature.startswith("Temp"):
        return "Temp_t; TempMax_6h"
    return "Retained directly"


def connected_components(features: list[str], correlation: pd.DataFrame, threshold: float = 0.90) -> list[list[str]]:
    adjacency = {feature: set() for feature in features}
    for index, left in enumerate(features):
        for right in features[index + 1:]:
            if abs(correlation.at[left, right]) >= threshold:
                adjacency[left].add(right)
                adjacency[right].add(left)
    components: list[list[str]] = []
    unseen = set(features)
    while unseen:
        start = unseen.pop()
        stack = [start]
        component = {start}
        while stack:
            node = stack.pop()
            neighbors = adjacency[node] & unseen
            unseen -= neighbors
            component |= neighbors
            stack.extend(neighbors)
        if len(component) > 1:
            components.append(sorted(component))
    return sorted(components, key=lambda values: (FEATURE_FAMILY[values[0]], values))


def qualifying_pairs(features: list[str], correlation: pd.DataFrame) -> list[tuple[str, str, float]]:
    pairs = []
    for index, left in enumerate(features):
        for right in features[index + 1:]:
            rho = float(correlation.at[left, right])
            if abs(rho) >= 0.90:
                pairs.append((left, right, rho))
    return sorted(pairs, key=lambda values: -abs(values[2]))


def build_redundancy(correlation: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    group_number = 1
    for family in ["Rainfall", "Relative Humidity", "Temperature"]:
        features = FAMILY_FEATURES[family]
        for component in connected_components(features, correlation):
            pairs = qualifying_pairs(component, correlation)
            strongest = pairs[0]
            rows.append(
                {
                    "redundancy_group_id": f"RG{group_number:02d}",
                    "scope": "within-family",
                    "predictor_families": family,
                    "member_features": "; ".join(component),
                    "qualifying_pair_count_abs_rho_ge_0_90": len(pairs),
                    "minimum_abs_qualifying_rho": min(abs(value[2]) for value in pairs),
                    "maximum_abs_qualifying_rho": max(abs(value[2]) for value in pairs),
                    "strongest_pair": f"{strongest[0]} / {strongest[1]}",
                    "strongest_spearman_rho": strongest[2],
                    "qualifying_pairs": "; ".join(f"{a} / {b} ({rho:.6f})" for a, b, rho in pairs),
                    "interpretation": "Train-only rank correlations show overlapping temporal-state or rolling-summary information.",
                    "architecture_implication": "Diagnostic only; retain representative current and summary variables using physical interpretation, not an automatic correlation deletion rule.",
                }
            )
            group_number += 1

    cross_pairs = []
    for index, left in enumerate(NUMERIC):
        for right in NUMERIC[index + 1:]:
            if FEATURE_FAMILY[left] == FEATURE_FAMILY[right]:
                continue
            rho = float(correlation.at[left, right])
            if abs(rho) >= 0.90:
                cross_pairs.append((left, right, rho))
    if cross_pairs:
        cross_pairs.sort(key=lambda values: -abs(values[2]))
        members = sorted({item for pair in cross_pairs for item in pair[:2]})
        strongest = cross_pairs[0]
        rows.append(
            {
                "redundancy_group_id": f"RG{group_number:02d}",
                "scope": "cross-family",
                "predictor_families": "; ".join(sorted({FEATURE_FAMILY[m] for m in members})),
                "member_features": "; ".join(members),
                "qualifying_pair_count_abs_rho_ge_0_90": len(cross_pairs),
                "minimum_abs_qualifying_rho": min(abs(value[2]) for value in cross_pairs),
                "maximum_abs_qualifying_rho": max(abs(value[2]) for value in cross_pairs),
                "strongest_pair": f"{strongest[0]} / {strongest[1]}",
                "strongest_spearman_rho": strongest[2],
                "qualifying_pairs": "; ".join(f"{a} / {b} ({rho:.6f})" for a, b, rho in cross_pairs),
                "interpretation": "Recent relative humidity and temperature summaries covary inversely in the Train period; this is physically plausible thermodynamic covariation, not interchangeability.",
                "architecture_implication": "Preserve both physical families but avoid retaining every lag and summary from both.",
            }
        )
    return pd.DataFrame(rows)


def build_inventory(frame: pd.DataFrame, audit: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    audit = audit.set_index("Column_Name")
    group_map: dict[str, list[str]] = {feature: [] for feature in CANDIDATES}
    for row in groups.itertuples(index=False):
        for feature in row.member_features.split("; "):
            group_map.setdefault(feature, []).append(row.redundancy_group_id)
    rows: list[dict] = []
    for feature in IDENTIFIERS + TARGET + CANDIDATES:
        if feature in IDENTIFIERS:
            column_role = "Identifier / role metadata"
            family = "Identifier"
            feature_type = "timestamp" if "Datetime" in feature else ("role label" if feature == "Role" else "station identifier")
            form, lag, window = "not applicable", None, None
            status, rationale = "Not a predictor", "Used only to identify rows, assign roles, or verify timing."
            unit = "datetime (ICT)" if "Datetime" in feature else "none"
            meaning = "Phase 3 row identifier or temporal-role metadata."
            prior = "Key, timestamp, or split metadata; excluded from previous model matrices"
        elif feature in TARGET:
            column_role = "Target"
            family = "Target"
            feature_type = "binary target"
            form, lag, window = "target at t+1", None, None
            status, rationale = "Target only", "Outcome label; prohibited as a predictor."
            unit = "0/1"
            meaning = "Frozen Def5 outcome at Target_Datetime = t+1."
            prior = "Target only; excluded from predictor matrices"
        else:
            column_role = "Candidate predictor"
            family = FEATURE_FAMILY[feature]
            feature_type = "nominal categorical" if feature == "LULC_Class" else "continuous numeric"
            form, lag, window = temporal_form(feature)
            status, rationale = proposal_reason(feature)
            unit = str(audit.at[feature, "Unit"]) if feature in audit.index else "not documented"
            meaning = interpretation(feature)
            prior = previous_role(feature)
        rows.append(
            {
                "feature_name": feature,
                "column_role": column_role,
                "predictor_family": family,
                "feature_type": feature_type,
                "temporal_or_spatial_form": form,
                "lag_hours": lag,
                "rolling_window_hours": window,
                "units": unit,
                "physical_or_geographical_interpretation": meaning,
                "available_at_forecast_time_t": feature in CANDIDATES,
                "possible_redundancy": "; ".join(group_map.get(feature, [])),
                "original_role_previous_architecture": prior,
                "compact_proposal_status": status,
                "compact_proposal_rationale": rationale,
                "similar_information_represented_by": replacement(feature) if feature in CANDIDATES else "not applicable",
                "phase3_dtype": str(frame[feature].dtype),
            }
        )
    return pd.DataFrame(rows)


def build_lulc(frame: pd.DataFrame, legend_text: str) -> pd.DataFrame:
    for code, (thai, _) in LULC_LEGEND.items():
        assert f'{code}. {thai}' in legend_text, f"LULC legend class {code} not found in source XML"
    counts = frame.groupby("LULC_Class").agg(
        phase3_station_hours=("Station", "size"),
        stations_represented=("Station", "nunique"),
    )
    rows = []
    for code, (thai, english) in LULC_LEGEND.items():
        observed = code in counts.index
        binary = "Built-up" if code == 9 else "Non-built-up"
        if code == 9:
            rationale = "The source label explicitly combines communities and built structures."
            ambiguity = "Human approval still required; the source is a categorical land-cover class, not a measured built fraction."
        elif code == 8:
            rationale = "The mixed class is not explicitly a settlement/built-structure class, so Non-built-up is the literal binary mapping."
            ambiguity = "SEMANTIC CAUTION: mining, degraded land, and waste-disposal areas may contain anthropogenic or impervious surfaces; no Phase 3 station uses this class."
        else:
            rationale = "The source class is not labelled as communities or built structures."
            ambiguity = "No observed semantic conflict with a Built-up/Non-built-up label." if observed else "Class has no Phase 3 station-hours; mapping remains a legend-based proposal."
        rows.append(
            {
                "original_lulc_class_code": code,
                "original_class_name_thai": thai,
                "english_interpretation_of_source_label": english,
                "phase3_station_hours": int(counts.at[code, "phase3_station_hours"]) if observed else 0,
                "stations_represented": int(counts.at[code, "stations_represented"]) if observed else 0,
                "proposed_binary_category": binary,
                "rationale": rationale,
                "ambiguity_or_human_decision_required": ambiguity,
                "source_legend": "Workplace_GIS/LULC10.shp.xml",
            }
        )
    return pd.DataFrame(rows)


def preprocessing(feature: str) -> str:
    if feature == "LULC_Class":
        return "After approval: code 9 -> Built-up=1; source-verified other codes -> Non-built-up=0. One binary input; no old multiclass one-hot."
    if feature in TIME:
        return "Retain the existing sine/cosine values; Train-only standardization for LR if applied consistently."
    return "Numeric input; fit any LR scaling on Train only. RF/XGBoost may retain raw numeric scale."


def build_proposal() -> pd.DataFrame:
    rows = []
    for order, feature in enumerate(PROPOSED_RETAINED, start=1):
        _, rationale = proposal_reason(feature)
        form, _, _ = temporal_form(feature)
        rows.append(
            {
                "proposal_order": order,
                "raw_feature_name": feature,
                "predictor_family": FEATURE_FAMILY[feature],
                "feature_type": "nominal categorical proposed as binary" if feature == "LULC_Class" else "continuous numeric",
                "temporal_or_spatial_form": form,
                "retention_rationale": rationale,
                "redundant_features_it_may_replace": "; ".join(
                    candidate for candidate in CANDIDATES
                    if candidate not in PROPOSED_RETAINED and feature in replacement(candidate).split("; ")
                ),
                "expected_preprocessing": preprocessing(feature),
                "expected_model_input_columns_after_preprocessing": 1,
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in frame[columns].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "/") for value in row) + " |")
    return "\n".join(lines)


def build_report(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    inventory: pd.DataFrame,
    redundancy: pd.DataFrame,
    lulc: pd.DataFrame,
    proposal: pd.DataFrame,
) -> str:
    family_counts = inventory[inventory["column_role"].eq("Candidate predictor")].groupby("predictor_family").size().reset_index(name="count")
    retained = inventory[inventory["compact_proposal_status"].eq("Retain")]
    excluded = inventory[inventory["compact_proposal_status"].eq("Exclude")]
    phase3_roles = summary[summary["role"].isin(["Train", "Validation", "Evaluation"])][
        ["role", "final_eligible_rows", "final_eligible_positive_count", "final_eligible_station_count"]
    ].copy()
    phase3_roles.columns = ["role", "rows", "positives", "stations"]
    for column in ["rows", "positives", "stations"]:
        phase3_roles[column] = phase3_roles[column].map(lambda value: f"{int(value):,}")
    family_counts["count"] = family_counts["count"].astype(str)
    redundancy_display = redundancy[["redundancy_group_id", "predictor_families", "member_features", "strongest_pair", "strongest_spearman_rho"]].copy()
    redundancy_display["strongest_spearman_rho"] = redundancy_display["strongest_spearman_rho"].map(lambda value: f"{value:.3f}")
    retained_display = retained[["feature_name", "predictor_family", "compact_proposal_rationale"]]
    excluded_display = excluded[["feature_name", "compact_proposal_rationale", "similar_information_represented_by"]]
    lulc_display = lulc[["original_lulc_class_code", "english_interpretation_of_source_label", "phase3_station_hours", "stations_represented", "proposed_binary_category", "ambiguity_or_human_decision_required"]].copy()
    lulc_display["phase3_station_hours"] = lulc_display["phase3_station_hours"].map(lambda value: f"{int(value):,}")
    lulc_display["stations_represented"] = lulc_display["stations_represented"].astype(str)

    return f"""# Phase 4 — Compact Feature Architecture and LULC Representation

## 1. Phase 3 population used

The frozen Phase 3 May-October population was used without changing rows or roles.

{markdown_table(phase3_roles, ['role', 'rows', 'positives', 'stations'], ['Role', 'Rows', 'Def5 positives', 'Stations'])}

Only Train rows were used for numerical correlation diagnostics. Validation and Evaluation performance was not inspected, and Evaluation labels were not used in feature decisions.

## 2. Previous architecture size

The Phase 3 schema contains **50 candidate raw predictors**: 49 numeric/cyclic/static/ERA5 variables and one nominal `LULC_Class`. The previous multiclass LULC encoding expanded the complete architecture to approximately **57 transformed model inputs**. Identifiers (`Station`, `Datetime`, `Target_Datetime`, `Role`) and the target are separated from predictors in the inventory. Phase 3 contains no audit/QC predictor candidates.

## 3. Predictor-family inventory

{markdown_table(family_counts, ['predictor_family', 'count'], ['Predictor family', 'Original raw predictors'])}

The full feature-level inventory records type, lag/window, documented unit, forecast-time availability, physical interpretation, redundancy flags, and previous architecture role.

## 4. Main redundancy findings

Spearman correlations were calculated on **{len(frame.loc[frame['Role'].eq('Train')]):,} Train rows only**. Groups below contain at least one pair with `|rho| >= 0.90`.

{markdown_table(redundancy_display, ['redundancy_group_id', 'predictor_families', 'member_features', 'strongest_pair', 'strongest_spearman_rho'], ['Group', 'Family', 'Members', 'Strongest pair', 'Train Spearman rho'])}

The closest pairs are `RainSum_3h` / `RainMax_3h` and `RainSum_6h` / `RainMax_6h`. RH and temperature lag/summary families also contain extensive high rank correlation. Cross-family negative correlations reflect coupled humidity-temperature behavior and do not make the two physical families interchangeable. Correlation was used as a redundancy diagnostic only.

## 5. Proposed compact architecture

One architecture containing **{len(proposal)} raw predictors** is proposed. It retains local rainfall, local humidity and temperature, diurnal and seasonal timing, terrain/urban context, LULC, and all six existing ERA5 atmospheric variables. No new atmospheric field, lag, engineered feature, PCA component, or alternate feature set was introduced.

**TOTAL RAW PREDICTORS = {len(proposal)}**

**EXPECTED MODEL INPUT DIMENSION AFTER PREPROCESSING = {int(proposal['expected_model_input_columns_after_preprocessing'].sum())}**, if the proposed binary LULC mapping is approved.

## 6. Features proposed for retention and rationale

{markdown_table(retained_display, ['feature_name', 'predictor_family', 'compact_proposal_rationale'], ['Feature', 'Family', 'Why retained'])}

For LR, any numeric scaling must be fitted on Train only. RF and XGBoost may use the retained numeric values directly. This preprocessing statement is architectural guidance only; no model-ready v2 dataset was created.

## 7. Features proposed for exclusion and rationale

{markdown_table(excluded_display, ['feature_name', 'compact_proposal_rationale', 'similar_information_represented_by'], ['Feature', 'Why not retained in the compact proposal', 'Similar information retained in'])}

These exclusions do not claim scientific irrelevance. They reduce repeated representations of the same recent station history while preserving current conditions and interpretable short/long summaries.

## 8. LULC source legend and semantic interpretation

The actual legend was recovered from the configured GIS root at `LULC10.shp.xml`. The station assignment source is `LST/Station_Static_Predictors_v1.csv`. The legend defines ten classes; Phase 3 uses eight of them.

{markdown_table(lulc_display, ['original_lulc_class_code', 'english_interpretation_of_source_label', 'phase3_station_hours', 'stations_represented', 'proposed_binary_category', 'ambiguity_or_human_decision_required'], ['Code', 'Source-class interpretation', 'Phase 3 hours', 'Stations', 'Proposed binary', 'Ambiguity / decision'])}

Classes 5 and 8 have no Phase 3 station-hours. They remain in the review so the proposed rule is explicit for every source class rather than silently handling unseen codes.

## 9. Recommendation

**Built-up vs Non-built-up** is the recommended wording. Code 9 is explicitly “communities and built structures,” which is broader and more literal than “urban.” Codes 1-8 and 10 are proposed as Non-built-up because their source labels are not the built-structure class. No class is proposed for row exclusion.

Class 8 requires explicit human acknowledgement. Its combined mining, degraded-land, and waste-disposal label may contain anthropogenic or impervious surfaces even though it is not the source's settlement/built-structure class. It affects zero current Phase 3 stations, but the rule matters for reproducibility and any later station additions.

## 10. Proposed final raw predictor count

Previous architecture: **~50 raw predictors / 57 transformed inputs**

Proposed architecture: **{len(proposal)} raw predictors / expected {int(proposal['expected_model_input_columns_after_preprocessing'].sum())} transformed inputs**

The count is a result of domain coverage and redundancy review, not a target quota and not a performance claim.

## 11. Important limitations

- Spearman correlation describes Train-period rank association, not causation, substitutability, or predictive value.
- Heavy zero concentration in hourly rain helps explain the near-identical ranking of rainfall sums and maxima.
- The proposal preserves representative histories rather than every lag. A later approved study design could reasonably choose a different representative without contradicting this diagnostic.
- LULC is one static class at each station and does not measure within-buffer land-cover fractions or change through time.
- The LULC binary proposal collapses diverse agriculture, forest, wetland, water, and other non-built classes. It is suitable only for the narrow question of built-structure context.
- `SUHI_5K` remains physically interpretable as urban-thermal context, but its exact upstream unit and derivation are not documented in the current pipeline evidence.
- No Validation or Evaluation performance, model importance, or SHAP evidence informed the proposal.

## HUMAN DECISION REQUIRED

- The compact architecture is PROPOSED, not frozen.
- The LULC binary representation is PROPOSED, not frozen.
- No model was trained.
- No AP or ROC AUC was calculated.
- No Validation or Evaluation performance was used.
- No SHAP value was calculated.
- The researcher must approve the final feature set and LULC semantics.
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(UNIVERSE, columns=IDENTIFIERS + TARGET + CANDIDATES)
    assert len(frame) == 776_441
    assert set(frame.columns) == set(IDENTIFIERS + TARGET + CANDIDATES)
    assert len(CANDIDATES) == 50 and len(NUMERIC) == 49
    assert len(PROPOSED_RETAINED) == 20 and len(set(PROPOSED_RETAINED)) == 20
    assert set(PROPOSED_RETAINED) <= set(CANDIDATES)

    train = frame.loc[frame["Role"].eq("Train"), NUMERIC]
    assert len(train) == 525_474
    correlation = train.corr(method="spearman")
    redundancy = build_redundancy(correlation)
    assert len(redundancy) == 5

    audit = pd.read_csv(TYPE_AUDIT)
    inventory = build_inventory(frame, audit, redundancy)
    legend_text = LULC_XML.read_text(encoding="utf-8")
    lulc = build_lulc(frame, legend_text)
    proposal = build_proposal()
    summary = pd.read_csv(PHASE3_SUMMARY)

    station_source = pd.read_csv(LULC_STATION_SOURCE)
    station_source = station_source.loc[:, ~station_source.columns.str.startswith("Unnamed")]
    source_map = station_source.set_index("Station")["LULC10_Class"].astype("int16")
    phase3_map = frame.groupby("Station")["LULC_Class"].first().astype("int16")
    assert source_map.loc[phase3_map.index].equals(phase3_map)
    assert lulc["phase3_station_hours"].sum() == len(frame)
    assert lulc["stations_represented"].sum() == frame["Station"].nunique()
    assert inventory["feature_name"].is_unique and len(inventory) == 55
    assert inventory["column_role"].eq("Candidate predictor").sum() == 50
    assert inventory["compact_proposal_status"].eq("Retain").sum() == 20
    assert inventory["compact_proposal_status"].eq("Exclude").sum() == 30
    assert proposal["expected_model_input_columns_after_preprocessing"].sum() == 20

    inventory.to_csv(OUT / "Feature_Inventory_v1.csv", index=False, float_format="%.6f")
    redundancy.to_csv(OUT / "Feature_Redundancy_Groups_v1.csv", index=False, float_format="%.6f")
    lulc.to_csv(OUT / "LULC_Class_Review_v1.csv", index=False)
    proposal.to_csv(OUT / "Compact_Feature_Proposal_v1.csv", index=False)
    (OUT / "Feature_Architecture_Report_v1.md").write_text(
        build_report(frame, summary, inventory, redundancy, lulc, proposal), encoding="utf-8"
    )

    print(f"Original candidate predictors: {len(CANDIDATES)}")
    print(f"Proposed compact predictors: {len(PROPOSED_RETAINED)}")
    print(redundancy[["redundancy_group_id", "predictor_families", "strongest_pair", "strongest_spearman_rho"]].to_string(index=False))
    print(f"Wrote Phase 4 outputs to {OUT}")


if __name__ == "__main__":
    main()
