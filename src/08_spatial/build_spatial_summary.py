"""Build frozen point-station spatial summaries for Phase 8."""

from __future__ import annotations
import sys as _sys
from pathlib import Path as _ConfigPath
_sys.path.insert(0, str(_ConfigPath(__file__).resolve().parents[1]))
from rainbomb_paths import load_paths as _load_paths
PATHS = _load_paths()

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = PATHS.runtime_root
LEGACY = PATHS.upstream_root

EVALUATION = ROOT / "03_OUTPUT/06_Final_Model_Evaluation/Evaluation2025_Actual_vs_Predicted_v1.parquet"
SHAP_DATA = ROOT / "03_OUTPUT/07_SHAP_XAI/Evaluation2025_SHAP_v1.parquet"
STATION_SHAP_LONG = ROOT / "03_OUTPUT/07_SHAP_XAI/Station_SHAP_Long_v1.csv"
STATION_SHAP_SUMMARY = ROOT / "03_OUTPUT/07_SHAP_XAI/Station_SHAP_Summary_v1.csv"
LOCAL_SHAP_EXAMPLES = ROOT / "03_OUTPUT/07_SHAP_XAI/Local_SHAP_Examples_v1.csv"
MODEL_READY = ROOT / "03_OUTPUT/05_Model_Comparison/ModelReady_Seasonal_Compact20_v1.parquet"
STATION_METADATA = LEGACY / "Hourly_rain/RESETWORKPLACE/02_processed/Station_Metadata_Cleaned.csv"

OUT = ROOT / "03_OUTPUT/08_Spatial_Analysis"
STATION_OUTPUT = OUT / "Station_Evaluation_Spatial_Summary_v1.csv"
BUILT_OUTPUT = OUT / "BuiltUp_vs_NonBuiltUp_Spatial_Summary_v1.csv"
CASE_OUTPUT = OUT / "Spatial_Event_CaseStudies_v1.csv"
OUTPUTS = [STATION_OUTPUT, BUILT_OUTPUT, CASE_OUTPUT]

TARGET = "Target_Def5_tPlus1"
PREDICTORS = [
    "Rain_t", "RainSum_3h", "RainSum_6h", "RH_t", "RHMin_6h",
    "Temp_t", "TempMax_6h", "Hour_sin", "Hour_cos", "Month_sin",
    "Month_cos", "Elevation_m", "SUHI_5K", "BuiltUp_Binary", "d2m",
    "u10", "v10", "q_850", "u_850", "v_850",
]
SHAP_COLUMNS = [f"SHAP_{feature}" for feature in PREDICTORS]
EXPECTED_ROWS = 121_296
EXPECTED_POSITIVES = 95
EXPECTED_STATIONS = 38
EXPECTED_OUTCOMES = {"TP": 71, "FP": 24_193, "TN": 97_008, "FN": 24}
EXPECTED_HASHES = {
    EVALUATION: "5FAA5F5EA634F34384463C49FC8AE96A9DC7F90C9EEA0E12B1F5145C158AC43A",
    SHAP_DATA: "95918E14000E8DA89D6090CBB05BE14BF686E9EA444244DB66D0F4681B4C1D90",
    STATION_SHAP_LONG: "22022D92EE7CDC2D86764CC42C37A93CE1DD21DC81988D192E39AD9A6AC8E97B",
    STATION_SHAP_SUMMARY: "57F0C4D3C71082E5BD9727136B09DB699EE433B420E61837F3539C601261A04A",
    MODEL_READY: "D77C69A86F1B1251E09F37451A59F68E0A25C7F9307542B549B17E8B2E7D8A83",
    STATION_METADATA: "C8AD54E3CD5A3AD1591C7C9766C1B3296A67A9A5DD8D657C4FC60BD2573396BB",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def verify_inputs() -> None:
    required = list(EXPECTED_HASHES) + [LOCAL_SHAP_EXAMPLES]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing frozen Phase 5–7 input(s): {missing}")
    for path, expected in EXPECTED_HASHES.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Frozen-input SHA256 mismatch for {path}: {actual}")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    evaluation = pd.read_parquet(EVALUATION).reset_index(drop=True)
    shap = pd.read_parquet(SHAP_DATA).reset_index(drop=True)
    model_ready = pd.read_parquet(
        MODEL_READY,
        filters=[("Role", "==", "Evaluation")],
        columns=["Station", "Datetime", "Target_Datetime", TARGET,
                 "Elevation_m", "SUHI_5K", "BuiltUp_Binary"],
    ).reset_index(drop=True)
    metadata = pd.read_csv(STATION_METADATA)

    keys = ["Station", "Datetime", "Target_Datetime", TARGET]
    if not evaluation[keys].equals(shap[keys]) or not evaluation[keys].equals(model_ready[keys]):
        raise RuntimeError("Evaluation, SHAP, and model-ready row keys do not align exactly.")
    if len(evaluation) != EXPECTED_ROWS or int(evaluation[TARGET].sum()) != EXPECTED_POSITIVES:
        raise RuntimeError("Evaluation population differs from the frozen Phase 6 population.")
    if evaluation["Station"].nunique() != EXPECTED_STATIONS:
        raise RuntimeError("Evaluation station count differs from the frozen Phase 6 population.")
    if evaluation["Outcome_0p5"].value_counts().to_dict() != EXPECTED_OUTCOMES:
        raise RuntimeError("Threshold-0.5 outcomes differ from the frozen Phase 6 labels.")
    if set(evaluation["Station"]) - set(metadata["Station"]):
        raise RuntimeError("One or more Evaluation stations are absent from station metadata.")
    if metadata["Station"].duplicated().any() or metadata[["Longitude", "Latitude"]].isna().any().any():
        raise RuntimeError("Station metadata identifiers or coordinates are invalid.")

    station_shap_long = pd.read_csv(STATION_SHAP_LONG)
    station_shap_summary = pd.read_csv(STATION_SHAP_SUMMARY)
    if len(station_shap_long) != EXPECTED_STATIONS * len(PREDICTORS):
        raise RuntimeError("Station SHAP long table has an unexpected shape.")
    if len(station_shap_summary) != EXPECTED_STATIONS:
        raise RuntimeError("Station SHAP summary has an unexpected shape.")
    if set(station_shap_summary["Station"]) != set(evaluation["Station"]):
        raise RuntimeError("Station SHAP coverage differs from Evaluation coverage.")

    static_counts = model_ready.groupby("Station")[["Elevation_m", "SUHI_5K", "BuiltUp_Binary"]].nunique(dropna=False)
    if not static_counts.eq(1).all().all():
        raise RuntimeError("A frozen static predictor varies within at least one station.")
    return evaluation, shap, model_ready, metadata


def build_station_summary(
    evaluation: pd.DataFrame, model_ready: pd.DataFrame, metadata: pd.DataFrame
) -> pd.DataFrame:
    outcome_counts = (
        pd.crosstab(evaluation["Station"], evaluation["Outcome_0p5"])
        .reindex(columns=["TP", "FP", "TN", "FN"], fill_value=0)
        .astype("int64")
    )
    rows = []
    for station, group in evaluation.groupby("Station", sort=True):
        tp, fp, tn, fn = [int(outcome_counts.loc[station, label]) for label in ["TP", "FP", "TN", "FN"]]
        actual = tp + fn
        negatives = tn + fp
        predicted_positive = tp + fp
        rows.append(
            {
                "Station": station,
                "Evaluation_Rows": len(group),
                "Actual_Def5_Count": actual,
                "Def5_Rate_per_100k_rows": safe_ratio(actual * 100_000, len(group)),
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn,
                "Recall": safe_ratio(tp, actual),
                "Precision": safe_ratio(tp, predicted_positive),
                "Specificity": safe_ratio(tn, negatives),
                "False_Positive_Rate": safe_ratio(fp, negatives),
                "FP_per_10k_negative_rows": safe_ratio(fp * 10_000, negatives),
                "Median_XGB_Score": float(group["XGB_Final_Score"].median()),
                "Q95_XGB_Score": float(group["XGB_Final_Score"].quantile(0.95)),
            }
        )
    summary = pd.DataFrame(rows)

    static = (
        model_ready.groupby("Station", as_index=False)[["Elevation_m", "SUHI_5K", "BuiltUp_Binary"]]
        .first()
    )
    coordinates = metadata[["Station", "Station_Name", "Province", "Longitude", "Latitude"]]
    summary = (
        coordinates.merge(summary, on="Station", how="inner", validate="one_to_one")
        .merge(static, on="Station", how="inner", validate="one_to_one")
    )
    summary["BuiltUp_Binary"] = summary["BuiltUp_Binary"].astype("int8")
    columns = [
        "Station", "Station_Name", "Province", "Longitude", "Latitude",
        "Elevation_m", "SUHI_5K", "BuiltUp_Binary", "Evaluation_Rows",
        "Actual_Def5_Count", "Def5_Rate_per_100k_rows", "TP", "FP", "TN", "FN",
        "Recall", "Precision", "Specificity", "False_Positive_Rate",
        "FP_per_10k_negative_rows", "Median_XGB_Score", "Q95_XGB_Score",
    ]
    summary = summary[columns].sort_values("Station").reset_index(drop=True)

    if len(summary) != EXPECTED_STATIONS or int(summary["Evaluation_Rows"].sum()) != EXPECTED_ROWS:
        raise RuntimeError("Station summary coverage does not reconcile with Evaluation.")
    if int(summary["Actual_Def5_Count"].sum()) != EXPECTED_POSITIVES:
        raise RuntimeError("Station summary positives do not reconcile with Evaluation.")
    for label, expected in EXPECTED_OUTCOMES.items():
        if int(summary[label].sum()) != expected:
            raise RuntimeError(f"Station summary {label} total does not reconcile with Evaluation.")
    return summary


def build_builtup_summary(
    evaluation: pd.DataFrame, station_summary: pd.DataFrame
) -> pd.DataFrame:
    mapping = station_summary[["Station", "BuiltUp_Binary", "SUHI_5K"]]
    rows_with_context = evaluation.merge(mapping, on="Station", validate="many_to_one")
    records = []
    for built_value, label in [(0, "Non-built-up"), (1, "Built-up")]:
        stations = station_summary.loc[station_summary["BuiltUp_Binary"].eq(built_value)]
        group = rows_with_context.loc[rows_with_context["BuiltUp_Binary"].eq(built_value)]
        counts = group["Outcome_0p5"].value_counts()
        tp, fp, tn, fn = [int(counts.get(outcome, 0)) for outcome in ["TP", "FP", "TN", "FN"]]
        positives = tp + fn
        negatives = fp + tn
        estimable_recall = stations["Recall"].dropna()
        records.append(
            {
                "BuiltUp_Binary": built_value,
                "Group_Label": label,
                "Number_of_Stations": len(stations),
                "Evaluation_Rows": len(group),
                "Actual_Def5_Count": positives,
                "Def5_Rate_per_100k_rows": safe_ratio(positives * 100_000, len(group)),
                "Median_XGB_Score": float(group["XGB_Final_Score"].median()),
                "Q95_XGB_Score": float(group["XGB_Final_Score"].quantile(0.95)),
                "Recall": safe_ratio(tp, positives),
                "Stations_with_Estimable_Recall": int(estimable_recall.size),
                "Median_Station_Recall_among_Estimable": (
                    float(estimable_recall.median()) if len(estimable_recall) else float("nan")
                ),
                "False_Positive_Rate": safe_ratio(fp, negatives),
                "Mean_SUHI_5K": float(stations["SUHI_5K"].mean()),
                "Median_SUHI_5K": float(stations["SUHI_5K"].median()),
            }
        )
    result = pd.DataFrame(records)
    if int(result["Number_of_Stations"].sum()) != EXPECTED_STATIONS:
        raise RuntimeError("Built-up groups do not cover all Evaluation stations.")
    if int(result["Evaluation_Rows"].sum()) != EXPECTED_ROWS:
        raise RuntimeError("Built-up group rows do not reconcile with Evaluation.")
    return result


def build_case_studies(shap: pd.DataFrame, station_summary: pd.DataFrame) -> pd.DataFrame:
    local_phase7 = pd.read_csv(LOCAL_SHAP_EXAMPLES, parse_dates=["Datetime", "Target_Datetime"])
    context = station_summary.set_index("Station")
    records = []
    for outcome in ["TP", "FP", "FN"]:
        group = shap.loc[shap["Outcome_0p5"].eq(outcome)].copy()
        median_score = float(group["XGB_Final_Score"].median())
        group["_Distance"] = np.abs(group["XGB_Final_Score"] - median_score)
        row = group.sort_values(
            ["_Distance", "Station", "Datetime", "Target_Datetime"], kind="mergesort"
        ).iloc[0]
        phase7 = local_phase7.loc[local_phase7["Outcome"].eq(outcome)].iloc[0]
        if not (
            row["Station"] == phase7["Station"]
            and pd.Timestamp(row["Datetime"]) == pd.Timestamp(phase7["Datetime"])
            and pd.Timestamp(row["Target_Datetime"]) == pd.Timestamp(phase7["Target_Datetime"])
            and np.isclose(float(row["XGB_Final_Score"]), float(phase7["XGB_Final_Score"]))
        ):
            raise RuntimeError(f"Phase 8 {outcome} selection differs from frozen Phase 7 local example.")
        ranked = sorted(
            ((feature, float(row[f"SHAP_{feature}"])) for feature in PREDICTORS),
            key=lambda item: (-abs(item[1]), PREDICTORS.index(item[0])),
        )[:5]
        station_context = context.loc[row["Station"]]
        record = {
            "Outcome": outcome,
            "Station": row["Station"],
            "Longitude": float(station_context["Longitude"]),
            "Latitude": float(station_context["Latitude"]),
            "Datetime": row["Datetime"],
            "Target_Datetime": row["Target_Datetime"],
            "Actual": int(row[TARGET]),
            "XGB_Final_Score": float(row["XGB_Final_Score"]),
            "Predicted_0p5": int(float(row["XGB_Final_Score"]) >= 0.5),
            "BuiltUp_Binary": int(row["BuiltUp_Binary"]),
            "SUHI_5K": float(row["SUHI_5K"]),
            "Elevation_m": float(row["Elevation_m"]),
            "Rain_t": float(row["Rain_t"]),
            "RainSum_3h": float(row["RainSum_3h"]),
            "RainSum_6h": float(row["RainSum_6h"]),
            "RH_t": float(row["RH_t"]),
            "Temp_t": float(row["Temp_t"]),
            "Outcome_Group_Median_XGB_Score": median_score,
        }
        for rank, (feature, value) in enumerate(ranked, start=1):
            record[f"Top{rank}_Feature"] = feature
            record[f"Top{rank}_Signed_SHAP"] = value
        records.append(record)
    return pd.DataFrame(records)


def main() -> None:
    existing = [str(path) for path in OUTPUTS if path.exists()]
    if existing:
        raise RuntimeError(f"Refusing to overwrite existing Phase 8 table(s): {existing}")
    verify_inputs()
    evaluation, shap, model_ready, metadata = load_inputs()
    station_summary = build_station_summary(evaluation, model_ready, metadata)
    builtup_summary = build_builtup_summary(evaluation, station_summary)
    cases = build_case_studies(shap, station_summary)

    OUT.mkdir(parents=True, exist_ok=True)
    station_summary.to_csv(STATION_OUTPUT, index=False)
    builtup_summary.to_csv(BUILT_OUTPUT, index=False)
    cases.to_csv(CASE_OUTPUT, index=False)
    if not all(path.exists() for path in OUTPUTS):
        raise RuntimeError("One or more required Phase 8 CSV outputs were not created.")

    print("STATUS: PASS")
    print(f"Mapped Evaluation stations: {len(station_summary)}")
    print(f"Stations with estimable recall: {station_summary['Recall'].notna().sum()}")
    print(f"Stations without actual positives: {station_summary['Recall'].isna().sum()}")
    print("Built-up groups: " + str(dict(zip(builtup_summary["Group_Label"], builtup_summary["Number_of_Stations"]))))


if __name__ == "__main__":
    main()
