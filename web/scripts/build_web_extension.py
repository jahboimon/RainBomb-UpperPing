"""Build read-only web derivatives for the RainBomb Atlas extension.

The script reads frozen research artifacts and writes only under this web project.
It does not refit, recalibrate, tune, or modify any research source artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from rainbomb_paths import load_paths  # noqa: E402

PATHS = load_paths()
ROOT = REPO_ROOT / "web"
DERIVED = ROOT / "data" / "derived"
MODEL_SOURCE = PATHS.final_analysis_root / "03_OUTPUT/06_Final_Model_Evaluation/XGB_Final_Compact20_v1.json"
MODELREADY_SOURCE = PATHS.final_analysis_root / "03_OUTPUT/05_Model_Comparison/ModelReady_Seasonal_Compact20_v1.parquet"
EVALUATION_SOURCE = PATHS.final_analysis_root / "03_OUTPUT/06_Final_Model_Evaluation/Evaluation2025_Actual_vs_Predicted_v1.parquet"
SHAP_SOURCE = PATHS.final_analysis_root / "03_OUTPUT/07_SHAP_XAI/Evaluation2025_SHAP_v1.parquet"
DEF5_TARGET_SOURCE = PATHS.upstream_root / "Def5_Final_Target_2012_2025_v1/02_Def5_Extended_Target/RainBomb_Target_Def5_2012_2025_v1.parquet"

FEATURES = [
    "Rain_t", "RainSum_3h", "RainSum_6h", "RH_t", "RHMin_6h", "Temp_t",
    "TempMax_6h", "Hour_sin", "Hour_cos", "Month_sin", "Month_cos",
    "Elevation_m", "SUHI_5K", "BuiltUp_Binary", "d2m", "u10", "v10",
    "q_850", "u_850", "v_850",
]
KEYS = ["Station", "Datetime", "Target_Datetime"]
SOURCE_HASHES = {
    str(MODEL_SOURCE): "0c43852b2cd1f60dc89efd225331bd6251691a8f835ea3c6df65cb5cfaa274cd",
    str(MODELREADY_SOURCE): "d77c69a86f1b1251e09f37451a59f68e0a25c7f9307542b549b17e8b2e7d8a83",
    str(EVALUATION_SOURCE): "5faa5f5ea634f34384463c49fc8ae96a9dc7f90c9eea0e12b1f5145c158ac43a",
    str(SHAP_SOURCE): "95918e14000e8da89d6090cbb05be14bf686e9ea444244db66d0f4681b4c1d90",
    str(DEF5_TARGET_SOURCE): "2e6d3e0379caa07700f42bc4c7124ac43236ad8f22b82d85ff6fc073d8cbd9eb",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sources() -> None:
    for raw_path, expected in SOURCE_HASHES.items():
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Frozen artifact hash mismatch: {path}\n{actual} != {expected}")


def json_value(value):
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%dT%H:%M:%S")
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def record(row: pd.Series, columns: list[str]) -> dict:
    return {column: json_value(row[column]) for column in columns}


def predict_json_model(model: dict, values: dict[str, float]) -> float:
    learner = model["learner"]
    names = learner["feature_names"]
    trees = learner["gradient_booster"]["model"]["trees"]
    base = float(learner["learner_model_param"]["base_score"].strip("[]"))
    margin = math.log(base / (1.0 - base))
    for tree in trees:
        node = 0
        while tree["left_children"][node] != -1:
            value = np.float32(values[names[tree["split_indices"][node]]])
            if np.isnan(value):
                go_left = bool(tree["default_left"][node])
            else:
                go_left = value < tree["split_conditions"][node]
            node = tree["left_children"][node] if go_left else tree["right_children"][node]
        margin += tree["split_conditions"][node]
    return 1.0 / (1.0 + math.exp(-margin))


def evenly_spaced(group: pd.DataFrame, count: int) -> pd.DataFrame:
    if len(group) <= count:
        return group
    positions = np.linspace(0, len(group) - 1, count, dtype=int)
    return group.iloc[positions]


def main() -> None:
    verify_sources()
    DERIVED.mkdir(parents=True, exist_ok=True)
    model_dir = DERIVED / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    model = json.loads(MODEL_SOURCE.read_text(encoding="utf-8"))
    if model["learner"]["feature_names"] != FEATURES:
        raise RuntimeError("Frozen model feature order differs from Compact20 expectation")

    shap_columns = KEYS + ["Target_Def5_tPlus1", "XGB_Final_Score", "Outcome_0p5"] + FEATURES + [f"SHAP_{name}" for name in FEATURES] + ["SHAP_Bias"]
    evaluation = pd.read_parquet(SHAP_SOURCE, columns=shap_columns).sort_values(KEYS).reset_index(drop=True)
    if len(evaluation) != 121_296 or int(evaluation["Target_Def5_tPlus1"].sum()) != 95:
        raise RuntimeError("Evaluation row or positive count differs from frozen definition")

    metadata = json.loads((DERIVED / "vector" / "station_metadata.geojson").read_text(encoding="utf-8"))
    names = {feature["properties"]["Station"]: feature["properties"].get("Station_Name") for feature in metadata["features"]}

    # Small, deterministic teaching sample: TP, FN and FP without frequency claims.
    teaching = pd.concat([
        evenly_spaced(evaluation[evaluation["Outcome_0p5"] == outcome], 3)
        for outcome in ["TP", "FN", "FP"]
    ]).sort_values(["Outcome_0p5", "Target_Datetime", "Station"]).reset_index(drop=True)
    raw_rows = []
    feature_rows = []
    model_rows = []
    for _, row in teaching.iterrows():
        common = {
            "Station": row["Station"],
            "Station_Name": names.get(row["Station"]),
            "Datetime": json_value(row["Datetime"]),
        }
        raw_rows.append(common | {
            "Rainfall": json_value(row["Rain_t"]),
            "Relative_Humidity": json_value(row["RH_t"]),
            "Temperature": json_value(row["Temp_t"]),
        })
        feature_rows.append(common | {name: json_value(row[name]) for name in FEATURES})
        model_rows.append(common | {
            "Target_Datetime": json_value(row["Target_Datetime"]),
            "Target_Def5_tPlus1": int(row["Target_Def5_tPlus1"]),
            "XGB_Final_Score": float(row["XGB_Final_Score"]),
            "Outcome_0p5": row["Outcome_0p5"],
        } | {name: json_value(row[name]) for name in FEATURES})

    lineage_payload = {
        "title": "RainBomb Atlas small data-lineage sample",
        "scope": "Evaluation 2025 teaching sample; not representative of class frequency",
        "explanation_th": "ตัวอย่างนี้แสดงการเปลี่ยนจากข้อมูลรายชั่วโมงของสถานี ไปสู่ตัวแปรที่ใช้โดยแบบจำลอง และการจับคู่ข้อมูลเวลา t กับเหตุการณ์ Def5 ที่เวลา t+1 ชั่วโมง",
        "source_artifacts": [str(MODELREADY_SOURCE), str(EVALUATION_SOURCE), str(SHAP_SOURCE)],
        "selection": "Three deterministic real rows each from TP, FN and FP; no synthetic values.",
        "raw_canonical": raw_rows,
        "feature_engineered": feature_rows,
        "model_ready": model_rows,
    }
    lineage_path = DERIVED / "samples_data_lineage_2025.json"
    lineage_path.write_text(json.dumps(lineage_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Restricted simulator baseline set: all actual Def5 cases plus the highest-score FP per station.
    positives = evaluation[evaluation["Target_Def5_tPlus1"] == 1]
    fp = evaluation[evaluation["Outcome_0p5"] == "FP"].sort_values("XGB_Final_Score", ascending=False).groupby("Station", as_index=False).head(1)
    candidates = pd.concat([positives, fp]).drop_duplicates(KEYS).sort_values(["Station", "Target_Datetime"]).reset_index(drop=True)
    rain_lookup = {(row.Station, row.Datetime): float(row.Rain_t) for row in evaluation[["Station", "Datetime", "Rain_t"]].itertuples(index=False)}
    all_event_metadata = pd.read_parquet(DEF5_TARGET_SOURCE)
    event_metadata = all_event_metadata[
        (all_event_metadata["Datetime"] >= pd.Timestamp("2025-05-01"))
        & (all_event_metadata["Datetime"] < pd.Timestamp("2025-11-01"))
    ]
    target_rain_lookup = {
        (row.Station, row.Datetime): float(row.Rainfall)
        for row in event_metadata.itertuples(index=False)
    }
    baselines = []
    for _, row in candidates.iterrows():
        lag_values = []
        complete = True
        for lag in range(6):
            value = rain_lookup.get((row["Station"], row["Datetime"] - pd.Timedelta(hours=lag)))
            if value is None:
                complete = False
                break
            lag_values.append(value)
        if not complete:
            continue
        sum3 = sum(lag_values[:3])
        sum6 = sum(lag_values)
        if not (math.isclose(sum3, float(row["RainSum_3h"]), abs_tol=1e-8) and math.isclose(sum6, float(row["RainSum_6h"]), abs_tol=1e-8)):
            continue
        baselines.append({
            "id": f"{row['Station']}|{json_value(row['Datetime'])}",
            "Station": row["Station"],
            "Station_Name": names.get(row["Station"]),
            "Datetime": json_value(row["Datetime"]),
            "Target_Datetime": json_value(row["Target_Datetime"]),
            "Target_Def5_tPlus1": int(row["Target_Def5_tPlus1"]),
            "Target_Rainfall_tPlus1": target_rain_lookup.get((row["Station"], row["Target_Datetime"])),
            "XGB_Final_Score": float(row["XGB_Final_Score"]),
            "Outcome_0p5": row["Outcome_0p5"],
            "rain_lags": {f"Rain_tMinus{lag}": lag_values[lag] for lag in range(6)},
            "features": {name: json_value(row[name]) for name in FEATURES},
            "baseline_shap": {name: float(row[f"SHAP_{name}"]) for name in FEATURES},
            "shap_bias": float(row["SHAP_Bias"]),
        })
    positive_baselines = sum(item["Target_Def5_tPlus1"] for item in baselines)
    outcomes_present = {item["Outcome_0p5"] for item in baselines}
    if positive_baselines < 80 or not {"TP", "FN", "FP"}.issubset(outcomes_present):
        raise RuntimeError(f"Insufficient verified scenario coverage: {positive_baselines}, {sorted(outcomes_present)}")

    range_features = ["Rain_t", "RH_t", "Temp_t", "q_850", "u_850", "v_850"]
    ranges = {}
    for feature in range_features:
        values = evaluation[feature].astype(float)
        ranges[feature] = {
            "min": float(values.min()), "p01": float(values.quantile(0.01)),
            "p99": float(values.quantile(0.99)), "max": float(values.max()),
        }
    for lag in range(1, 6):
        ranges[f"Rain_tMinus{lag}"] = dict(ranges["Rain_t"])
    ranges["Hour"] = {"min": 0, "p01": 0, "p99": 23, "max": 23}

    scenario_payload = {
        "title": "RainBomb Atlas station-based scenario baselines",
        "scope": "Real Evaluation 2025 cases only",
        "source_artifact": str(SHAP_SOURCE),
        "source_artifacts": [str(SHAP_SOURCE), str(DEF5_TARGET_SOURCE)],
        "selection": "Real TP/FN rows plus the highest-score FP per station, restricted to cases with exact six-hour rainfall context whose sums reproduce frozen features; cases without complete exposed lag rows are omitted rather than imputed.",
        "baseline_count": len(baselines),
        "actual_def5_count": positive_baselines,
        "ranges": ranges,
        "baselines": baselines,
    }
    scenario_path = DERIVED / "scenario_baselines_2025.json"
    scenario_path.write_text(json.dumps(scenario_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Official observed Def5 event catalogue for historical browsing only.
    # It remains separate from model outcomes, which are available for Evaluation 2025 only.
    metadata_by_station = {feature["properties"]["Station"]: feature for feature in metadata["features"]}
    historical_features = []
    for row in all_event_metadata.sort_values(["Datetime", "Station"]).itertuples(index=False):
        station_feature = metadata_by_station.get(row.Station)
        if station_feature is None or not station_feature.get("geometry"):
            raise RuntimeError(f"Missing web geometry for official Def5 station: {row.Station}")
        station_properties = station_feature["properties"]
        historical_features.append({
            "type": "Feature",
            "properties": {
                "Station": row.Station,
                "Station_Name": station_properties.get("Station_Name"),
                "Province": station_properties.get("Province"),
                "District": station_properties.get("District"),
                "Target_Datetime": json_value(row.Datetime),
                "Rainfall": json_value(row.Rainfall),
                "P99_Frozen": json_value(row.P99_Frozen),
                "Antecedent_3h_Rain": json_value(row.Antecedent_3h_Rain),
                "Post1_Rain": json_value(row.Post1_Rain),
                "Post2_Rain": json_value(row.Post2_Rain),
                "Post3_Rain": json_value(row.Post3_Rain),
                "Post3_Max": json_value(row.Post3_Max),
                "Def5": int(row.Def5),
                "Observation_Scope": "Official Def5 2012–2025; no model outcome attached",
            },
            "geometry": station_feature["geometry"],
        })
    historical_payload = {
        "type": "FeatureCollection",
        "name": "Official_Def5_Events_2012_2025",
        "features": historical_features,
    }
    historical_path = DERIVED / "vector" / "def5_events_2012_2025.geojson"
    historical_path.write_text(json.dumps(historical_payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    # Reproducible validation rows spanning the complete Evaluation table.
    validation_rows = evaluation.iloc[np.linspace(0, len(evaluation) - 1, 512, dtype=int)].copy()
    errors = []
    validation = []
    for _, row in validation_rows.iterrows():
        values = {name: float(row[name]) for name in FEATURES}
        predicted = predict_json_model(model, values)
        saved = float(row["XGB_Final_Score"])
        errors.append(abs(predicted - saved))
        validation.append({
            "Station": row["Station"], "Datetime": json_value(row["Datetime"]),
            "features": values, "saved_score": saved,
        })
    max_error = max(errors)
    tolerance = 1e-6
    if max_error > tolerance:
        raise RuntimeError(f"Exact JSON traversal failed validation: {max_error} > {tolerance}")
    validation_payload = {
        "model_sha256": SOURCE_HASHES[str(MODEL_SOURCE)],
        "source_artifact": str(SHAP_SOURCE),
        "sampling": "512 evenly spaced rows in deterministic sorted Evaluation order",
        "row_count": len(validation),
        "offline_max_absolute_error": max_error,
        "tolerance": tolerance,
        "offline_status": "PASS",
        "rows": validation,
    }
    validation_path = model_dir / "xgb_validation_2025.json"
    validation_path.write_text(json.dumps(validation_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    model_copy = model_dir / "xgb_final_compact20_v1.json"
    shutil.copyfile(MODEL_SOURCE, model_copy)
    if sha256(model_copy) != SOURCE_HASHES[str(MODEL_SOURCE)]:
        raise RuntimeError("Copied browser model does not match frozen model")

    manifest_path = DERIVED / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["scope"] = "Evaluation 2025 model outputs plus a separate official observed Def5 event catalogue for 2012–2025."
    manifest["timezone"] = "ERA5 was converted from UTC to Asia/Bangkok (ICT, UTC+7) before station-hour joining; station-source timezone remains unverified, so stored timestamps are displayed without further conversion."
    additions = [
        {
            "derived_file": "dist/data/derived/samples_data_lineage_2025.json",
            "source": f"{MODELREADY_SOURCE}; {EVALUATION_SOURCE}; {SHAP_SOURCE}",
            "record_count": len(teaching),
            "conversion": "Small deterministic real-row teaching sample; exact key reuse; no synthetic values or scientific recalculation.",
            "size_bytes": lineage_path.stat().st_size,
            "sha256": sha256(lineage_path),
        },
        {
            "derived_file": "dist/data/derived/scenario_baselines_2025.json",
            "source": f"{SHAP_SOURCE}; {DEF5_TARGET_SOURCE}",
            "record_count": len(baselines),
            "conversion": "Real Evaluation baselines with exact station-hour rainfall lag lookup; frozen 3 h and 6 h sums verified.",
            "size_bytes": scenario_path.stat().st_size,
            "sha256": sha256(scenario_path),
        },
        {
            "derived_file": "dist/data/derived/model/xgb_validation_2025.json",
            "source": f"{MODEL_SOURCE}; {SHAP_SOURCE}",
            "record_count": len(validation),
            "conversion": f"Deterministic validation sample; exact JSON traversal max absolute error {max_error:.3e} at tolerance {tolerance:.1e}.",
            "size_bytes": validation_path.stat().st_size,
            "sha256": sha256(validation_path),
        },
        {
            "derived_file": "dist/data/derived/model/xgb_final_compact20_v1.json",
            "source": str(MODEL_SOURCE),
            "record_count": 53,
            "conversion": "Byte-for-byte copy of the frozen XGBoost JSON model for exact static browser inference; no conversion.",
            "size_bytes": model_copy.stat().st_size,
            "sha256": sha256(model_copy),
        },
    ]
    existing = {item["derived_file"] for item in manifest.get("tabular", [])}
    manifest["tabular"] = [item for item in manifest.get("tabular", []) if item["derived_file"] not in {addition["derived_file"] for addition in additions}]
    manifest["tabular"].extend(additions)
    historical_manifest = {
        "derived_file": "dist/data/derived/vector/def5_events_2012_2025.geojson",
        "source": str(DEF5_TARGET_SOURCE),
        "source_crs": "Station coordinates joined from the verified web station-metadata derivative",
        "web_crs": "EPSG:4326 (RFC 7946)",
        "feature_count": len(historical_features),
        "fields": list(historical_features[0]["properties"].keys()),
        "conversion": "Exact observed Def5 event attributes joined to station geometry by Station; no model outcome inferred and no scientific recalculation.",
        "size_bytes": historical_path.stat().st_size,
        "sha256": sha256(historical_path),
    }
    manifest["vectors"] = [item for item in manifest.get("vectors", []) if item["derived_file"] != historical_manifest["derived_file"]]
    manifest["vectors"].append(historical_manifest)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "lineage_rows": len(teaching), "scenario_baselines": len(baselines),
        "actual_def5_baselines": positive_baselines, "validation_rows": len(validation),
        "official_def5_events": len(historical_features),
        "max_absolute_error": max_error, "tolerance": tolerance, "status": "PASS",
    }, indent=2))


if __name__ == "__main__":
    main()
