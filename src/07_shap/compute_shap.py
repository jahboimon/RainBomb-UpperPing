"""Compute native TreeSHAP values for the frozen final XGBoost model."""

from __future__ import annotations
import sys as _sys
from pathlib import Path as _ConfigPath
_sys.path.insert(0, str(_ConfigPath(__file__).resolve().parents[1]))
from rainbomb_paths import load_paths as _load_paths
PATHS = _load_paths()

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ModuleNotFoundError:
    XGB_SITE = PATHS.xgb_site_packages
    if not XGB_SITE.exists():
        raise
    sys.path.append(str(XGB_SITE))
    import xgboost as xgb


ROOT = PATHS.runtime_root
MODEL = ROOT / "03_OUTPUT/06_Final_Model_Evaluation/XGB_Final_Compact20_v1.json"
MANIFEST = ROOT / "03_OUTPUT/06_Final_Model_Evaluation/Final_Model_Freeze_v1.md"
TRAINING_SUMMARY = ROOT / "03_OUTPUT/06_Final_Model_Evaluation/Final_Model_Training_Summary_v1.csv"
MODEL_READY = ROOT / "03_OUTPUT/05_Model_Comparison/ModelReady_Seasonal_Compact20_v1.parquet"
SCORES = ROOT / "03_OUTPUT/06_Final_Model_Evaluation/Evaluation2025_Actual_vs_Predicted_v1.parquet"
OUT_DIR = ROOT / "03_OUTPUT/07_SHAP_XAI"
OUTPUT = OUT_DIR / "Evaluation2025_SHAP_v1.parquet"

TARGET = "Target_Def5_tPlus1"
IDENTIFIERS = ["Station", "Datetime", "Target_Datetime"]
PREDICTORS = [
    "Rain_t", "RainSum_3h", "RainSum_6h", "RH_t", "RHMin_6h",
    "Temp_t", "TempMax_6h", "Hour_sin", "Hour_cos", "Month_sin",
    "Month_cos", "Elevation_m", "SUHI_5K", "BuiltUp_Binary", "d2m",
    "u10", "v10", "q_850", "u_850", "v_850",
]
SHAP_COLUMNS = [f"SHAP_{feature}" for feature in PREDICTORS]
EXPECTED_MODEL_SHA256 = "0C43852B2CD1F60DC89EFD225331BD6251691A8F835EA3C6DF65CB5CFAA274CD"
EXPECTED_ROWS = 121_296
EXPECTED_POSITIVES = 95
EXPECTED_STATIONS = 38
TARGET_TOLERANCE = 1e-6
MATERIAL_MARGIN_TOLERANCE = 1e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype="float64")
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def load_and_verify_model() -> xgb.Booster:
    required = [MODEL, MANIFEST, TRAINING_SUMMARY, MODEL_READY, SCORES]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing frozen Phase 5/6 input(s): {missing}")
    if sha256(MODEL) != EXPECTED_MODEL_SHA256:
        raise RuntimeError("Final-model SHA256 differs from the frozen Phase 6 model.")

    summary = pd.read_csv(TRAINING_SUMMARY).iloc[0]
    if str(summary["Final_Model_SHA256"]).upper() != EXPECTED_MODEL_SHA256:
        raise RuntimeError("Training-summary model hash does not match the frozen model.")

    booster = xgb.Booster()
    booster.load_model(MODEL)
    if booster.num_boosted_rounds() != 53:
        raise RuntimeError("Frozen model does not contain exactly 53 boosting rounds.")
    if booster.feature_names != PREDICTORS:
        raise RuntimeError("Frozen model feature order differs from Compact20.")
    attributes = booster.attributes()
    if attributes.get("phase6_evaluation_used_in_fit") != "false":
        raise RuntimeError("Frozen model metadata does not exclude Evaluation from fitting.")
    if json.loads(attributes["phase6_feature_names_json"]) != PREDICTORS:
        raise RuntimeError("Frozen model metadata feature schema differs from Compact20.")
    return booster


def load_and_verify_evaluation() -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = IDENTIFIERS + ["Role", TARGET] + PREDICTORS
    evaluation = pd.read_parquet(
        MODEL_READY, columns=columns, filters=[("Role", "==", "Evaluation")]
    ).reset_index(drop=True)
    scores = pd.read_parquet(SCORES).reset_index(drop=True)

    if set(evaluation["Role"].unique()) != {"Evaluation"}:
        raise RuntimeError("Model-ready subset contains a non-Evaluation role.")
    if len(evaluation) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS:,} rows; found {len(evaluation):,}.")
    if int(evaluation[TARGET].sum()) != EXPECTED_POSITIVES:
        raise RuntimeError("Evaluation Def5-positive count differs from the frozen population.")
    if evaluation["Station"].nunique() != EXPECTED_STATIONS:
        raise RuntimeError("Evaluation station count differs from the frozen population.")
    if evaluation.duplicated(IDENTIFIERS).any():
        raise RuntimeError("Evaluation identifiers are not unique.")
    if not np.isfinite(evaluation[PREDICTORS].to_numpy(dtype="float64")).all():
        raise RuntimeError("One or more Compact20 predictor values are non-finite.")

    keys = IDENTIFIERS + [TARGET]
    if len(scores) != len(evaluation) or not evaluation[keys].equals(scores[keys]):
        raise RuntimeError("Frozen Evaluation score rows do not align exactly with model-ready rows.")
    if set(scores["Outcome_0p5"].unique()) != {"TP", "FP", "TN", "FN"}:
        raise RuntimeError("Saved threshold-0.5 outcomes are incomplete.")
    if scores["XGB_Final_Score"].isna().any():
        raise RuntimeError("Saved final-model scores contain missing values.")
    return evaluation, scores


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"Refusing to overwrite existing Phase 7 SHAP output: {OUTPUT}")
    booster = load_and_verify_model()
    evaluation, scores = load_and_verify_evaluation()

    x_values = evaluation[PREDICTORS].astype("float32")
    matrix = xgb.DMatrix(x_values, feature_names=PREDICTORS)
    contributions = booster.predict(matrix, pred_contribs=True, validate_features=True)
    raw_margin = booster.predict(matrix, output_margin=True, validate_features=True)
    if contributions.shape != (EXPECTED_ROWS, len(PREDICTORS) + 1):
        raise RuntimeError(f"Unexpected TreeSHAP array shape: {contributions.shape}")
    if not np.isfinite(contributions).all() or not np.isfinite(raw_margin).all():
        raise RuntimeError("TreeSHAP contributions or raw margins contain non-finite values.")

    reconstructed_margin = (
        contributions[:, :-1].astype("float64").sum(axis=1)
        + contributions[:, -1].astype("float64")
    )
    margin_error = float(
        np.max(np.abs(reconstructed_margin - raw_margin.astype("float64")))
    )
    score_error = float(
        np.max(
            np.abs(
                sigmoid(reconstructed_margin)
                - scores["XGB_Final_Score"].to_numpy(dtype="float64")
            )
        )
    )
    if margin_error > MATERIAL_MARGIN_TOLERANCE or score_error > TARGET_TOLERANCE:
        raise RuntimeError(
            "STOP_REVIEW_REQUIRED: native TreeSHAP reconstruction failed; "
            f"margin error={margin_error:.12g}, score error={score_error:.12g}."
        )
    status = "PASS" if margin_error <= TARGET_TOLERANCE else "PASS_WITH_CAVEATS"

    output = scores[
        IDENTIFIERS + [TARGET, "XGB_Final_Score", "Outcome_0p5"]
    ].copy()
    for feature in PREDICTORS:
        output[feature] = evaluation[feature]
    shap_frame = pd.DataFrame(
        contributions[:, :-1], columns=SHAP_COLUMNS, dtype="float32"
    )
    output = pd.concat([output, shap_frame], axis=1)
    output["SHAP_Bias"] = contributions[:, -1].astype("float32")

    expected_columns = (
        IDENTIFIERS
        + [TARGET, "XGB_Final_Score", "Outcome_0p5"]
        + PREDICTORS
        + SHAP_COLUMNS
        + ["SHAP_Bias"]
    )
    if output.columns.tolist() != expected_columns or len(output) != EXPECTED_ROWS:
        raise RuntimeError("Final SHAP table schema or row count is incorrect.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT, index=False, compression="zstd")
    print(f"STATUS: {status}")
    print(f"Rows explained: {len(output):,}")
    print(f"Max absolute raw-margin reconstruction error: {margin_error:.12g}")
    print(f"Max absolute score reconstruction error: {score_error:.12g}")
    print("SHAP scale: raw model margin / log-odds")


if __name__ == "__main__":
    main()
