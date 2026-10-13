"""Freeze and refit the final Compact20 XGBoost on May-October 2020-2024.

The freeze manifest is written before model fitting. Evaluation 2025 is never
loaded by this script. The final round count comes from the frozen Phase 5
decision and is not re-estimated here.
"""

from __future__ import annotations
import sys as _sys
from pathlib import Path as _ConfigPath
_sys.path.insert(0, str(_ConfigPath(__file__).resolve().parents[1]))
from rainbomb_paths import load_paths as _load_paths
PATHS = _load_paths()

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from xgboost import XGBClassifier
except ModuleNotFoundError:
    XGB_SITE = PATHS.xgb_site_packages
    if not XGB_SITE.exists():
        raise
    sys.path.append(str(XGB_SITE))
    from xgboost import XGBClassifier


SEED = 154499
ROOT = PATHS.runtime_root
DATA = ROOT / "03_OUTPUT" / "05_Model_Comparison" / "ModelReady_Seasonal_Compact20_v1.parquet"
PHASE5_COMPARISON = ROOT / "03_OUTPUT" / "05_Model_Comparison" / "Model_Comparison_Validation_v1.csv"
OUT = ROOT / "03_OUTPUT" / "06_Final_Model_Evaluation"
MANIFEST = OUT / "Final_Model_Freeze_v1.md"
MODEL = OUT / "XGB_Final_Compact20_v1.json"
TRAINING_SUMMARY = OUT / "Final_Model_Training_Summary_v1.csv"
TARGET = "Target_Def5_tPlus1"
IDENTIFIERS = ["Station", "Datetime", "Target_Datetime", "Role"]
PREDICTORS = [
    "Rain_t", "RainSum_3h", "RainSum_6h", "RH_t", "RHMin_6h",
    "Temp_t", "TempMax_6h", "Hour_sin", "Hour_cos", "Month_sin",
    "Month_cos", "Elevation_m", "SUHI_5K", "BuiltUp_Binary", "d2m",
    "u10", "v10", "q_850", "u_850", "v_850",
]
EXPECTED = {
    "rows": 655_145,
    "positives": 386,
    "negatives": 654_759,
    "stations": 40,
}
FINAL_ROUNDS = 53
FROZEN_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "n_estimators": FINAL_ROUNDS,
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "random_state": SEED,
    "n_jobs": -1,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_phase5_selection() -> None:
    comparison = pd.read_csv(PHASE5_COMPARISON)
    required = {"Logistic Regression", "Random Forest", "XGBoost"}
    if set(comparison["Model"]) != required:
        raise RuntimeError("Phase 5 comparison does not contain the three frozen algorithms.")
    xgb = comparison.loc[comparison["Model"].eq("XGBoost")].iloc[0]
    if not np.isclose(float(xgb["AP"]), 0.02190539039002943, rtol=0, atol=1e-12):
        raise RuntimeError("Phase 5 XGBoost Validation AP differs from the frozen selection record.")
    if int(xgb["Best_Iteration"]) != 52:
        raise RuntimeError("Phase 5 XGBoost best_iteration is not the frozen value 52.")
    if not bool(xgb["AP"] == comparison["AP"].max()):
        raise RuntimeError("XGBoost is not the highest-AP model in the frozen Phase 5 table.")


def verify_development(frame: pd.DataFrame) -> None:
    if len(PREDICTORS) != 20 or len(set(PREDICTORS)) != 20:
        raise RuntimeError("Compact20 schema is not exactly 20 unique predictors.")
    if set(frame["Role"].unique()) != {"Train", "Validation"}:
        raise RuntimeError("Final development data contains a role other than Train or Validation.")
    observed = {
        "rows": len(frame),
        "positives": int(frame[TARGET].sum()),
        "negatives": int(len(frame) - frame[TARGET].sum()),
        "stations": int(frame["Station"].nunique()),
    }
    if observed != EXPECTED:
        raise RuntimeError(f"Final development population mismatch: expected {EXPECTED}, observed {observed}.")
    if frame.duplicated(["Station", "Datetime"]).any():
        raise RuntimeError("Duplicate Station + Datetime in final development data.")
    if frame.duplicated(["Station", "Target_Datetime"]).any():
        raise RuntimeError("Duplicate Station + Target_Datetime in final development data.")
    if not (frame["Target_Datetime"] - frame["Datetime"]).eq(pd.Timedelta(hours=1)).all():
        raise RuntimeError("Final development target lead is not exactly one hour.")
    if not frame["Target_Datetime"].dt.month.between(5, 10).all():
        raise RuntimeError("Final development data contains a target month outside May-October.")
    if not frame["Target_Datetime"].dt.year.between(2020, 2024).all():
        raise RuntimeError("Final development data contains a target year outside 2020-2024.")
    if not set(frame[TARGET].unique()).issubset({0, 1}):
        raise RuntimeError("Final development target is not binary.")
    if not np.isfinite(frame[PREDICTORS].to_numpy(dtype="float64", copy=False)).all():
        raise RuntimeError("Final development predictors are missing or non-finite.")


def build_manifest(scale_pos_weight: float, data_hash: str, created_utc: str) -> str:
    feature_lines = "\n".join(f"{index}. `{name}`" for index, name in enumerate(PREDICTORS, start=1))
    parameter_lines = "\n".join(f"- `{key} = {value}`" for key, value in FROZEN_PARAMS.items())
    return f"""# Final Model Freeze v1

Freeze timestamp (UTC): `{created_utc}`

**FINAL MODEL:** XGBoost Compact20

**DEVELOPMENT DATA:** May–October 2020–2024

**FINAL EVALUATION:** May–October 2025

**NUMBER OF RAW MODEL INPUTS:** 20

**FINAL BOOSTING ROUNDS:** 53

**THRESHOLD:** No operational threshold selected.

Threshold 0.5 is diagnostic only.

## Frozen scientific design

Predictors at station-hour t forecast `Target_Def5_tPlus1` at t+1 hour. The final development fit combines the frozen Train 2020–2023 and Validation 2024 May–October rows. Evaluation 2025 is excluded from fitting.

## Frozen predictors

{feature_lines}

## Frozen hyperparameters

{parameter_lines}

No early stopping is used in the final refit. The 53 rounds equal Phase 5's zero-based `best_iteration = 52` plus one.

## Final class weighting

The frozen rule is `scale_pos_weight = development negatives / development positives`. For 654,759 negatives and 386 positives, the derived value is `{scale_pos_weight:.12f}`. This applies the frozen rule to the combined development population and is not tuning.

## Population and provenance

- Development rows: 655,145
- Development positives: 386
- Development negatives: 654,759
- Development stations: 40
- Season: May–October
- Model-ready input SHA256: `{data_hash}`
- Random seed: 154499
- No threshold optimization
- No Evaluation use in model fitting

## Evaluation independence declaration

Evaluation 2025 has not informed:

- model selection
- feature selection
- hyperparameters
- number of boosting rounds
- class weighting rule
- threshold

This manifest is written before the final fitted model and before Evaluation 2025 is loaded or scored.
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    existing = [path for path in [MANIFEST, MODEL, TRAINING_SUMMARY] if path.exists()]
    if existing:
        raise RuntimeError(f"Refusing to overwrite existing Phase 6 freeze artifacts: {existing}")

    verify_phase5_selection()
    data_hash = sha256(DATA)
    columns = IDENTIFIERS + [TARGET] + PREDICTORS
    development = pd.read_parquet(
        DATA,
        columns=columns,
        filters=[("Role", "in", ["Train", "Validation"])],
    )
    verify_development(development)
    scale_pos_weight = EXPECTED["negatives"] / EXPECTED["positives"]
    observed_scale = (len(development) - int(development[TARGET].sum())) / int(development[TARGET].sum())
    if not np.isclose(scale_pos_weight, observed_scale, rtol=0, atol=1e-15):
        raise RuntimeError("Derived development scale_pos_weight is inconsistent.")

    freeze_utc = datetime.now(timezone.utc).isoformat()
    MANIFEST.write_text(build_manifest(scale_pos_weight, data_hash, freeze_utc), encoding="utf-8")
    if not MANIFEST.exists():
        raise RuntimeError("Freeze manifest was not written before final fitting.")

    x = development[PREDICTORS].astype("float32")
    y = development[TARGET].to_numpy(dtype="int8")
    model = XGBClassifier(**FROZEN_PARAMS, scale_pos_weight=scale_pos_weight)
    model.fit(x, y)
    booster = model.get_booster()
    if booster.num_boosted_rounds() != FINAL_ROUNDS:
        raise RuntimeError(
            f"Expected {FINAL_ROUNDS} final boosting rounds, found {booster.num_boosted_rounds()}."
        )
    booster.set_attr(
        phase6_algorithm="XGBoost Compact20",
        phase6_development_period="May-October 2020-2024",
        phase6_evaluation_used_in_fit="false",
        phase6_feature_names_json=json.dumps(PREDICTORS),
        phase6_frozen_params_json=json.dumps(FROZEN_PARAMS, sort_keys=True),
        phase6_manifest_sha256=sha256(MANIFEST),
        phase6_modelready_sha256=data_hash,
        phase6_n_estimators=str(FINAL_ROUNDS),
        phase6_scale_pos_weight=f"{scale_pos_weight:.16g}",
    )
    model.save_model(MODEL)
    if not MODEL.exists():
        raise RuntimeError("Final model was not saved.")
    if MODEL.stat().st_mtime_ns < MANIFEST.stat().st_mtime_ns:
        raise RuntimeError("Final model timestamp predates the freeze manifest.")

    model_hash = sha256(MODEL)
    summary = pd.DataFrame(
        [
            {
                "Status": "FROZEN",
                "Algorithm": "XGBoost Compact20",
                "Development_Period": "May-October 2020-2024",
                "Development_Rows": EXPECTED["rows"],
                "Development_Positives": EXPECTED["positives"],
                "Development_Negatives": EXPECTED["negatives"],
                "Development_Stations": EXPECTED["stations"],
                "Raw_Model_Inputs": len(PREDICTORS),
                "Scale_Pos_Weight_Rule": "development negatives / development positives",
                "Scale_Pos_Weight": scale_pos_weight,
                "Final_Boosting_Rounds": FINAL_ROUNDS,
                "Phase5_Best_Iteration_Zero_Based": 52,
                "Random_Seed": SEED,
                "Early_Stopping_Used_In_Final_Fit": False,
                "Evaluation_Data_Used_In_Fit": False,
                "Operational_Threshold_Selected": False,
                "ModelReady_SHA256": data_hash,
                "Freeze_Manifest_SHA256": sha256(MANIFEST),
                "Final_Model_SHA256": model_hash,
                "Freeze_UTC": freeze_utc,
                "Model_Saved_UTC": datetime.now(timezone.utc).isoformat(),
            }
        ]
    )
    summary.to_csv(TRAINING_SUMMARY, index=False)

    print("FINAL MODEL FREEZE: PASS")
    print(f"Development: {len(development):,} rows / {int(y.sum()):,} positives")
    print(f"scale_pos_weight: {scale_pos_weight:.12f}")
    print(f"Final boosting rounds: {booster.num_boosted_rounds()}")
    print(f"Manifest written before model: {MANIFEST.stat().st_mtime_ns <= MODEL.stat().st_mtime_ns}")
    print("Evaluation 2025 was not loaded or used by this script.")


if __name__ == "__main__":
    main()
