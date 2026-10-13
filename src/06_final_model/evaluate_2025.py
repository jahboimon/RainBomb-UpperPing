"""Score the frozen final XGBoost once on the untouched Evaluation 2025 role.

This script refuses to run if any Evaluation output already exists. It requires
the freeze manifest and fitted final model to predate Evaluation loading.
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
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
OUT = ROOT / "03_OUTPUT" / "06_Final_Model_Evaluation"
MANIFEST = OUT / "Final_Model_Freeze_v1.md"
MODEL = OUT / "XGB_Final_Compact20_v1.json"
TRAINING_SUMMARY = OUT / "Final_Model_Training_Summary_v1.csv"
METRICS = OUT / "Evaluation2025_Metrics_v1.csv"
PREDICTIONS = OUT / "Evaluation2025_Actual_vs_Predicted_v1.parquet"
SAMPLE = OUT / "Evaluation2025_Actual_vs_Predict_Sample_v1.csv"
CONFUSION_FIGURE = OUT / "Evaluation2025_ConfusionMatrix_v1.png"
PR_FIGURE = OUT / "Evaluation2025_PR_Curve_v1.png"
ROC_FIGURE = OUT / "Evaluation2025_ROC_Curve_v1.png"
REPORT = OUT / "Phase6_Final_Evaluation_Report_v1.md"
EVALUATION_OUTPUTS = [METRICS, PREDICTIONS, SAMPLE, CONFUSION_FIGURE, PR_FIGURE, ROC_FIGURE, REPORT]
TARGET = "Target_Def5_tPlus1"
IDENTIFIERS = ["Station", "Datetime", "Target_Datetime"]
PREDICTORS = [
    "Rain_t", "RainSum_3h", "RainSum_6h", "RH_t", "RHMin_6h",
    "Temp_t", "TempMax_6h", "Hour_sin", "Hour_cos", "Month_sin",
    "Month_cos", "Elevation_m", "SUHI_5K", "BuiltUp_Binary", "d2m",
    "u10", "v10", "q_850", "u_850", "v_850",
]
EXPECTED_ROWS = 121_296
EXPECTED_POSITIVES = 95
EXPECTED_STATIONS = 38
EXPECTED_SCALE_POS_WEIGHT = 654_759 / 386


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_freeze() -> XGBClassifier:
    missing = [path for path in [MANIFEST, MODEL, TRAINING_SUMMARY] if not path.exists()]
    if missing:
        raise RuntimeError(f"Freeze artifacts must exist before Evaluation scoring: {missing}")
    if MODEL.stat().st_mtime_ns < MANIFEST.stat().st_mtime_ns:
        raise RuntimeError("Final model predates the freeze manifest.")
    manifest = MANIFEST.read_text(encoding="utf-8")
    required_text = [
        "FINAL MODEL:** XGBoost Compact20",
        "FINAL BOOSTING ROUNDS:** 53",
        "No operational threshold selected",
        "Evaluation 2025 has not informed",
        "This manifest is written before the final fitted model",
    ]
    for text in required_text:
        if text not in manifest:
            raise RuntimeError(f"Freeze manifest is missing required declaration: {text}")

    summary = pd.read_csv(TRAINING_SUMMARY).iloc[0]
    if int(summary["Development_Rows"]) != 655_145 or int(summary["Development_Positives"]) != 386:
        raise RuntimeError("Final training summary population mismatch.")
    if int(summary["Final_Boosting_Rounds"]) != 53:
        raise RuntimeError("Final training summary does not freeze 53 rounds.")
    if not np.isclose(float(summary["Scale_Pos_Weight"]), EXPECTED_SCALE_POS_WEIGHT):
        raise RuntimeError("Final training summary scale_pos_weight mismatch.")
    if str(summary["Evaluation_Data_Used_In_Fit"]).lower() != "false":
        raise RuntimeError("Final training summary does not exclude Evaluation from fitting.")
    if summary["Final_Model_SHA256"] != sha256(MODEL):
        raise RuntimeError("Final model hash differs from the frozen training summary.")

    model = XGBClassifier()
    model.load_model(MODEL)
    booster = model.get_booster()
    if booster.num_boosted_rounds() != 53:
        raise RuntimeError("Saved final model does not contain exactly 53 boosting rounds.")
    attributes = booster.attributes()
    if attributes.get("phase6_evaluation_used_in_fit") != "false":
        raise RuntimeError("Saved final model metadata does not exclude Evaluation from fitting.")
    if json.loads(attributes["phase6_feature_names_json"]) != PREDICTORS:
        raise RuntimeError("Saved final model feature schema differs from Compact20.")
    return model


def verify_evaluation(frame: pd.DataFrame) -> None:
    if set(frame["Role"].unique()) != {"Evaluation"}:
        raise RuntimeError("Evaluation table contains a non-Evaluation role.")
    if len(frame) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS:,} Evaluation rows, found {len(frame):,}.")
    if int(frame[TARGET].sum()) != EXPECTED_POSITIVES:
        raise RuntimeError("Evaluation positive count mismatch.")
    if int(frame["Station"].nunique()) != EXPECTED_STATIONS:
        raise RuntimeError("Evaluation station count mismatch.")
    if frame.duplicated(["Station", "Datetime"]).any():
        raise RuntimeError("Duplicate Station + Datetime in Evaluation.")
    if frame.duplicated(["Station", "Target_Datetime"]).any():
        raise RuntimeError("Duplicate Station + Target_Datetime in Evaluation.")
    if not (frame["Target_Datetime"] - frame["Datetime"]).eq(pd.Timedelta(hours=1)).all():
        raise RuntimeError("Evaluation target lead is not exactly one hour.")
    if set(frame["Target_Datetime"].dt.year.unique()) != {2025}:
        raise RuntimeError("Evaluation contains a target year other than 2025.")
    if not frame["Target_Datetime"].dt.month.between(5, 10).all():
        raise RuntimeError("Evaluation contains a target month outside May-October.")
    if len(PREDICTORS) != 20 or not np.isfinite(
        frame[PREDICTORS].to_numpy(dtype="float64", copy=False)
    ).all():
        raise RuntimeError("Evaluation Compact20 schema is incomplete or non-finite.")


def outcome_labels(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return np.select(
        [
            (actual == 1) & (predicted == 1),
            (actual == 0) & (predicted == 1),
            (actual == 0) & (predicted == 0),
            (actual == 1) & (predicted == 0),
        ],
        ["TP", "FP", "TN", "FN"],
        default="ERROR",
    )


def save_confusion(tn: int, fp: int, fn: int, tp: int) -> None:
    matrix = np.array([[tn, fp], [fn, tp]], dtype=int)
    fig, ax = plt.subplots(figsize=(6.3, 5.4))
    image = ax.imshow(matrix, cmap="Blues")
    threshold = (matrix.max() + matrix.min()) / 2
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{matrix[i, j]:,}", ha="center", va="center", fontsize=15,
                fontweight="bold", color="white" if matrix[i, j] > threshold else "#1F2937",
            )
    ax.set_xticks([0, 1], labels=["Predicted 0", "Predicted 1"])
    ax.set_yticks([0, 1], labels=["Actual 0", "Actual 1"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title("Final XGBoost — Evaluation 2025\nThreshold 0.5 diagnostic", pad=14)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Evaluation rows")
    fig.tight_layout()
    fig.savefig(CONFUSION_FIGURE, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_curves(actual: np.ndarray, scores: np.ndarray, prevalence: float, ap: float, auc: float) -> None:
    precision, recall, _ = precision_recall_curve(actual, scores)
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    ax.plot(recall, precision, color="#D55E00", lw=2, label=f"Final XGBoost (AP={ap:.4f})")
    ax.axhline(prevalence, color="#6B7280", lw=1.4, linestyle="--", label=f"Prevalence={prevalence:.6f}")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1), title="Final XGBoost — Evaluation 2025 Precision–Recall Curve")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(PR_FIGURE, dpi=220, bbox_inches="tight")
    plt.close(fig)

    false_positive_rate, true_positive_rate, _ = roc_curve(actual, scores)
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    ax.plot(false_positive_rate, true_positive_rate, color="#D55E00", lw=2, label=f"Final XGBoost (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#6B7280", lw=1.2, linestyle="--", label="No-skill reference")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1), title="Final XGBoost — Evaluation 2025 ROC Curve")
    ax.grid(alpha=0.22)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(ROC_FIGURE, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(row: pd.Series, sample_rows: int) -> None:
    report = f"""# Phase 6 — Final XGBoost Freeze and Temporal Evaluation 2025

## 1. Why XGBoost was selected in Phase 5

The primary model-selection metric was pre-specified as Average Precision. Phase 5 Validation AP was 0.002351 for Logistic Regression, 0.010527 for Random Forest, and 0.021905 for XGBoost. XGBoost Compact20 was therefore selected and frozen before Evaluation 2025 was accessed. LR and RF were not reconsidered using Evaluation results.

## 2. Frozen Compact20 architecture

The final estimator uses exactly 20 raw predictors: `Rain_t`, `RainSum_3h`, `RainSum_6h`, `RH_t`, `RHMin_6h`, `Temp_t`, `TempMax_6h`, `Hour_sin`, `Hour_cos`, `Month_sin`, `Month_cos`, `Elevation_m`, `SUHI_5K`, `BuiltUp_Binary`, `d2m`, `u10`, `v10`, `q_850`, `u_850`, and `v_850`. No predictor was added, removed, engineered, selected, or transformed beyond the frozen architecture.

## 3. Final training population: May–October 2020–2024

The final refit combined Train 2020–2023 and Validation 2024: 655,145 rows, 386 positives, 654,759 negatives, and 40 stations. Evaluation 2025 was excluded from fitting.

## 4. Final hyperparameters

The frozen model used `objective=binary:logistic`, `eval_metric=aucpr`, `tree_method=hist`, `n_estimators=53`, `learning_rate=0.05`, `max_depth=4`, `min_child_weight=5`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0`, `reg_alpha=0.0`, `random_state=154499`, and `n_jobs=-1`. No Evaluation `eval_set` and no early stopping were used.

## 5. Why final boosting rounds equal 53

Phase 5 early stopping reported zero-based `best_iteration=52` on Validation 2024. The final fixed estimator count is therefore 52 + 1 = 53 boosting rounds. Early stopping was not rerun during the final refit.

## 6. Final scale_pos_weight

The frozen rule was reapplied to the combined development population: 654,759 negatives / 386 positives = {EXPECTED_SCALE_POS_WEIGHT:.12f}. This data-derived update is not hyperparameter tuning.

## 7. Model freeze preceded Evaluation scoring

`Final_Model_Freeze_v1.md` was written first. The 53-round final model was then fitted and saved as `XGB_Final_Compact20_v1.json`. The evaluation script verified the manifest, fitted-model hash, round count, feature schema, and file chronology before loading Evaluation 2025.

## 8. Evaluation 2025 population

Evaluation contains {int(row['Evaluation_Rows']):,} May–October 2025 rows, {int(row['Evaluation_Positives']):,} positives, {int(row['Evaluation_Negatives']):,} negatives, and {int(row['Evaluation_Stations']):,} stations. All 20 predictors are complete and finite. Keys are unique and the target lead is exactly one hour.

## 9. Evaluation Average Precision

Final Evaluation AP is **{float(row['AP']):.8f}**. AP / PR-AUC is the primary temporal-evaluation metric because Def5 is extremely imbalanced.

## 10. Evaluation ROC AUC

Final Evaluation ROC AUC is **{float(row['ROC_AUC']):.8f}**. ROC AUC is secondary.

## 11. Evaluation prevalence

Evaluation prevalence is {float(row['Evaluation_Prevalence']):.8f} ({float(row['Evaluation_Prevalence']) * 100:.5f}%). AP / prevalence is {float(row['AP_over_Prevalence']):.6f}. This ratio is a baseline-relative comparison and does not mean “times more accurate.”

## 12. Threshold-0.5 confusion matrix

Threshold 0.5 is descriptive only. No operational threshold was selected.

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | {int(row['TN']):,} | {int(row['FP']):,} |
| Actual 1 | {int(row['FN']):,} | {int(row['TP']):,} |

## 13. Threshold-0.5 precision, recall, specificity, and F1

At threshold 0.5, precision is {float(row['Precision']):.8f}, recall is {float(row['Recall']):.8f}, specificity is {float(row['Specificity']):.8f}, and F1 is {float(row['F1']):.8f}. These values describe one fixed operating point and were not used to alter the model.

## 14. PR curve interpretation

The PR curve shows precision-recall behavior for the frozen final XGBoost across possible score cutoffs and includes the Evaluation prevalence baseline. The curve is descriptive; no cutoff was selected from it.

## 15. ROC curve interpretation

The ROC curve shows sensitivity versus false-positive rate across possible score cutoffs for the frozen final estimator. It is a secondary summary and was not used for post-Evaluation tuning.

## 16. Limitations

- Def5 events remain extremely rare, so apparently small false-positive rates can correspond to many false-positive station-hours.
- The model score is not a calibrated probability.
- Threshold 0.5 is diagnostic and is not an operational warning threshold.
- Phase 5 Validation metrics came from the Train-only development model with Validation early stopping. Phase 6 Evaluation metrics come from a different fitted estimator refit on combined Train + Validation data. They are not a strict paired performance-change test.
- Temporal Evaluation 2025 assesses one future season and does not by itself establish spatial transferability.

## 17. No Evaluation-based tuning occurred

Evaluation 2025 did not inform model selection, feature selection, hyperparameters, boosting rounds, class-weighting rule, or threshold. Whatever the Evaluation result, no model redesign, threshold search, calibration, or post-hoc rescue was performed.

## 18. Readiness for Phase 7 SHAP

Phase 6 is complete and the frozen final model plus Evaluation score table are ready for a separately authorized Phase 7 explanation workflow. No SHAP, feature importance, PDP, spatial SHAP, station SHAP, spatial analysis, or spatial prediction was performed in Phase 6.

## Validation-versus-Evaluation context

Historical Phase 5 XGBoost Validation metrics were AP 0.021905 and ROC AUC 0.819619. They came from the Train-only development model with Validation early stopping. The Phase 6 metrics come from the final 53-round refit trained on combined Train + Validation. The two results are not from the identical fitted estimator and must not be treated as a strict paired performance-change test. No tuning was performed in response to their difference.

## Actual-vs-Predict output

`Evaluation2025_Actual_vs_Predicted_v1.parquet` stores every Evaluation row with the final model score, threshold-0.5 prediction, and TP/FP/TN/FN label. `Evaluation2025_Actual_vs_Predict_Sample_v1.csv` contains {sample_rows} rows selected reproducibly within available outcome groups using seed 154499, without selecting examples based on attractive scores.

## STATUS

PASS
"""
    REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    existing = [path for path in EVALUATION_OUTPUTS if path.exists()]
    if existing:
        raise RuntimeError(
            "Evaluation outputs already exist; refusing to score Evaluation 2025 more than once: "
            f"{existing}"
        )

    model = verify_freeze()

    # Evaluation data is loaded only after the manifest and final model pass verification.
    columns = IDENTIFIERS + ["Role", TARGET] + PREDICTORS
    evaluation = pd.read_parquet(
        DATA,
        columns=columns,
        filters=[("Role", "==", "Evaluation")],
    )
    verify_evaluation(evaluation)
    x_evaluation = evaluation[PREDICTORS].astype("float32")
    actual = evaluation[TARGET].to_numpy(dtype="int8")

    # The only model-scoring call for Evaluation 2025 in this workflow.
    scores = model.predict_proba(x_evaluation)[:, 1]
    if not np.isfinite(scores).all():
        raise RuntimeError("Final Evaluation model scores contain non-finite values.")
    predicted = (scores >= 0.5).astype("int8")
    outcomes = outcome_labels(actual, predicted)
    if "ERROR" in outcomes:
        raise RuntimeError("Unexpected Evaluation outcome classification.")

    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    prevalence = float(actual.mean())
    ap = float(average_precision_score(actual, scores))
    auc = float(roc_auc_score(actual, scores))
    row = {
        "Status": "PASS",
        "Model": "XGBoost Compact20 Final",
        "Evaluation_Period": "May-October 2025",
        "Evaluation_Rows": len(evaluation),
        "Evaluation_Positives": int(actual.sum()),
        "Evaluation_Negatives": int(len(actual) - actual.sum()),
        "Evaluation_Stations": int(evaluation["Station"].nunique()),
        "Evaluation_Prevalence": prevalence,
        "AP": ap,
        "ROC_AUC": auc,
        "AP_over_Prevalence": ap / prevalence,
        "Threshold": 0.5,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "Precision": float(precision_score(actual, predicted, zero_division=0)),
        "Recall": float(recall_score(actual, predicted, zero_division=0)),
        "Specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "F1": float(f1_score(actual, predicted, zero_division=0)),
        "Operational_Threshold_Selected": False,
        "Evaluation_Based_Tuning_Performed": False,
        "Final_Model_SHA256": sha256(MODEL),
    }
    metrics = pd.DataFrame([row])

    predictions = evaluation[IDENTIFIERS + [TARGET]].reset_index(drop=True).copy()
    predictions["XGB_Final_Score"] = scores.astype("float32")
    predictions["XGB_Predicted_0p5"] = predicted
    predictions["Outcome_0p5"] = outcomes

    selected: list[int] = []
    for number, outcome in enumerate(["TP", "FP", "TN", "FN"]):
        candidates = predictions.index[predictions["Outcome_0p5"].eq(outcome)].to_numpy()
        if len(candidates):
            rng = np.random.default_rng(SEED + number)
            selected.extend(
                map(int, rng.choice(candidates, size=min(5, len(candidates)), replace=False))
            )
    sample = predictions.loc[sorted(set(selected))].copy()

    predictions.to_parquet(PREDICTIONS, index=False, compression="zstd")
    sample.to_csv(SAMPLE, index=False)
    metrics.to_csv(METRICS, index=False)
    save_confusion(int(tn), int(fp), int(fn), int(tp))
    save_curves(actual, scores, prevalence, ap, auc)
    write_report(metrics.iloc[0], len(sample))

    if not all(path.exists() for path in EVALUATION_OUTPUTS):
        raise RuntimeError("One or more required Evaluation outputs were not produced.")
    print("STATUS: PASS")
    print(f"Evaluation: {len(evaluation):,} rows / {int(actual.sum()):,} positives / {evaluation['Station'].nunique():,} stations")
    print(f"Prevalence: {prevalence:.12f}")
    print(f"AP: {ap:.12f}")
    print(f"ROC AUC: {auc:.12f}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: [[{tn}, {fp}], [{fn}, {tp}]]")
    print(
        f"Precision={row['Precision']:.12f}, Recall={row['Recall']:.12f}, "
        f"Specificity={row['Specificity']:.12f}, F1={row['F1']:.12f}"
    )
    print("Evaluation 2025 scored exactly once after model freeze; no tuning followed.")


if __name__ == "__main__":
    main()
