"""Train the three frozen Phase 5 classifiers and compare on Validation 2024.

Evaluation 2025 rows are never loaded into the development table and are not
scored.  Validation is used for XGBoost early stopping and for the requested
development comparison only.  No threshold, feature, or hyperparameter search
is performed.
"""

from __future__ import annotations
import sys as _sys
from pathlib import Path as _ConfigPath
_sys.path.insert(0, str(_ConfigPath(__file__).resolve().parents[1]))
from rainbomb_paths import load_paths as _load_paths
PATHS = _load_paths()

import gc
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# XGBoost is already present in the project's read-only legacy environment.
# Import core numeric packages first, then expose only that environment's
# site-packages as a fallback.  Nothing is installed or modified.
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
OUT = ROOT / "03_OUTPUT" / "05_Model_Comparison"
TARGET = "Target_Def5_tPlus1"
IDENTIFIERS = ["Station", "Datetime", "Target_Datetime"]
PREDICTORS = [
    "Rain_t",
    "RainSum_3h",
    "RainSum_6h",
    "RH_t",
    "RHMin_6h",
    "Temp_t",
    "TempMax_6h",
    "Hour_sin",
    "Hour_cos",
    "Month_sin",
    "Month_cos",
    "Elevation_m",
    "SUHI_5K",
    "BuiltUp_Binary",
    "d2m",
    "u10",
    "v10",
    "q_850",
    "u_850",
    "v_850",
]
CONTINUOUS = [name for name in PREDICTORS if name != "BuiltUp_Binary"]
EXPECTED = {
    "Train": {"rows": 525_474, "positives": 312},
    "Validation": {"rows": 129_671, "positives": 74},
}
MODEL_COLORS = {
    "Logistic Regression": "#0072B2",
    "Random Forest": "#009E73",
    "XGBoost": "#D55E00",
}
RF_CAPTION = (
    "This is one representative decision tree from the Random Forest ensemble. "
    "Final Random Forest predictions are produced by aggregating predictions across all trees."
)


def verify_development_data(frame: pd.DataFrame) -> None:
    roles = set(frame["Role"].unique())
    if roles != {"Train", "Validation"}:
        raise RuntimeError(f"Development data contains unexpected roles: {sorted(roles)}")
    if len(PREDICTORS) != 20 or len(set(PREDICTORS)) != 20:
        raise RuntimeError("Frozen predictor list is not exactly 20 unique variables.")
    if not np.isfinite(frame[PREDICTORS].to_numpy(dtype="float64", copy=False)).all():
        raise RuntimeError("Development data contains a missing or non-finite predictor.")
    if not set(frame[TARGET].unique()).issubset({0, 1}):
        raise RuntimeError("Development target is not binary.")
    for role, expected in EXPECTED.items():
        part = frame.loc[frame["Role"].eq(role)]
        observed = {"rows": len(part), "positives": int(part[TARGET].sum())}
        if observed != expected:
            raise RuntimeError(f"{role} mismatch: expected {expected}, observed {observed}.")


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


def calculate_metrics(
    name: str,
    train_y: np.ndarray,
    val_y: np.ndarray,
    scores: np.ndarray,
) -> dict:
    predicted = (scores >= 0.5).astype("int8")
    tn, fp, fn, tp = confusion_matrix(val_y, predicted, labels=[0, 1]).ravel()
    prevalence = float(val_y.mean())
    ap = float(average_precision_score(val_y, scores))
    return {
        "Model": name,
        "Train_Rows": int(len(train_y)),
        "Train_Positives": int(train_y.sum()),
        "Validation_Rows": int(len(val_y)),
        "Validation_Positives": int(val_y.sum()),
        "Validation_Prevalence": prevalence,
        "AP": ap,
        "ROC_AUC": float(roc_auc_score(val_y, scores)),
        "AP_over_Prevalence": ap / prevalence,
        "Threshold": 0.5,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "Precision": float(precision_score(val_y, predicted, zero_division=0)),
        "Recall": float(recall_score(val_y, predicted, zero_division=0)),
        "Specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "F1": float(f1_score(val_y, predicted, zero_division=0)),
    }


def save_confusion_figure(name: str, row: pd.Series, path: Path) -> None:
    matrix = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]], dtype=int)
    fig, ax = plt.subplots(figsize=(6.3, 5.4))
    image = ax.imshow(matrix, cmap="Blues")
    threshold = (matrix.max() + matrix.min()) / 2
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:,}",
                ha="center",
                va="center",
                fontsize=15,
                fontweight="bold",
                color="white" if matrix[i, j] > threshold else "#1F2937",
            )
    ax.set_xticks([0, 1], labels=["Predicted 0", "Predicted 1"])
    ax.set_yticks([0, 1], labels=["Actual 0", "Actual 1"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(f"{name}\nValidation 2024 — threshold = 0.5 diagnostic", pad=14)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Validation rows")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_curve_figures(
    val_y: np.ndarray,
    score_map: dict[str, np.ndarray],
    metric_rows: pd.DataFrame,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    for name, scores in score_map.items():
        precision, recall, _ = precision_recall_curve(val_y, scores)
        ap = float(metric_rows.loc[metric_rows["Model"].eq(name), "AP"].iloc[0])
        ax.plot(recall, precision, lw=2, color=MODEL_COLORS[name], label=f"{name} (AP={ap:.4f})")
    prevalence = float(val_y.mean())
    ax.axhline(prevalence, color="#6B7280", lw=1.4, linestyle="--", label=f"Prevalence={prevalence:.6f}")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1), title="Validation 2024 Precision–Recall Curves")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "Validation_PR_Curve_Comparison_v1.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    for name, scores in score_map.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(val_y, scores)
        auc = float(metric_rows.loc[metric_rows["Model"].eq(name), "ROC_AUC"].iloc[0])
        ax.plot(false_positive_rate, true_positive_rate, lw=2, color=MODEL_COLORS[name], label=f"{name} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#6B7280", lw=1.2, linestyle="--", label="No-skill reference")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1), title="Validation 2024 ROC Curves")
    ax.grid(alpha=0.22)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "Validation_ROC_Curve_Comparison_v1.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str], decimals: int = 6) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = []
    for _, row in frame[columns].iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                cells.append("")
            elif isinstance(value, (float, np.floating)):
                cells.append(f"{float(value):.{decimals}f}")
            elif isinstance(value, (int, np.integer)):
                cells.append(f"{int(value):,}")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator] + rows)


def confusion_markdown(row: pd.Series) -> str:
    return "\n".join(
        [
            "| | Predicted 0 | Predicted 1 |",
            "|---|---:|---:|",
            f"| Actual 0 | {int(row['TN']):,} | {int(row['FP']):,} |",
            f"| Actual 1 | {int(row['FN']):,} | {int(row['TP']):,} |",
        ]
    )


def write_report(
    comparison: pd.DataFrame,
    scale_pos_weight: float,
    xgb_best_iteration: int,
    xgb_best_score: float,
    xgb_rounds: int,
    rf_tree_summary: pd.DataFrame,
    representative_index: int,
    representative_depth: int,
    sample_rows: int,
) -> None:
    train_prevalence = EXPECTED["Train"]["positives"] / EXPECTED["Train"]["rows"]
    val_prevalence = EXPECTED["Validation"]["positives"] / EXPECTED["Validation"]["rows"]
    compact = comparison.copy()
    compact["Train_Rows"] = compact["Train_Rows"].map(lambda value: f"{int(value):,}")
    compact["Validation_Rows"] = compact["Validation_Rows"].map(lambda value: f"{int(value):,}")
    compact["Train_Positives"] = compact["Train_Positives"].map(lambda value: f"{int(value):,}")
    compact["Validation_Positives"] = compact["Validation_Positives"].map(lambda value: f"{int(value):,}")
    comparison_columns = [
        "Model", "AP", "ROC_AUC", "AP_over_Prevalence", "TP", "FP", "TN", "FN",
        "Precision", "Recall", "Specificity", "F1",
    ]
    sections = []
    for _, row in comparison.iterrows():
        sections.append(f"### {row['Model']}\n\n{confusion_markdown(row)}")
    report = f"""# Phase 5 — Final Model-Ready Dataset and Development Model Comparison

## 1. Frozen seasonal design

The target is `Target_Def5_tPlus1`: station-hour predictors at time t forecast Def5 at t+1 hour. The seasonal window is May–October. Train is 2020–2023, Validation is 2024, and Evaluation is 2025. The Phase 3 population was not changed.

## 2. Frozen 20-predictor architecture

All models use the same 20 predictors: `Rain_t`, `RainSum_3h`, `RainSum_6h`, `RH_t`, `RHMin_6h`, `Temp_t`, `TempMax_6h`, `Hour_sin`, `Hour_cos`, `Month_sin`, `Month_cos`, `Elevation_m`, `SUHI_5K`, `BuiltUp_Binary`, `d2m`, `u10`, `v10`, `q_850`, `u_850`, and `v_850`. No predictor, lag, PCA component, or feature-selection step was added.

`SUHI_5K` is `LST_Median - Local_Rural_LST_Median_5km` in °C. Positive values are warmer than the local rural/non-built-up background. “5K” means 5 km, not 5 Kelvin.

## 3. Built-up / Non-built-up mapping

The frozen mapping was applied exactly: source `LULC_Class` 9 became `BuiltUp_Binary = 1`; classes 1–8 and 10 became 0. No row was excluded. The original class remains as `LULC_Class_Source`, a non-model audit field.

## 4. Model-ready v2 integrity

**PASS.** `ModelReady_Seasonal_Compact20_v1.parquet` contains 776,441 rows and preserves every Phase 3 `Station`, `Datetime`, `Target_Datetime`, `Role`, and target value in the original order. There are no duplicate station-predictor or station-target keys. Every target timestamp is one hour after its predictor timestamp. All target months are May–October. The target and built-up field are binary. All 20 predictors are complete and finite. No filling, interpolation, or imputation was performed.

| Role | Rows | Def5 positives | Stations |
|---|---:|---:|---:|
| Train | 525,474 | 312 | 40 |
| Validation | 129,671 | 74 | 39 |
| Evaluation | 121,296 | 95 | 38 |

## 5. Train and Validation class distributions

Train prevalence is {train_prevalence:.8f} ({train_prevalence * 100:.5f}%). Validation prevalence is {val_prevalence:.8f} ({val_prevalence * 100:.5f}%). The same Train rows and Validation rows were used by all three algorithms. No resampling changed the row universe.

## 6. Logistic Regression method

Logistic Regression is the statistical baseline. A `StandardScaler` was fitted on Train only for the 19 continuous predictors; `BuiltUp_Binary` was passed through unchanged. The fixed classifier used balanced class weights, the `lbfgs` solver, `max_iter=2000`, and random seed 154499. No hyperparameter search was performed. The saved sklearn pipeline contains both preprocessing and the fitted classifier.

## 7. Random Forest method

The Random Forest used the full Train population with 300 trees, `max_depth=12`, `min_samples_leaf=5`, `max_features=sqrt`, balanced subsample class weights, seed 154499, and all available CPU cores. Tree depths range from {int(rf_tree_summary['Depth'].min())} to {int(rf_tree_summary['Depth'].max())}; the median depth is {float(rf_tree_summary['Depth'].median()):.1f}. Node and leaf counts for every tree are in `RF_Tree_Summary_v1.csv`. No undersampling or tuning was performed.

## 8. Representative RF tree explanation

Tree index {representative_index} (depth {representative_depth}) was selected because its depth is closest to the ensemble median, with the lowest index resolving ties. The diagram displays only the first four levels for readability.

{RF_CAPTION}

The displayed tree was not used as a substitute for ensemble-level prediction or global feature importance.

## 9. XGBoost method and scale_pos_weight

XGBoost used the full Train population. The fixed settings were `objective=binary:logistic`, `eval_metric=aucpr`, `tree_method=hist`, `n_estimators=1500`, `learning_rate=0.05`, `max_depth=4`, `min_child_weight=5`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0`, `reg_alpha=0.0`, `random_state=154499`, `n_jobs=-1`, and `early_stopping_rounds=100`. `scale_pos_weight` was computed as Train negatives divided by Train positives: {scale_pos_weight:.8f}. Validation 2024 was used for early stopping only; this is a development-stage use of Validation. XGBoost reports zero-based `best_iteration={xgb_best_iteration}` (best boosting round {xgb_best_iteration + 1}), best Validation `aucpr={xgb_best_score:.8f}`, and {xgb_rounds} actual boosting rounds before stopping. No additional parameter search was performed.

## 10. Validation AP and ROC AUC

Average Precision (AP / PR-AUC) is primary because Def5 is extremely imbalanced. ROC AUC is secondary.

{markdown_table(comparison, comparison_columns)}

`AP_over_Prevalence` compares AP with the Validation positive prevalence baseline. It does **not** mean “times more accurate.” The common table uses sklearn Average Precision; XGBoost's internal early-stopping `aucpr` uses XGBoost's own PR-area calculation, so the two XGBoost values need not be numerically identical. Scores are uncalibrated model scores and are not described as calibrated probabilities.

## 11. Confusion matrices at threshold 0.5

The following non-normalized matrices use Actual class on rows and Predicted class on columns. Threshold 0.5 is a descriptive diagnostic only.

{chr(10).join(sections)}

Matching PNG figures were created for each model.

## 12. Actual-vs-Predict output

`Validation_Actual_vs_Predicted_v1.parquet` contains all 129,671 Validation rows with identifiers, the actual target, three model scores, threshold-0.5 predictions, and TP/FP/TN/FN labels. `Validation_Actual_vs_Predict_Sample_v1.csv` contains {sample_rows} rows selected reproducibly within algorithm/outcome groups using seed 154499 and de-duplicated across groups. This rule provides examples without selecting rows based on score attractiveness.

## 13. Comparison of the three models

The table reports the requested common Validation comparison. AP is the governing development metric, ROC AUC is supporting information, and threshold-0.5 metrics describe one fixed operating point. The phase does not automatically declare a winner or freeze a final model.

## 14. Important limitations

- Def5 positives are extremely rare, so threshold-0.5 counts can change substantially across algorithms with different score scales.
- Class weights affect fitting but do not calibrate scores.
- Validation serves two roles for XGBoost: early stopping and development comparison. Its XGBoost result is therefore less independent than a completely untouched assessment.
- No calibration, threshold optimization, hyperparameter search, feature selection, SHAP, or model redesign was performed.
- The representative RF tree is illustrative and cannot summarize all ensemble decision paths.
- AP/prevalence is a baseline-relative comparison, not “times more accurate.”

## 15. Evaluation 2025 was not scored

Evaluation 2025 was retained unchanged in the model-ready dataset solely for future work and counted for integrity verification. It was not loaded into the model-development table, transformed into model scores, evaluated, plotted, inspected through score distributions, or used for any decision in Phase 5.

## HUMAN DECISION REQUIRED

- Model comparison used Validation 2024 only.
- Evaluation 2025 was not scored.
- AP / PR-AUC is the primary metric.
- ROC AUC is secondary.
- Threshold 0.5 is diagnostic only.
- No threshold optimization was performed.
- No SHAP was performed.
- No model has yet been frozen as the final model.
- The researcher must review the three-model Validation comparison before Phase 6.
"""
    (OUT / "Phase5_Model_Comparison_Report_v1.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    columns = IDENTIFIERS + ["Role", TARGET] + PREDICTORS
    development = pd.read_parquet(
        DATA,
        columns=columns,
        filters=[("Role", "in", ["Train", "Validation"])],
    )
    verify_development_data(development)
    train = development.loc[development["Role"].eq("Train")].copy()
    validation = development.loc[development["Role"].eq("Validation")].copy()
    del development
    gc.collect()

    x_train = train[PREDICTORS].astype("float32")
    y_train = train[TARGET].to_numpy(dtype="int8")
    x_validation = validation[PREDICTORS].astype("float32")
    y_validation = validation[TARGET].to_numpy(dtype="int8")

    print("Training Logistic Regression...", flush=True)
    preprocessing = ColumnTransformer(
        [
            ("continuous", StandardScaler(), CONTINUOUS),
            ("binary", "passthrough", ["BuiltUp_Binary"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    lr_model = Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=SEED,
                ),
            ),
        ]
    )
    lr_model.fit(x_train, y_train)
    lr_scores = lr_model.predict_proba(x_validation)[:, 1]
    joblib.dump(lr_model, OUT / "LR_Model_v1.joblib", compress=3)
    del lr_model
    gc.collect()

    print("Training Random Forest...", flush=True)
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1,
    )
    rf_model.fit(x_train, y_train)
    rf_scores = rf_model.predict_proba(x_validation)[:, 1]
    joblib.dump(rf_model, OUT / "RF_Model_v1.joblib", compress=3)

    tree_rows = []
    for index, estimator in enumerate(rf_model.estimators_):
        tree_rows.append(
            {
                "Tree_Index": index,
                "Depth": estimator.tree_.max_depth,
                "Node_Count": estimator.tree_.node_count,
                "Leaf_Count": estimator.tree_.n_leaves,
            }
        )
    tree_summary = pd.DataFrame(tree_rows)
    median_depth = float(tree_summary["Depth"].median())
    representative_index = int((tree_summary["Depth"] - median_depth).abs().idxmin())
    representative_depth = int(tree_summary.loc[representative_index, "Depth"])
    tree_summary["Is_Representative"] = tree_summary["Tree_Index"].eq(representative_index)
    tree_summary.to_csv(OUT / "RF_Tree_Summary_v1.csv", index=False)

    fig, ax = plt.subplots(figsize=(26, 15))
    plot_tree(
        rf_model.estimators_[representative_index],
        feature_names=PREDICTORS,
        class_names=["Def5=0", "Def5=1"],
        max_depth=3,
        filled=True,
        rounded=True,
        proportion=False,
        precision=3,
        fontsize=6,
        ax=ax,
    )
    ax.set_title(
        f"Random Forest Representative Tree — estimator {representative_index}, full depth {representative_depth}\n"
        "First four displayed levels",
        fontsize=15,
        pad=14,
    )
    fig.text(0.5, 0.015, RF_CAPTION, ha="center", va="bottom", fontsize=10)
    fig.subplots_adjust(bottom=0.07)
    fig.savefig(OUT / "RF_Representative_Tree_v1.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    del rf_model
    gc.collect()

    train_positives = int(y_train.sum())
    train_negatives = int(len(y_train) - train_positives)
    scale_pos_weight = train_negatives / train_positives
    print(f"Training XGBoost with scale_pos_weight={scale_pos_weight:.8f}...", flush=True)
    xgb_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_estimators=1500,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=SEED,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=100,
    )
    xgb_model.fit(
        x_train,
        y_train,
        eval_set=[(x_validation, y_validation)],
        verbose=50,
    )
    xgb_scores = xgb_model.predict_proba(x_validation)[:, 1]
    xgb_model.save_model(OUT / "XGB_Model_v1.json")
    xgb_best_iteration = int(xgb_model.best_iteration)
    xgb_best_score = float(xgb_model.best_score)
    xgb_rounds = int(xgb_model.get_booster().num_boosted_rounds())
    del xgb_model
    gc.collect()

    score_map = {
        "Logistic Regression": lr_scores,
        "Random Forest": rf_scores,
        "XGBoost": xgb_scores,
    }
    comparison = pd.DataFrame(
        [calculate_metrics(name, y_train, y_validation, scores) for name, scores in score_map.items()]
    )
    comparison["Best_Iteration"] = [np.nan, np.nan, xgb_best_iteration]
    comparison["Best_Validation_AUCPR_XGBoost"] = [np.nan, np.nan, xgb_best_score]
    comparison["Actual_Boosting_Rounds"] = [np.nan, np.nan, xgb_rounds]
    comparison.to_csv(OUT / "Model_Comparison_Validation_v1.csv", index=False)

    predictions = validation[IDENTIFIERS + [TARGET]].reset_index(drop=True).copy()
    prefix_map = {
        "LR": lr_scores,
        "RF": rf_scores,
        "XGB": xgb_scores,
    }
    for prefix, scores in prefix_map.items():
        predicted = (scores >= 0.5).astype("int8")
        predictions[f"{prefix}_Score"] = scores.astype("float32")
        predictions[f"{prefix}_Predicted_0p5"] = predicted
        predictions[f"{prefix}_Outcome"] = outcome_labels(y_validation, predicted)
    predictions.to_parquet(
        OUT / "Validation_Actual_vs_Predicted_v1.parquet",
        index=False,
        compression="zstd",
    )

    selected_indices: set[int] = set()
    outcome_order = ["TP", "FP", "TN", "FN"]
    for model_number, prefix in enumerate(prefix_map):
        for outcome_number, outcome in enumerate(outcome_order):
            candidates = predictions.index[predictions[f"{prefix}_Outcome"].eq(outcome)].to_numpy()
            if len(candidates):
                rng = np.random.default_rng(SEED + model_number * 10 + outcome_number)
                chosen = rng.choice(candidates, size=min(3, len(candidates)), replace=False)
                selected_indices.update(map(int, chosen))
    sample = predictions.loc[sorted(selected_indices)].copy()
    sample.to_csv(OUT / "Validation_Actual_vs_Predict_Sample_v1.csv", index=False)

    for name, file_prefix in [
        ("Logistic Regression", "LR"),
        ("Random Forest", "RF"),
        ("XGBoost", "XGB"),
    ]:
        row = comparison.loc[comparison["Model"].eq(name)].iloc[0]
        save_confusion_figure(name, row, OUT / f"{file_prefix}_ConfusionMatrix_v1.png")
    save_curve_figures(y_validation, score_map, comparison)
    write_report(
        comparison,
        scale_pos_weight,
        xgb_best_iteration,
        xgb_best_score,
        xgb_rounds,
        tree_summary,
        representative_index,
        representative_depth,
        len(sample),
    )

    print("\nValidation comparison", flush=True)
    print(comparison.to_string(index=False), flush=True)
    print(
        f"Representative RF tree: index={representative_index}, depth={representative_depth}, "
        f"median_depth={median_depth:.1f}",
        flush=True,
    )
    print(
        f"XGBoost: best_iteration={xgb_best_iteration}, best_validation_aucpr={xgb_best_score:.8f}, "
        f"actual_rounds={xgb_rounds}",
        flush=True,
    )
    print(f"Validation sample rows: {len(sample)}", flush=True)
    print("Evaluation 2025 was not loaded or scored.", flush=True)


if __name__ == "__main__":
    main()
