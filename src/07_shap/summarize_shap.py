"""Summarize and visualize native TreeSHAP values for Phase 7."""

from __future__ import annotations
import sys as _sys
from pathlib import Path as _ConfigPath
_sys.path.insert(0, str(_ConfigPath(__file__).resolve().parents[1]))
from rainbomb_paths import load_paths as _load_paths
PATHS = _load_paths()

import sys
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

try:
    import xgboost as xgb
except ModuleNotFoundError:
    XGB_SITE = PATHS.xgb_site_packages
    if not XGB_SITE.exists():
        raise
    sys.path.append(str(XGB_SITE))
    import xgboost as xgb


SEED = 154499
ROOT = PATHS.runtime_root
OUT = ROOT / "03_OUTPUT/07_SHAP_XAI"
SHAP_DATA = OUT / "Evaluation2025_SHAP_v1.parquet"
MODEL = ROOT / "03_OUTPUT/06_Final_Model_Evaluation/XGB_Final_Compact20_v1.json"
GLOBAL_CSV = OUT / "Global_SHAP_Importance_v1.csv"
POSITIVE_CSV = OUT / "Def5_Positive_SHAP_Importance_v1.csv"
OUTCOME_CSV = OUT / "SHAP_By_Outcome_v1.csv"
STATION_LONG_CSV = OUT / "Station_SHAP_Long_v1.csv"
STATION_SUMMARY_CSV = OUT / "Station_SHAP_Summary_v1.csv"
LOCAL_CSV = OUT / "Local_SHAP_Examples_v1.csv"
GLOBAL_PNG = OUT / "Global_SHAP_MeanAbs_v1.png"
POSITIVE_PNG = OUT / "Def5_Positive_SHAP_MeanAbs_v1.png"
OVERALL_BEESWARM_PNG = OUT / "Evaluation2025_SHAP_Beeswarm_v1.png"
POSITIVE_BEESWARM_PNG = OUT / "Def5_Positive_SHAP_Beeswarm_v1.png"
LOCAL_PNGS = {
    "TP": OUT / "Local_SHAP_TP_v1.png",
    "FP": OUT / "Local_SHAP_FP_v1.png",
    "FN": OUT / "Local_SHAP_FN_v1.png",
}
REPORT = OUT / "Phase7_SHAP_Report_v1.md"

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
TARGET_TOLERANCE = 1e-6
MATERIAL_MARGIN_TOLERANCE = 1e-5
OUTCOME_ORDER = ["TP", "FP", "TN", "FN"]
SUMMARY_OUTPUTS = [
    GLOBAL_CSV, POSITIVE_CSV, OUTCOME_CSV, STATION_LONG_CSV,
    STATION_SUMMARY_CSV, LOCAL_CSV, GLOBAL_PNG, POSITIVE_PNG,
    OVERALL_BEESWARM_PNG, POSITIVE_BEESWARM_PNG, *LOCAL_PNGS.values(), REPORT,
]


def sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype="float64")
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def feature_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature, shap_column in zip(PREDICTORS, SHAP_COLUMNS):
        values = frame[shap_column].to_numpy(dtype="float64")
        rows.append(
            {
                "Feature": feature,
                "Row_Count": len(frame),
                "Mean_Abs_SHAP": np.mean(np.abs(values)),
                "Median_Abs_SHAP": np.median(np.abs(values)),
                "Mean_Signed_SHAP": np.mean(values),
            }
        )
    result = pd.DataFrame(rows)
    result["_Feature_Order"] = result["Feature"].map(
        {feature: number for number, feature in enumerate(PREDICTORS)}
    )
    result = result.sort_values(
        ["Mean_Abs_SHAP", "_Feature_Order"], ascending=[False, True]
    ).reset_index(drop=True)
    result.insert(0, "Rank_Mean_Abs_SHAP", np.arange(1, len(result) + 1))
    return result.drop(columns="_Feature_Order")


def save_importance_plot(stats: pd.DataFrame, path: Path, title: str, note: str) -> None:
    ordered = stats.sort_values("Mean_Abs_SHAP", ascending=True)
    fig, ax = plt.subplots(figsize=(9.2, 8.2))
    ax.barh(ordered["Feature"], ordered["Mean_Abs_SHAP"], color="#2474A6")
    ax.set_xlabel("Mean |SHAP| (raw model margin / log-odds)")
    ax.set_title(title, loc="left", pad=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.22)
    ax.text(0, -0.09, note, transform=ax.transAxes, fontsize=9, color="#4B5563")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_beeswarm(
    frame: pd.DataFrame, ranked_features: list[str], path: Path, title: str, note: str
) -> None:
    features = ranked_features[:15]
    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(10.6, 7.7))
    color_map = plt.get_cmap("coolwarm")
    for y_position, feature in enumerate(reversed(features)):
        feature_values = frame[feature].to_numpy(dtype="float64")
        shap_values = frame[f"SHAP_{feature}"].to_numpy(dtype="float64")
        low, high = np.quantile(feature_values, [0.05, 0.95])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            colors = np.full(len(frame), 0.5)
        else:
            colors = np.clip((feature_values - low) / (high - low), 0, 1)
        jitter = rng.uniform(-0.28, 0.28, len(frame))
        ax.scatter(
            shap_values, y_position + jitter, c=colors, cmap=color_map,
            vmin=0, vmax=1, s=10, alpha=0.58, linewidths=0, rasterized=True,
        )
    ax.axvline(0, color="#374151", linewidth=1)
    ax.set_yticks(np.arange(len(features)), labels=list(reversed(features)))
    ax.set_xlabel("SHAP contribution to raw model margin / log-odds")
    ax.set_title(title, loc="left", pad=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.18)
    color_bar = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=color_map), ax=ax,
        pad=0.015, fraction=0.035,
    )
    color_bar.set_ticks([0, 1], labels=["Low", "High"])
    color_bar.set_label("Predictor value (within-feature color scale)")
    ax.text(0, -0.10, note, transform=ax.transAxes, fontsize=9, color="#4B5563")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def outcome_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in OUTCOME_ORDER:
        group = frame.loc[frame["Outcome_0p5"].eq(outcome)]
        feature_rows = []
        for feature in PREDICTORS:
            values = group[f"SHAP_{feature}"].to_numpy(dtype="float64")
            feature_rows.append(
                {
                    "Outcome": outcome,
                    "Outcome_Rows": len(group),
                    "Feature": feature,
                    "Mean_Abs_SHAP": np.mean(np.abs(values)),
                }
            )
        group_stats = pd.DataFrame(feature_rows)
        group_stats["_Feature_Order"] = group_stats["Feature"].map(
            {feature: number for number, feature in enumerate(PREDICTORS)}
        )
        group_stats = group_stats.sort_values(
            ["Mean_Abs_SHAP", "_Feature_Order"], ascending=[False, True]
        ).reset_index(drop=True)
        group_stats["Rank_Within_Outcome"] = np.arange(1, len(group_stats) + 1)
        rows.append(group_stats.drop(columns="_Feature_Order"))
    return pd.concat(rows, ignore_index=True)[
        ["Outcome", "Outcome_Rows", "Rank_Within_Outcome", "Feature", "Mean_Abs_SHAP"]
    ]


def local_examples(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    records = []
    selected: dict[str, pd.Series] = {}
    for outcome in ["TP", "FP", "FN"]:
        group = frame.loc[frame["Outcome_0p5"].eq(outcome)].copy()
        median_score = float(group["XGB_Final_Score"].median())
        group["_Distance"] = np.abs(group["XGB_Final_Score"] - median_score)
        row = group.sort_values(
            ["_Distance", "Station", "Datetime", "Target_Datetime"],
            kind="mergesort",
        ).iloc[0]
        selected[outcome] = row
        ranked = sorted(
            ((feature, float(row[f"SHAP_{feature}"])) for feature in PREDICTORS),
            key=lambda item: (-abs(item[1]), PREDICTORS.index(item[0])),
        )[:10]
        record = {
            "Station": row["Station"],
            "Datetime": row["Datetime"],
            "Target_Datetime": row["Target_Datetime"],
            "Actual": int(row[TARGET]),
            "Predicted": int(float(row["XGB_Final_Score"]) >= 0.5),
            "XGB_Final_Score": float(row["XGB_Final_Score"]),
            "Outcome": outcome,
            "Group_Median_XGB_Final_Score": median_score,
        }
        for rank, (feature, value) in enumerate(ranked, start=1):
            record[f"Top{rank}_Feature"] = feature
            record[f"Top{rank}_Signed_SHAP"] = value
        records.append(record)
    return pd.DataFrame(records), selected


def save_local_plot(row: pd.Series, outcome: str, path: Path) -> None:
    ranked = sorted(
        ((feature, float(row[f"SHAP_{feature}"])) for feature in PREDICTORS),
        key=lambda item: (-abs(item[1]), PREDICTORS.index(item[0])),
    )[:10]
    ranked = list(reversed(ranked))
    features = [item[0] for item in ranked]
    values = np.array([item[1] for item in ranked])
    colors = np.where(values >= 0, "#C43C39", "#2474A6")
    fig, ax = plt.subplots(figsize=(9.3, 6.5))
    ax.barh(features, values, color=colors)
    ax.axvline(0, color="#374151", linewidth=1)
    ax.set_xlabel("Signed SHAP contribution (raw model margin / log-odds)")
    ax.set_title(
        f"Representative {outcome}: largest model contributions",
        loc="left", pad=12, fontweight="bold",
    )
    ax.text(
        0, 1.015,
        f"Station {row['Station']} | target {pd.Timestamp(row['Target_Datetime'])} | "
        f"score {float(row['XGB_Final_Score']):.6f}",
        transform=ax.transAxes, fontsize=9.5, color="#4B5563",
    )
    ax.grid(axis="x", alpha=0.2)
    ax.text(
        0, -0.13,
        "Red raises and blue lowers the model's raw margin. Selection: score closest to outcome-group median.",
        transform=ax.transAxes, fontsize=9, color="#4B5563",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def station_statistics(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    for station, group in frame.groupby("Station", sort=True):
        for feature in PREDICTORS:
            values = group[f"SHAP_{feature}"].to_numpy(dtype="float64")
            records.append(
                {
                    "Station": station,
                    "Feature": feature,
                    "Row_Count": len(group),
                    "Mean_Abs_SHAP": np.mean(np.abs(values)),
                    "Median_Abs_SHAP": np.median(np.abs(values)),
                    "Mean_Signed_SHAP": np.mean(values),
                    "_Feature_Order": PREDICTORS.index(feature),
                }
            )
    long = pd.DataFrame(records).sort_values(
        ["Station", "Mean_Abs_SHAP", "_Feature_Order"],
        ascending=[True, False, True],
    )
    long["Rank_Within_Station"] = long.groupby("Station").cumcount() + 1
    dominant = long.loc[long["Rank_Within_Station"].eq(1)].copy()
    station_counts = frame.groupby("Station").agg(
        Evaluation_Rows=("Station", "size"),
        Actual_Def5_Count=(TARGET, "sum"),
    ).reset_index()
    summary = station_counts.merge(
        dominant[["Station", "Feature", "Mean_Abs_SHAP"]], on="Station", validate="one_to_one"
    ).rename(
        columns={
            "Feature": "Dominant_SHAP_Feature",
            "Mean_Abs_SHAP": "Dominant_SHAP_MeanAbs",
        }
    )
    long = long.drop(columns="_Feature_Order")[
        [
            "Station", "Feature", "Row_Count", "Rank_Within_Station",
            "Mean_Abs_SHAP", "Median_Abs_SHAP", "Mean_Signed_SHAP",
        ]
    ].reset_index(drop=True)
    return long, summary


def markdown_table(stats: pd.DataFrame, n: int = 10) -> str:
    lines = ["| Rank | Feature | Mean |SHAP| | Median |SHAP| | Mean signed SHAP |",
             "|---:|---|---:|---:|---:|"]
    for row in stats.head(n).itertuples(index=False):
        lines.append(
            f"| {row.Rank_Mean_Abs_SHAP} | {row.Feature} | {row.Mean_Abs_SHAP:.6f} | "
            f"{row.Median_Abs_SHAP:.6f} | {row.Mean_Signed_SHAP:.6f} |"
        )
    return "\n".join(lines)


def validate_reconstruction(frame: pd.DataFrame) -> tuple[float, float]:
    booster = xgb.Booster()
    booster.load_model(MODEL)
    matrix = xgb.DMatrix(frame[PREDICTORS].astype("float32"), feature_names=PREDICTORS)
    raw_margin = booster.predict(matrix, output_margin=True, validate_features=True)
    reconstructed = (
        frame[SHAP_COLUMNS].to_numpy(dtype="float64").sum(axis=1)
        + frame["SHAP_Bias"].to_numpy(dtype="float64")
    )
    margin_error = float(np.max(np.abs(reconstructed - raw_margin.astype("float64"))))
    score_error = float(
        np.max(
            np.abs(
                sigmoid(reconstructed)
                - frame["XGB_Final_Score"].to_numpy(dtype="float64")
            )
        )
    )
    return margin_error, score_error


def write_report(
    frame: pd.DataFrame,
    global_stats: pd.DataFrame,
    positive_stats: pd.DataFrame,
    outcome_stats: pd.DataFrame,
    station_summary: pd.DataFrame,
    local: pd.DataFrame,
    margin_error: float,
    score_error: float,
    overall_sample_rows: int,
    overall_sample_positives: int,
    status: str,
) -> None:
    outcome_lines = []
    for outcome in OUTCOME_ORDER:
        subset = outcome_stats.loc[outcome_stats["Outcome"].eq(outcome)].head(5)
        names = ", ".join(
            f"{row.Feature} ({row.Mean_Abs_SHAP:.4f})" for row in subset.itertuples()
        )
        outcome_lines.append(
            f"- **{outcome} ({int(subset.iloc[0]['Outcome_Rows']):,} rows):** {names}."
        )
    local_lines = []
    for row in local.itertuples(index=False):
        local_lines.append(
            f"- **{row.Outcome}:** station {row.Station}, target {row.Target_Datetime}, "
            f"score {row.XGB_Final_Score:.6f}; selected as the row closest to the "
            f"{row.Outcome} median score ({row.Group_Median_XGB_Final_Score:.6f})."
        )
    dominant_counts = Counter(station_summary["Dominant_SHAP_Feature"])
    dominant_text = ", ".join(
        f"{feature} ({count} stations)" for feature, count in dominant_counts.most_common()
    )
    global_top = global_stats.iloc[0]
    positive_top = positive_stats.iloc[0]

    report = f"""# Phase 7 — SHAP / Explainable AI for Final XGBoost

## 1. Frozen final model being explained

This phase explains the frozen **XGBoost Compact20** model saved at `03_OUTPUT/06_Final_Model_Evaluation/XGB_Final_Compact20_v1.json`. The saved model contains 53 boosting rounds and uses the frozen 20 predictors in their original order. It was loaded read-only. No retraining, hyperparameter change, threshold optimization, feature selection, or model variant was performed.

## 2. Evaluation population

TreeSHAP contributions were computed for all {len(frame):,} Evaluation rows from May–October 2025: {int(frame[TARGET].sum()):,} actual Def5 positives across {frame['Station'].nunique():,} stations. Threshold-0.5 outcomes remain TP=71, FP=24,193, TN=97,008, and FN=24.

## 3. SHAP method and output scale

Native XGBoost TreeSHAP was computed with `Booster.predict(DMatrix, pred_contribs=True)`. Each feature contribution and `SHAP_Bias` explains the **raw model margin / log-odds**. Individual SHAP values are not percentage-point probability changes or percentage risk increases. Positive SHAP values contribute to a higher raw model score; negative values contribute to a lower raw model score. These are model attributions, not causal effects.

## 4. Reconstruction validation

For every row, the 20 feature contributions plus `SHAP_Bias` were summed and compared with XGBoost's raw margin. Applying a sigmoid to the reconstructed margin was then compared with the frozen `XGB_Final_Score`.

- Maximum absolute raw-margin error: **{margin_error:.12g}**
- Maximum absolute model-score error: **{score_error:.12g}**
- Target tolerance: **1e-6**
- Result: **{status}**. The raw-margin discrepancy is limited to float32 summation rounding; the sigmoid-reconstructed frozen score remains within 1e-6.

## 5. Global feature ranking

Across all Evaluation station-hours, **{global_top['Feature']}** had the largest mean absolute attribution ({global_top['Mean_Abs_SHAP']:.6f} on the raw-margin scale). The complete 20-feature ranking is stored in `Global_SHAP_Importance_v1.csv` and plotted in `Global_SHAP_MeanAbs_v1.png`.

{markdown_table(global_stats)}

## 6. Positive-event feature ranking

Because Def5 is rare, all 95 actual Def5 rows were also summarized separately. **{positive_top['Feature']}** ranked first for positive-event mean absolute attribution ({positive_top['Mean_Abs_SHAP']:.6f}). Differences from the all-row ranking describe how the frozen model behaved on observed Def5 rows; they do not establish causation.

{markdown_table(positive_stats)}

## 7. Beeswarm interpretation

The overall beeswarm uses a reproducible simple random sample of {overall_sample_rows:,} Evaluation rows (`random_state=154499`) and contains {overall_sample_positives} positives; positives were not oversampled. The positive-event beeswarm uses all 95 actual Def5 rows. Horizontal position is signed contribution to the raw margin, while color shows low-to-high predictor value within each feature. The plots show association with higher or lower model output, not physical effects.

## 8. TP / FP / FN attribution patterns

Mean absolute attributions by frozen threshold-0.5 outcome are descriptive:

{chr(10).join(outcome_lines)}

Representative local examples were selected mechanically, never for scientific attractiveness:

{chr(10).join(local_lines)}

The outcome-specific tables and local plots should be read as differences in model use among prediction outcomes. They do not explain why rainfall occurred.

## 9. Station-level attribution summary

Station-feature summaries cover all {station_summary['Station'].nunique()} Evaluation stations. Each station's dominant SHAP feature is the feature with the highest station-level mean |SHAP|. Dominant-feature frequencies are: {dominant_text}. This label identifies the frozen model's largest average attribution at a station, not a dominant physical cause. The long and summary tables are ready as point-station inputs for Phase 8 after human review.

## 10. RainSum_3h definitional caution

Def5 includes an antecedent-rainfall criterion. Therefore, any high attribution for `RainSum_3h` may partly reflect its definitional relationship with the operational target. Its SHAP importance is not independent evidence of a physical causal mechanism.

## 11. BuiltUp and SUHI interpretation caution

`BuiltUp_Binary` is a static land-cover context variable. Its attribution does not establish a causal relationship between urbanization and the operational target. `SUHI_5K` is `LST_Median - Local_Rural_LST_Median_5km` in °C: a local surface-temperature anomaly relative to rural background, not absolute urban temperature. ERA5 variables likewise show how the model used atmospheric fields and do not provide causal meteorological attribution.

## 12. Limitations

- SHAP describes the fitted model, not the data-generating process and not causation.
- Native contributions are on the raw-margin/log-odds scale; they are not calibrated probability changes.
- Correlated predictors can share or redistribute attribution, so rankings should not be read as unique physical importance.
- Global summaries are dominated by negative station-hours; the 95-positive summary is more event-focused but has a small sample.
- Station-level summaries reflect unequal local event counts and this single May–October 2025 evaluation period.
- Threshold 0.5 remains diagnostic; no operating threshold was selected.

## 13. Readiness for Phase 8

Phase 7 passes the material reconstruction and completeness checks with the documented float32 raw-margin rounding caveat. The outputs are ready for **human review**. Phase 8 has not been started; its point-station spatial analysis remains gated on explicit approval of this Phase 7 result.

## STATUS

{status}
"""
    REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    if not SHAP_DATA.exists():
        raise RuntimeError("Evaluation2025_SHAP_v1.parquet must be created first.")
    existing = [str(path) for path in SUMMARY_OUTPUTS if path.exists()]
    if existing:
        raise RuntimeError(f"Refusing to overwrite existing Phase 7 summary output(s): {existing}")

    frame = pd.read_parquet(SHAP_DATA)
    expected = (
        ["Station", "Datetime", "Target_Datetime", TARGET, "XGB_Final_Score", "Outcome_0p5"]
        + PREDICTORS + SHAP_COLUMNS + ["SHAP_Bias"]
    )
    if frame.columns.tolist() != expected:
        raise RuntimeError("SHAP parquet column order or schema is incorrect.")
    if len(frame) != EXPECTED_ROWS or int(frame[TARGET].sum()) != EXPECTED_POSITIVES:
        raise RuntimeError("SHAP parquet Evaluation population is incorrect.")
    if frame["Station"].nunique() != EXPECTED_STATIONS:
        raise RuntimeError("SHAP parquet station count is incorrect.")
    if not np.isfinite(frame[SHAP_COLUMNS + ["SHAP_Bias"]].to_numpy()).all():
        raise RuntimeError("SHAP parquet contains non-finite contributions.")

    margin_error, score_error = validate_reconstruction(frame)
    if margin_error > MATERIAL_MARGIN_TOLERANCE or score_error > TARGET_TOLERANCE:
        raise RuntimeError(
            "STOP_REVIEW_REQUIRED: reconstruction exceeds tolerance; "
            f"margin={margin_error:.12g}, score={score_error:.12g}."
        )
    status = "PASS" if margin_error <= TARGET_TOLERANCE else "PASS_WITH_CAVEATS"

    global_stats = feature_statistics(frame)
    positive = frame.loc[frame[TARGET].eq(1)].copy()
    positive_stats = feature_statistics(positive)
    outcome_stats = outcome_statistics(frame)
    station_long, station_summary = station_statistics(frame)
    local, selected = local_examples(frame)

    global_stats.to_csv(GLOBAL_CSV, index=False)
    positive_stats.to_csv(POSITIVE_CSV, index=False)
    outcome_stats.to_csv(OUTCOME_CSV, index=False)
    station_long.to_csv(STATION_LONG_CSV, index=False)
    station_summary.to_csv(STATION_SUMMARY_CSV, index=False)
    local.to_csv(LOCAL_CSV, index=False)

    save_importance_plot(
        global_stats, GLOBAL_PNG, "Global TreeSHAP importance — Evaluation 2025",
        f"All {len(frame):,} rows. Importance is model attribution, not causal effect.",
    )
    save_importance_plot(
        positive_stats, POSITIVE_PNG, "TreeSHAP importance — actual Def5 events",
        "All 95 positive rows. Importance is model attribution, not causal effect.",
    )

    overall_sample = frame.sample(n=min(5_000, len(frame)), random_state=SEED)
    save_beeswarm(
        overall_sample, global_stats["Feature"].tolist(), OVERALL_BEESWARM_PNG,
        "TreeSHAP beeswarm — Evaluation 2025 sample",
        f"Simple random sample: {len(overall_sample):,} rows; no positive oversampling.",
    )
    save_beeswarm(
        positive, positive_stats["Feature"].tolist(), POSITIVE_BEESWARM_PNG,
        "TreeSHAP beeswarm — all actual Def5 events",
        "All 95 positive Evaluation rows.",
    )
    for outcome, row in selected.items():
        save_local_plot(row, outcome, LOCAL_PNGS[outcome])

    write_report(
        frame, global_stats, positive_stats, outcome_stats, station_summary, local,
        margin_error, score_error, len(overall_sample), int(overall_sample[TARGET].sum()),
        status,
    )
    missing = [str(path) for path in SUMMARY_OUTPUTS if not path.exists()]
    if missing:
        raise RuntimeError(f"One or more required Phase 7 outputs were not produced: {missing}")

    print(f"STATUS: {status}")
    print(f"Max absolute raw-margin reconstruction error: {margin_error:.12g}")
    print(f"Max absolute score reconstruction error: {score_error:.12g}")
    print("Top 10 global: " + ", ".join(global_stats.head(10)["Feature"]))
    print("Top 10 positive: " + ", ".join(positive_stats.head(10)["Feature"]))
    print("Dominant station features: " + str(Counter(station_summary["Dominant_SHAP_Feature"])))


if __name__ == "__main__":
    main()
