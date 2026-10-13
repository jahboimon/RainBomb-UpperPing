"""Isolated Annual-P99 target and Compact20 XGBoost sensitivity experiment.

All canonical and frozen inputs are read-only. Every created artifact is written
below this script's directory. Annual P99 is retrospective (uses the full target
year); the rolling comparator uses only wet hours strictly before the target year.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


HERE = Path(__file__).resolve().parent

CONFIG_PATH = HERE.parent / "config.json"
if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"Missing local config: {CONFIG_PATH}. "
        "Copy config.example.json to config.json and set local paths."
    )

with CONFIG_PATH.open("r", encoding="utf-8-sig") as f:
    CONFIG = json.load(f)

CANON = Path(CONFIG["canonical_root"])
FINAL = Path(CONFIG["final_analysis_root"])
RAIN_PATH = CANON / "Rainfall_Canonical_2012_2025_Final_v1" / "PROCESS_Hourly_Rain_Canonical_2012_2025_v1.parquet"
FROZEN_P99_PATH = CANON / "RainBomb_Def4_2012_2025_v1" / "Def4_Frozen_P99_2012_2024H1_v1.csv"
FROZEN_EVENTS_PATH = CANON / "Def5_Final_Target_2012_2025_v1" / "02_Def5_Extended_Target" / "RainBomb_Target_Def5_2012_2025_v1.parquet"
COMPACT_PATH = FINAL / "03_OUTPUT" / "05_Model_Comparison" / "ModelReady_Seasonal_Compact20_v1.parquet"
BASELINE_METRICS_PATH = FINAL / "03_OUTPUT" / "06_Final_Model_Evaluation" / "Evaluation2025_Metrics_v1.csv"
BASELINE_MODEL_PATH = FINAL / "03_OUTPUT" / "06_Final_Model_Evaluation" / "XGB_Final_Compact20_v1.json"

TARGET = "Target_AnnualP99_tPlus1"
PREDICTORS = [
    "Rain_t", "RainSum_3h", "RainSum_6h", "RH_t", "RHMin_6h", "Temp_t",
    "TempMax_6h", "Hour_sin", "Hour_cos", "Month_sin", "Month_cos",
    "Elevation_m", "SUHI_5K", "BuiltUp_Binary", "d2m", "u10", "v10",
    "q_850", "u_850", "v_850",
]
SEED = 154499
MIN_ROLLING_WET_HOURS = 100


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(HERE / name, index=False)


def metric_row(label: str, y: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (score >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prevalence = float(y.mean())
    return {
        "Experiment": label, "Rows": len(y), "Positives": int(y.sum()),
        "Negatives": int(len(y) - y.sum()), "Prevalence": prevalence,
        "AP": average_precision_score(y, score), "ROC_AUC": roc_auc_score(y, score),
        "AP_over_Prevalence": average_precision_score(y, score) / prevalence,
        "Threshold": threshold, "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "Specificity": float(tn / (tn + fp)),
        "F1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else math.nan,
    }


def build_thresholds(rain: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    wet = rain.loc[rain["Rainfall"].gt(0) & rain["Rainfall"].notna(), ["Station", "Datetime", "Rainfall"]].copy()
    wet["Year"] = wet["Datetime"].dt.year
    annual = wet.groupby(["Station", "Year"], sort=True)["Rainfall"].agg(
        Wet_Hours="size", Annual_P99=lambda x: x.quantile(0.99, interpolation="linear")
    ).reset_index()
    annual["Low_Sample_LT100"] = annual["Wet_Hours"].lt(100)
    frozen = pd.read_csv(FROZEN_P99_PATH)[["Station", "P99_Frozen"]]
    annual = annual.merge(frozen, on="Station", how="left", validate="many_to_one")
    annual["Annual_minus_Frozen_mm"] = annual["Annual_P99"] - annual["P99_Frozen"]
    annual["Annual_to_Frozen_Ratio"] = annual["Annual_P99"] / annual["P99_Frozen"]

    rows = []
    for station, part in wet.groupby("Station", sort=True):
        for year in sorted(rain.loc[rain["Station"].eq(station), "Datetime"].dt.year.unique()):
            prior = part.loc[part["Year"].lt(year), "Rainfall"]
            rows.append({
                "Station": station, "Year": int(year), "Prior_Wet_Hours": int(len(prior)),
                "Rolling_P99": float(prior.quantile(0.99, interpolation="linear")) if len(prior) >= MIN_ROLLING_WET_HOURS else np.nan,
                "Minimum_Prior_Wet_Hours": MIN_ROLLING_WET_HOURS,
                "Operationally_Available": bool(len(prior) >= MIN_ROLLING_WET_HOURS),
            })
    rolling = pd.DataFrame(rows)
    return annual, rolling


def event_mask(rain: pd.DataFrame, threshold: pd.Series) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    station = rain["Station"]
    dt = rain["Datetime"]
    val = rain["Rainfall"].to_numpy(float)
    shifted, continuous = {}, np.ones(len(rain), dtype=bool)
    for k in (-3, -2, -1, 1, 2, 3):
        shifted[k] = rain.groupby("Station", sort=False)["Rainfall"].shift(-k).to_numpy(float)
        shifted_dt = rain.groupby("Station", sort=False)["Datetime"].shift(-k)
        continuous &= shifted_dt.eq(dt + pd.Timedelta(hours=k)).to_numpy()
    finite = np.isfinite(val) & np.isfinite(threshold.to_numpy(float))
    for k in shifted:
        finite &= np.isfinite(shifted[k])
    ante = shifted[-3] + shifted[-2] + shifted[-1]
    postmax = np.maximum.reduce([shifted[1], shifted[2], shifted[3]])
    mask = continuous & finite & (val >= threshold.to_numpy(float)) & (val >= 20) & (ante <= 5) & (postmax < 20)
    return mask, {"ante": ante, "postmax": postmax, **{f"rain_{k:+d}": v for k, v in shifted.items()}}


def event_table(rain: pd.DataFrame, mask: np.ndarray, threshold: pd.Series, aux: dict, name: str) -> pd.DataFrame:
    out = rain.loc[mask, ["Station", "Datetime", "Rainfall"]].copy()
    out = out.rename(columns={"Datetime": "Event_Datetime", "Rainfall": "Rain_E_mm"})
    out["Year"] = out["Event_Datetime"].dt.year
    out["Month"] = out["Event_Datetime"].dt.month
    out["Hour"] = out["Event_Datetime"].dt.hour
    out["Threshold_mm"] = threshold.loc[mask].to_numpy()
    out["Antecedent_3h_Sum_mm"] = aux["ante"][mask]
    out["Post_3h_Max_mm"] = aux["postmax"][mask]
    out["Definition"] = name
    return out


def keyset(frame: pd.DataFrame) -> set[tuple[str, pd.Timestamp]]:
    return set(zip(frame["Station"], frame["Event_Datetime"]))


def comparison_row(name: str, events: pd.DataFrame, frozen: pd.DataFrame) -> dict:
    a, f = keyset(events), keyset(frozen)
    inter, union = a & f, a | f
    return {"Definition": name, "Events": len(a), "Overlap_with_Frozen": len(inter),
            "Added_vs_Frozen": len(a - f), "Removed_vs_Frozen": len(f - a),
            "Jaccard_vs_Frozen": len(inter) / len(union) if union else math.nan}


def count_tables(events: pd.DataFrame, prefix: str) -> None:
    for field, label in [("Year", "year"), ("Station", "station"), ("Month", "month"), ("Hour", "hour")]:
        out = events.groupby(field).size().rename("Event_Count").reset_index()
        write_csv(out, f"{prefix}_count_by_{label}.csv")


def hydrographs(rain: pd.DataFrame, annual: pd.DataFrame, frozen: pd.DataFrame) -> pd.DataFrame:
    ak, fk = keyset(annual), keyset(frozen)
    groups = {"Annual_Added": ak - fk, "Annual_Removed": fk - ak, "Overlap": ak & fk}
    lookup = rain.set_index(["Station", "Datetime"])["Rainfall"]
    rows = []
    for group, keys in groups.items():
        candidates = []
        for station, dt in sorted(keys):
            value = lookup.get((station, dt), np.nan)
            if np.isfinite(value):
                candidates.append((station, dt, value))
        if not candidates:
            continue
        vals = np.array([x[2] for x in candidates])
        chosen = [candidates[i] for i in np.argsort(vals)[np.linspace(0, len(vals)-1, min(3, len(vals))).astype(int)]]
        for station, dt, _ in chosen:
            for offset in range(-3, 4):
                rows.append({"Group": group, "Station": station, "Event_Datetime": dt,
                             "Offset_h": offset, "Rainfall_mm": lookup.get((station, dt + pd.Timedelta(hours=offset)), np.nan)})
    return pd.DataFrame(rows)


def save_figures(annual_thresholds: pd.DataFrame, comparisons: pd.DataFrame, hydro: pd.DataFrame,
                 y_eval: np.ndarray, score: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(annual_thresholds["Year"], annual_thresholds["Annual_to_Frozen_Ratio"], s=10, alpha=.35)
    ax.axhline(1, color="black", ls="--", lw=1)
    ax.set(xlabel="Year", ylabel="Annual P99 / frozen P99", title="Annual station wet-hour P99 relative to frozen P99")
    fig.tight_layout(); fig.savefig(HERE / "annual_vs_frozen_p99.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(comparisons["Definition"], comparisons["Events"], color=["#4c78a8", "#f58518", "#54a24b"])
    ax.set(ylabel="Events", title="Event counts by threshold definition")
    ax.tick_params(axis="x", rotation=18); fig.tight_layout(); fig.savefig(HERE / "event_count_comparison.png", dpi=180); plt.close(fig)

    if not hydro.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for (group, station, dt), p in hydro.groupby(["Group", "Station", "Event_Datetime"]):
            ax.plot(p["Offset_h"], p["Rainfall_mm"], marker="o", alpha=.7, label=f"{group}: {station} {dt:%Y-%m-%d %Hh}")
        ax.set(xlabel="Hours relative to E", ylabel="Rainfall (mm/h)", title="Representative 7-hour hydrographs")
        ax.legend(fontsize=6, ncol=2); fig.tight_layout(); fig.savefig(HERE / "representative_hydrographs.png", dpi=180); plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_eval, score)
    fig, ax = plt.subplots(figsize=(6, 5)); ax.plot(recall, precision)
    ax.axhline(y_eval.mean(), color="gray", ls="--", label="Prevalence")
    ax.set(xlabel="Recall", ylabel="Precision", title="Annual-P99 XGBoost: Evaluation 2025 PR curve"); ax.legend()
    fig.tight_layout(); fig.savefig(HERE / "annual_p99_model_pr_curve_2025.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(score[y_eval == 0], bins=60, alpha=.65, density=True, label="Negative")
    ax.hist(score[y_eval == 1], bins=30, alpha=.65, density=True, label="Positive")
    ax.set_yscale("log"); ax.set(xlabel="Model score", ylabel="Density (log)", title="Annual-P99 XGBoost score distribution: Evaluation 2025")
    ax.legend(); fig.tight_layout(); fig.savefig(HERE / "annual_p99_model_score_distribution_2025.png", dpi=180); plt.close(fig)


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    protected = [RAIN_PATH, FROZEN_P99_PATH, FROZEN_EVENTS_PATH, COMPACT_PATH, BASELINE_METRICS_PATH, BASELINE_MODEL_PATH]
    hashes_before = {str(p): sha256(p) for p in protected}

    rain = pd.read_parquet(RAIN_PATH, columns=["Station", "Datetime", "Rainfall"])
    rain["Datetime"] = pd.to_datetime(rain["Datetime"])
    rain = rain.sort_values(["Station", "Datetime"], kind="mergesort").reset_index(drop=True)
    if rain.duplicated(["Station", "Datetime"]).any():
        raise RuntimeError("Duplicate canonical station-hour keys")
    annual_t, rolling_t = build_thresholds(rain)
    write_csv(annual_t, "annual_p99_thresholds.csv")
    write_csv(rolling_t, "rolling_historical_p99_thresholds.csv")

    years = rain["Datetime"].dt.year
    annual_map = annual_t.set_index(["Station", "Year"])["Annual_P99"]
    annual_threshold = pd.Series(pd.MultiIndex.from_arrays([rain["Station"], years]).map(annual_map), index=rain.index, dtype=float)
    rolling_map = rolling_t.set_index(["Station", "Year"])["Rolling_P99"]
    rolling_threshold = pd.Series(pd.MultiIndex.from_arrays([rain["Station"], years]).map(rolling_map), index=rain.index, dtype=float)
    frozen_map = pd.read_csv(FROZEN_P99_PATH).set_index("Station")["P99_Frozen"]
    frozen_threshold = rain["Station"].map(frozen_map).astype(float)

    annual_mask, annual_aux = event_mask(rain, annual_threshold)
    rolling_mask, rolling_aux = event_mask(rain, rolling_threshold)
    frozen_mask, frozen_aux = event_mask(rain, frozen_threshold)
    annual_events = event_table(rain, annual_mask, annual_threshold, annual_aux, "Annual_P99")
    rolling_events = event_table(rain, rolling_mask, rolling_threshold, rolling_aux, "Rolling_PriorYears_P99")
    frozen_events = event_table(rain, frozen_mask, frozen_threshold, frozen_aux, "Frozen_P99_Recomputed")

    official = pd.read_parquet(FROZEN_EVENTS_PATH)
    official_dt = "Datetime" if "Datetime" in official.columns else "Event_Datetime"
    official = official[["Station", official_dt]].rename(columns={official_dt: "Event_Datetime"})
    official["Event_Datetime"] = pd.to_datetime(official["Event_Datetime"])
    if keyset(official) != keyset(frozen_events):
        raise RuntimeError("Frozen Def5 recomputation does not exactly match official events")

    write_csv(annual_events, "annual_p99_events.csv")
    write_csv(rolling_events, "rolling_historical_p99_events.csv")
    count_tables(annual_events, "annual_p99")
    count_tables(rolling_events, "rolling_historical_p99")
    comparisons = pd.DataFrame([
        comparison_row("Frozen_P99", frozen_events, frozen_events),
        comparison_row("Annual_P99", annual_events, frozen_events),
        comparison_row("Rolling_PriorYears_P99", rolling_events, frozen_events),
    ])
    write_csv(comparisons, "target_definition_comparison.csv")

    ak, fk = keyset(annual_events), keyset(frozen_events)
    changes = []
    for status, keys in [("Added_by_Annual", ak-fk), ("Removed_by_Annual", fk-ak)]:
        for station, dt in sorted(keys): changes.append({"Change": status, "Station": station, "Event_Datetime": dt})
    write_csv(pd.DataFrame(changes), "annual_p99_added_removed_events.csv")
    hydro = hydrographs(rain, annual_events, frozen_events)
    write_csv(hydro, "representative_hydrographs.csv")

    compact = pd.read_parquet(COMPACT_PATH)
    compact["Datetime"] = pd.to_datetime(compact["Datetime"]); compact["Target_Datetime"] = pd.to_datetime(compact["Target_Datetime"])
    event_index = pd.MultiIndex.from_frame(annual_events[["Station", "Event_Datetime"]])
    row_index = pd.MultiIndex.from_frame(compact[["Station", "Target_Datetime"]].rename(columns={"Target_Datetime": "Event_Datetime"}))
    compact[TARGET] = row_index.isin(event_index).astype(np.int8)
    compact.to_parquet(HERE / "annual_p99_modelready_compact20.parquet", index=False, compression="zstd")
    role_summary = compact.groupby("Role")[TARGET].agg(Rows="size", Positives="sum").reset_index()
    role_summary["Negatives"] = role_summary["Rows"] - role_summary["Positives"]
    role_summary["Prevalence"] = role_summary["Positives"] / role_summary["Rows"]
    write_csv(role_summary, "annual_p99_model_population.csv")

    train = compact.loc[compact["Role"].eq("Train")]
    valid = compact.loc[compact["Role"].eq("Validation")]
    if train[TARGET].sum() == 0 or valid[TARGET].sum() == 0:
        raise RuntimeError("Annual target lacks positives in Train or Validation")
    train_weight = (len(train) - train[TARGET].sum()) / train[TARGET].sum()
    params = dict(objective="binary:logistic", eval_metric="aucpr", tree_method="hist", n_estimators=1500,
                  learning_rate=.05, max_depth=4, min_child_weight=5, subsample=.8, colsample_bytree=.8,
                  reg_lambda=1.0, reg_alpha=0.0, random_state=SEED, n_jobs=-1,
                  scale_pos_weight=float(train_weight), early_stopping_rounds=100)
    dev_model = XGBClassifier(**params)
    dev_model.fit(train[PREDICTORS].astype("float32"), train[TARGET].to_numpy(),
                  eval_set=[(valid[PREDICTORS].astype("float32"), valid[TARGET].to_numpy())], verbose=False)
    best_iteration = int(dev_model.best_iteration)
    valid_score = dev_model.predict_proba(valid[PREDICTORS].astype("float32"))[:, 1]
    valid_metrics = metric_row("Annual_P99_Validation2024", valid[TARGET].to_numpy(), valid_score)
    valid_metrics.update({"Best_Iteration_ZeroBased": best_iteration, "Selected_Rounds": best_iteration + 1,
                          "Scale_Pos_Weight": float(train_weight), "Used_For_Tuning": True})

    development = compact.loc[compact["Role"].isin(["Train", "Validation"])]
    dev_weight = (len(development) - development[TARGET].sum()) / development[TARGET].sum()
    final_params = {k: v for k, v in params.items() if k not in {"early_stopping_rounds", "n_estimators", "scale_pos_weight"}}
    final_params.update(n_estimators=best_iteration + 1, scale_pos_weight=float(dev_weight))
    final_model = XGBClassifier(**final_params)
    final_model.fit(development[PREDICTORS].astype("float32"), development[TARGET].to_numpy())
    final_model.save_model(HERE / "annual_p99_xgb_compact20.json")

    evaluation = compact.loc[compact["Role"].eq("Evaluation")].copy()
    eval_score = final_model.predict_proba(evaluation[PREDICTORS].astype("float32"))[:, 1]
    evaluation["Model_Score"] = eval_score
    evaluation["Predicted_0_5"] = (eval_score >= .5).astype(np.int8)
    evaluation[["Station", "Datetime", "Target_Datetime", TARGET, "Model_Score", "Predicted_0_5"]].to_parquet(
        HERE / "annual_p99_evaluation2025_predictions.parquet", index=False, compression="zstd")
    eval_metrics = metric_row("Annual_P99_Evaluation2025", evaluation[TARGET].to_numpy(), eval_score)
    eval_metrics.update({"Best_Iteration_ZeroBased": best_iteration, "Selected_Rounds": best_iteration + 1,
                         "Scale_Pos_Weight": float(dev_weight), "Used_For_Tuning": False})
    model_metrics = pd.DataFrame([valid_metrics, eval_metrics])
    write_csv(model_metrics, "annual_p99_model_metrics.csv")

    eval_annual = annual_events.loc[annual_events["Year"].eq(2025) & annual_events["Month"].between(5, 10)]
    eval_keys = set(zip(evaluation["Station"], evaluation["Target_Datetime"]))
    covered = len(keyset(eval_annual) & eval_keys)
    coverage = pd.DataFrame([{
        "Annual_P99_MayOct2025_Events": len(eval_annual), "Model_Evaluable_Events": covered,
        "Model_Universe_Coverage": covered / len(eval_annual) if len(eval_annual) else math.nan,
        "Detected_At_Threshold_0_5": int(eval_metrics["TP"]),
        "Detection_Coverage_Among_Model_Evaluable": eval_metrics["Recall"],
        "Detection_Coverage_Among_All_Annual_Events": eval_metrics["TP"] / len(eval_annual) if len(eval_annual) else math.nan,
    }])
    write_csv(coverage, "annual_p99_event_coverage_2025.csv")

    baseline = pd.read_csv(BASELINE_METRICS_PATH).iloc[0]
    side = pd.DataFrame([
        {"Target": "Frozen_Def5", **{k: baseline[k] for k in ["Evaluation_Rows", "Evaluation_Positives", "Evaluation_Prevalence", "AP", "ROC_AUC", "TP", "FP", "TN", "FN", "Precision", "Recall", "Specificity", "F1"]}},
        {"Target": "Annual_P99", "Evaluation_Rows": eval_metrics["Rows"], "Evaluation_Positives": eval_metrics["Positives"],
         "Evaluation_Prevalence": eval_metrics["Prevalence"], **{k: eval_metrics[k] for k in ["AP", "ROC_AUC", "TP", "FP", "TN", "FN", "Precision", "Recall", "Specificity", "F1"]}},
    ])
    write_csv(side, "annual_vs_frozen_model_metrics_2025.csv")
    save_figures(annual_t, comparisons, hydro, evaluation[TARGET].to_numpy(), eval_score)

    hashes_after = {str(p): sha256(p) for p in protected}
    unchanged = hashes_before == hashes_after
    if not unchanged:
        raise RuntimeError("A protected input hash changed")
    qa = {
        "status": "PASS", "python": sys.version, "protected_inputs_unchanged": unchanged,
        "hashes_before": hashes_before, "hashes_after": hashes_after,
        "rules": {"missing_is_never_zero": True, "exact_hourly_continuity": True,
                  "annual_p99_is_retrospective": True, "rolling_uses_prior_years_only": True,
                  "evaluation_2025_used_for_tuning": False, "baseline_artifacts_written": False},
        "checks": {"canonical_duplicate_keys": 0, "frozen_event_exact_match": True,
                   "annual_threshold_rows": len(annual_t), "annual_events": len(annual_events),
                   "rolling_events": len(rolling_events), "best_iteration_zero_based": best_iteration},
    }
    (HERE / "provenance_and_qa.json").write_text(json.dumps(qa, indent=2, default=str), encoding="utf-8")
    pd.DataFrame([{"Path": p, "SHA256_Before": hashes_before[p], "SHA256_After": hashes_after[p], "Unchanged": hashes_before[p] == hashes_after[p]} for p in hashes_before]).to_csv(HERE / "protected_input_hashes.csv", index=False)
    print(json.dumps({"status": "PASS", "annual_events": len(annual_events), "rolling_events": len(rolling_events),
                      "best_iteration": best_iteration, "evaluation_metrics": eval_metrics}, indent=2, default=str))


if __name__ == "__main__":
    main()
