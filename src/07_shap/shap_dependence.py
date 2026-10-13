"""Descriptive SHAP dependence diagnostics for the frozen Compact20 XGBoost model.

This script reads the already-computed Evaluation 2025 TreeSHAP table. It does
not load the booster for prediction, rescore observations, fit a model, optimize
a threshold, or alter any SHAP contribution. SHAP values are interpreted on the
native XGBoost raw-margin (log-odds) scale as model attribution, not causation.
"""

from __future__ import annotations
import sys as _sys
from pathlib import Path as _ConfigPath
_sys.path.insert(0, str(_ConfigPath(__file__).resolve().parents[1]))
from rainbomb_paths import load_paths as _load_paths
PATHS = _load_paths()

import hashlib
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = PATHS.runtime_root
SHAP_PATH = ROOT / "03_OUTPUT/07_SHAP_XAI/Evaluation2025_SHAP_v1.parquet"
GLOBAL_IMPORTANCE_PATH = ROOT / "03_OUTPUT/07_SHAP_XAI/Global_SHAP_Importance_v1.csv"
MODEL_PATH = ROOT / "03_OUTPUT/06_Final_Model_Evaluation/XGB_Final_Compact20_v1.json"
OUT_DIR = ROOT / "03_OUTPUT/07_SHAP_XAI/Dependence_Diagnostics"

EXPECTED_SHA256 = {
    SHAP_PATH: "95918E14000E8DA89D6090CBB05BE14BF686E9EA444244DB66D0F4681B4C1D90",
    GLOBAL_IMPORTANCE_PATH: "E0F28A8B6D48C927B7F7FCA584A4E77AE46DB8A7B5BDAE1E26DBC9357A820FE3",
    MODEL_PATH: "0C43852B2CD1F60DC89EFD225331BD6251691A8F835EA3C6DF65CB5CFAA274CD",
}

FEATURES = [
    "u_850",
    "q_850",
    "Rain_t",
    "u10",
    "Temp_t",
    "RH_t",
    "RainSum_3h",
    "SUHI_5K",
    "Elevation_m",
]

UNITS = {
    "u_850": "m s⁻¹",
    "q_850": "kg kg⁻¹",
    "Rain_t": "mm hour⁻¹",
    "u10": "m s⁻¹",
    "Temp_t": "°C",
    "RH_t": "%",
    "RainSum_3h": "mm over 3 hours",
    "SUHI_5K": "°C anomaly",
    "Elevation_m": "m",
}

PLOT_NAMES = {
    "u_850": "SHAP_Dependence_u850_v1.png",
    "q_850": "SHAP_Dependence_q850_v1.png",
    "Rain_t": "SHAP_Dependence_Rain_t_v1.png",
    "u10": "SHAP_Dependence_u10_v1.png",
    "Temp_t": "SHAP_Dependence_Temp_t_v1.png",
    "RH_t": "SHAP_Dependence_RH_t_v1.png",
    "RainSum_3h": "SHAP_Dependence_RainSum_3h_v1.png",
    "SUHI_5K": "SHAP_Dependence_SUHI_5K_v1.png",
    "Elevation_m": "SHAP_Dependence_Elevation_v1.png",
}

RANDOM_STATE = 154499
ALL_BINS = 20
POSITIVE_BINS = 8
SCATTER_N = 6000
CSV_FLOAT_FORMAT = "%.12g"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_frozen_inputs() -> None:
    for path, expected in EXPECTED_SHA256.items():
        if not path.exists():
            raise FileNotFoundError(f"Required frozen input is missing: {path}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"Frozen input hash mismatch for {path.name}: expected {expected}, got {actual}"
            )


def assign_quantile_bins(values: pd.Series, max_bins: int, zero_heavy: bool) -> pd.Series:
    """Assign robust quantile bins; keep zero rainfall as its own bin."""
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    labels = pd.Series(pd.NA, index=numeric.index, dtype="Int64")

    if zero_heavy and (numeric == 0).mean() >= 0.20 and (numeric > 0).any():
        if (numeric < 0).any():
            raise ValueError("Zero-heavy special handling expects nonnegative rainfall values.")
        labels.loc[numeric == 0] = 0
        positive = numeric.loc[numeric > 0]
        q = min(max_bins - 1, int(positive.nunique()), len(positive))
        if q > 0:
            categories = pd.qcut(positive, q=q, duplicates="drop")
            labels.loc[positive.index] = categories.cat.codes.astype("int64") + 1
    else:
        q = min(max_bins, int(numeric.nunique()), len(numeric))
        if q <= 1:
            labels.loc[:] = 0
        else:
            categories = pd.qcut(numeric, q=q, duplicates="drop")
            labels.loc[:] = categories.cat.codes.astype("int64")

    if labels.isna().any():
        raise RuntimeError("Bin assignment left unassigned observations.")
    return labels


def summarize_feature(data: pd.DataFrame, feature: str, max_bins: int) -> pd.DataFrame:
    shap_col = f"SHAP_{feature}"
    zero_heavy = feature in {"Rain_t", "RainSum_3h"}
    work = data[[feature, shap_col]].copy()
    work["_bin"] = assign_quantile_bins(work[feature], max_bins, zero_heavy)

    summary = (
        work.groupby("_bin", observed=True)
        .agg(
            Bin_Lower=(feature, "min"),
            Bin_Upper=(feature, "max"),
            N=(feature, "size"),
            Feature_Median=(feature, "median"),
            SHAP_Median=(shap_col, "median"),
            SHAP_Mean=(shap_col, "mean"),
            SHAP_Q25=(shap_col, lambda s: s.quantile(0.25)),
            SHAP_Q75=(shap_col, lambda s: s.quantile(0.75)),
            Fraction_SHAP_Positive=(shap_col, lambda s: (s > 0).mean()),
        )
        .reset_index(drop=True)
        .sort_values("Feature_Median", kind="stable")
        .reset_index(drop=True)
    )
    summary.insert(0, "Bin_ID", np.arange(1, len(summary) + 1, dtype=int))
    summary.insert(0, "Feature", feature)
    return summary


def direction_label(values: np.ndarray) -> str:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return "Not estimable"
    positive_fraction = float(np.mean(finite > 0))
    negative_fraction = float(np.mean(finite < 0))
    if positive_fraction >= 0.60:
        return "Median SHAP predominantly positive (higher model output)"
    if negative_fraction >= 0.60:
        return "Median SHAP predominantly negative (lower model output)"
    return "Mixed signed median SHAP"


def transition_for_feature(summary: pd.DataFrame, feature: str) -> dict[str, object]:
    ordered = summary.loc[summary["Feature"] == feature].sort_values("Feature_Median")
    med = ordered["SHAP_Median"].to_numpy(dtype=float)
    x = ordered["Feature_Median"].to_numpy(dtype=float)

    if len(med) < 2 or not np.isfinite(med).all():
        return {
            "Feature": feature,
            "Transition_Type": "NO STABLE TRANSITION IDENTIFIED",
            "Approx_Lower": np.nan,
            "Approx_Upper": np.nan,
            "Below_Range_Direction": "Not estimable",
            "Above_Range_Direction": "Not estimable",
            "Notes": "Insufficient binned median SHAP information.",
        }

    signs = np.sign(med)
    # Treat exact-zero medians conservatively by propagating the nearest nonzero sign.
    for idx in range(1, len(signs)):
        if signs[idx] == 0:
            signs[idx] = signs[idx - 1]
    for idx in range(len(signs) - 2, -1, -1):
        if signs[idx] == 0:
            signs[idx] = signs[idx + 1]

    changes = [idx for idx in range(len(signs) - 1) if signs[idx] * signs[idx + 1] < 0]
    if not changes:
        return {
            "Feature": feature,
            "Transition_Type": "NO STABLE TRANSITION IDENTIFIED",
            "Approx_Lower": np.nan,
            "Approx_Upper": np.nan,
            "Below_Range_Direction": direction_label(med),
            "Above_Range_Direction": direction_label(med),
            "Notes": "Binned median SHAP did not change sign across the observed Evaluation range.",
        }

    intervals = [(float(min(x[i], x[i + 1])), float(max(x[i], x[i + 1]))) for i in changes]
    interval_text = "; ".join(f"{lo:.6g} to {hi:.6g}" for lo, hi in intervals)
    if len(changes) == 1:
        idx = changes[0]
        orientation = "NEGATIVE-TO-POSITIVE" if signs[idx] < signs[idx + 1] else "POSITIVE-TO-NEGATIVE"
        transition_type = f"SINGLE {orientation}"
        below = direction_label(med[: idx + 1])
        above = direction_label(med[idx + 1 :])
        lower, upper = intervals[0]
        notes = (
            "One approximate SHAP zero-crossing range in the all-row binned median pattern; "
            "this is a model-response transition, not a physical threshold."
        )
    else:
        transition_type = "NONLINEAR / MULTIPLE TRANSITIONS"
        lower = min(lo for lo, _ in intervals)
        upper = max(hi for _, hi in intervals)
        below = direction_label(med[: changes[0] + 1])
        above = direction_label(med[changes[-1] + 1 :])
        notes = (
            f"{len(changes)} sign changes in binned median SHAP; individual approximate ranges: "
            f"{interval_text}. No single threshold is supported."
        )

    return {
        "Feature": feature,
        "Transition_Type": transition_type,
        "Approx_Lower": lower,
        "Approx_Upper": upper,
        "Below_Range_Direction": below,
        "Above_Range_Direction": above,
        "Notes": notes,
    }


def save_dependence_plot(
    data: pd.DataFrame, summary: pd.DataFrame, feature: str, seed_offset: int
) -> None:
    shap_col = f"SHAP_{feature}"
    sample_n = min(SCATTER_N, len(data))
    sample = data[[feature, shap_col]].sample(
        n=sample_n, random_state=RANDOM_STATE + seed_offset, replace=False
    )
    bins = summary.loc[summary["Feature"] == feature].sort_values("Feature_Median")

    fig, ax = plt.subplots(figsize=(10.2, 6.4), dpi=180)
    ax.scatter(
        sample[feature],
        sample[shap_col],
        s=9,
        alpha=0.13,
        color="#355C7D",
        linewidths=0,
        rasterized=True,
        label=f"Evaluation sample (n={sample_n:,})",
    )
    x = bins["Feature_Median"].to_numpy(dtype=float)
    median = bins["SHAP_Median"].to_numpy(dtype=float)
    q25 = bins["SHAP_Q25"].to_numpy(dtype=float)
    q75 = bins["SHAP_Q75"].to_numpy(dtype=float)
    ax.fill_between(x, q25, q75, color="#F4A261", alpha=0.28, label="Binned SHAP IQR")
    ax.plot(x, median, color="#D1495B", marker="o", markersize=4, linewidth=2.1,
            label="Binned median SHAP")
    ax.axhline(0, color="#222222", linewidth=1.0, linestyle="--")
    ax.set_title(f"Frozen XGBoost SHAP dependence: {feature}", loc="left", weight="bold")
    ax.set_xlabel(f"Observed {feature} ({UNITS[feature]})")
    ax.set_ylabel("Signed SHAP contribution to raw margin (log-odds)")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="best")
    ax.text(
        0.01,
        0.01,
        "SHAP > 0 raises the model raw margin; SHAP < 0 lowers it. Attribution, not causation.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#444444",
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / PLOT_NAMES[feature], bbox_inches="tight", facecolor="white")
    plt.close(fig)


def hour_summary_and_plot(data: pd.DataFrame) -> pd.DataFrame:
    work = data[["Datetime", "Target_Def5_tPlus1", "SHAP_Hour_sin", "SHAP_Hour_cos"]].copy()
    work["Hour"] = pd.to_datetime(work["Datetime"]).dt.hour.astype(int)
    work["Hour_Total_SHAP"] = work["SHAP_Hour_sin"] + work["SHAP_Hour_cos"]
    summary = (
        work.groupby("Hour", observed=True)
        .agg(
            N=("Hour_Total_SHAP", "size"),
            Hour_Total_SHAP_Mean=("Hour_Total_SHAP", "mean"),
            Hour_Total_SHAP_Median=("Hour_Total_SHAP", "median"),
            Hour_Total_SHAP_Q25=("Hour_Total_SHAP", lambda s: s.quantile(0.25)),
            Hour_Total_SHAP_Q75=("Hour_Total_SHAP", lambda s: s.quantile(0.75)),
            Fraction_Hour_Total_SHAP_Positive=("Hour_Total_SHAP", lambda s: (s > 0).mean()),
            Actual_Def5_Count=("Target_Def5_tPlus1", "sum"),
        )
        .reset_index()
    )
    if summary["Hour"].tolist() != list(range(24)):
        raise RuntimeError("Expected all predictor hours 0-23 in Evaluation data.")

    fig, ax = plt.subplots(figsize=(10.2, 6.0), dpi=180)
    x = summary["Hour"].to_numpy(dtype=float)
    q25 = summary["Hour_Total_SHAP_Q25"].to_numpy(dtype=float)
    q75 = summary["Hour_Total_SHAP_Q75"].to_numpy(dtype=float)
    med = summary["Hour_Total_SHAP_Median"].to_numpy(dtype=float)
    ax.fill_between(x, q25, q75, color="#68A7AD", alpha=0.32, label="IQR")
    ax.plot(x, med, color="#B23A48", marker="o", linewidth=2.2, label="Median combined SHAP")
    ax.axhline(0, color="#222222", linewidth=1.0, linestyle="--")
    ax.set_xticks(range(24))
    ax.set_xlim(-0.4, 23.4)
    ax.set_title("Frozen XGBoost diurnal model response", loc="left", weight="bold")
    ax.set_xlabel("Predictor hour at forecast time t (ICT; Datetime.hour)")
    ax.set_ylabel("SHAP_Hour_sin + SHAP_Hour_cos (raw margin / log-odds)")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="best")
    ax.text(
        0.01,
        0.01,
        "The paired cyclic encoding is interpreted jointly. Positive raises and negative lowers model output.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#444444",
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "SHAP_HourOfDay_Response_v1.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return summary


def format_value(value: float, feature: str) -> str:
    if pd.isna(value):
        return "—"
    if feature == "q_850":
        return f"{value:.5f}"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.3g}"


def contiguous_hours(hours: list[int]) -> str:
    if not hours:
        return "none"
    runs: list[tuple[int, int]] = []
    start = previous = hours[0]
    for hour in hours[1:]:
        if hour == previous + 1:
            previous = hour
            continue
        runs.append((start, previous))
        start = previous = hour
    runs.append((start, previous))
    return ", ".join(f"{a:02d}:00" if a == b else f"{a:02d}:00–{b:02d}:00" for a, b in runs)


def caution_for(feature: str) -> str:
    cautions = {
        "RainSum_3h": "Definitional coupling: Def5 includes antecedent rainfall; not independent causal evidence.",
        "Rain_t": "Model attribution to observed rainfall at t; not a physical trigger value.",
        "SUHI_5K": "Local LST anomaly versus 5 km rural background; not absolute urban temperature or causal urban effect.",
        "Elevation_m": "Small model contribution would not imply that topography lacks physical influence.",
        "u_850": "Eastward wind component in m s⁻¹; sign is not interpreted beyond documented component direction.",
        "u10": "Eastward 10 m wind component in m s⁻¹; sign is not interpreted beyond documented component direction.",
        "q_850": "Specific humidity in kg kg⁻¹; attribution is model use, not causal meteorological attribution.",
    }
    return cautions.get(feature, "Descriptive model attribution on raw-margin scale; not a causal or physical threshold.")


def build_report(
    data: pd.DataFrame,
    all_summary: pd.DataFrame,
    positive_summary: pd.DataFrame,
    transitions: pd.DataFrame,
    positive_values: pd.DataFrame,
    hour_summary: pd.DataFrame,
) -> str:
    positive_hours = hour_summary.loc[hour_summary["Hour_Total_SHAP_Median"] > 0, "Hour"].astype(int).tolist()
    negative_hours = hour_summary.loc[hour_summary["Hour_Total_SHAP_Median"] < 0, "Hour"].astype(int).tolist()
    max_hour = hour_summary.loc[hour_summary["Hour_Total_SHAP_Median"].idxmax()]
    min_hour = hour_summary.loc[hour_summary["Hour_Total_SHAP_Median"].idxmin()]
    multiple = transitions.loc[
        transitions["Transition_Type"] == "NONLINEAR / MULTIPLE TRANSITIONS", "Feature"
    ].tolist()

    lines = [
        "# SHAP Dependence / Model-Response Diagnostics v1",
        "",
        "## Scope and frozen sources",
        "",
        f"This optional extension describes the frozen Compact20 XGBoost model response using all {len(data):,} Evaluation 2025 rows ({int(data['Target_Def5_tPlus1'].sum()):,} actual Def5 rows). It reads `Evaluation2025_SHAP_v1.parquet` and `Global_SHAP_Importance_v1.csv`; the frozen `XGB_Final_Compact20_v1.json` is hash-checked for provenance but is not loaded for prediction. No model was retrained or rescored, no SHAP value was modified, and no threshold was optimized.",
        "",
        "TreeSHAP values are native XGBoost raw-margin/log-odds contributions. A positive value raises the model raw margin relative to its baseline; a negative value lowers it. These diagnostics describe model attribution, not causal effects or physical thresholds.",
        "",
        "## Predictors and units",
        "",
        "The continuous diagnostics cover `u_850` (m s⁻¹), `q_850` (kg kg⁻¹), `Rain_t` (mm hour⁻¹), `u10` (m s⁻¹), `Temp_t` (°C), `RH_t` (%), `RainSum_3h` (mm over 3 hours), `SUHI_5K` (°C anomaly), and `Elevation_m` (m). Wind-component signs are retained without interpretation beyond the documented eastward/westward component convention. `SUHI_5K` follows the frozen project definition `LST_Median - Local_Rural_LST_Median_5km` in °C.",
        "",
        "## Binning and plots",
        "",
        f"All-row summaries use up to {ALL_BINS} quantile bins. The 95-row Def5 subset uses up to {POSITIVE_BINS} quantile bins to avoid forcing sparsely populated groups. Zero-heavy rainfall keeps an exact-zero bin and quantile-bins positive values. Each plot overlays these all-row median and IQR summaries on a reproducible unstratified sample of up to {SCATTER_N:,} Evaluation rows (`random_state = {RANDOM_STATE}` plus a fixed feature offset). No regression or causal smoother is fitted.",
        "",
        "## Predictor model-response shapes and approximate transitions",
        "",
    ]

    table_rows = []
    for feature in FEATURES:
        bins = all_summary.loc[all_summary["Feature"] == feature].sort_values("Feature_Median")
        trans = transitions.loc[transitions["Feature"] == feature].iloc[0]
        peak = bins.loc[bins["SHAP_Median"].idxmax()]
        trough = bins.loc[bins["SHAP_Median"].idxmin()]
        pattern = (
            f"Binned median SHAP ranged from {trough['SHAP_Median']:.3f} near "
            f"{format_value(trough['Feature_Median'], feature)} to {peak['SHAP_Median']:.3f} near "
            f"{format_value(peak['Feature_Median'], feature)} {UNITS[feature]}; "
            f"{str(trans['Transition_Type']).lower()}."
        )
        if pd.notna(trans["Approx_Lower"]):
            transition_text = (
                f"~{format_value(float(trans['Approx_Lower']), feature)}–"
                f"{format_value(float(trans['Approx_Upper']), feature)} {UNITS[feature]}"
            )
        else:
            transition_text = "No stable transition identified"
        positive_bins = bins.loc[bins["SHAP_Median"] > 0]
        if len(positive_bins):
            higher = (
                f"Positive binned medians occur within observed values ~"
                f"{format_value(float(positive_bins['Bin_Lower'].min()), feature)}–"
                f"{format_value(float(positive_bins['Bin_Upper'].max()), feature)} {UNITS[feature]}"
            )
        else:
            higher = "No bin had positive median SHAP"
        lines.extend([f"### {feature}", "", pattern, "", str(trans["Notes"]), ""])
        table_rows.append((feature, pattern, transition_text, higher, caution_for(feature)))

    lines.extend([
        "## Nonlinear or multiple transitions",
        "",
        (", ".join(f"`{f}`" for f in multiple) if multiple else "No feature showed multiple sign changes in the all-row binned median SHAP pattern."),
        "",
        "Multiple sign changes are labeled `NONLINEAR / MULTIPLE TRANSITIONS`; their reported lower and upper values are an envelope around the individual crossing ranges, not a single threshold.",
        "",
        "## Hour-of-day combined SHAP response",
        "",
        f"Using predictor `Datetime.hour`, the combined `SHAP_Hour_sin + SHAP_Hour_cos` median was positive at {contiguous_hours(positive_hours)} and negative at {contiguous_hours(negative_hours)}. The highest median combined contribution occurred at {int(max_hour['Hour']):02d}:00 ({max_hour['Hour_Total_SHAP_Median']:.3f}); the lowest occurred at {int(min_hour['Hour']):02d}:00 ({min_hour['Hour_Total_SHAP_Median']:.3f}). This is the frozen model's joint cyclic time encoding, not a physical interpretation of either sine or cosine term alone.",
        "",
        "## Actual Def5 event context",
        "",
        "The following interquartile value ranges and attribution fractions describe only the 95 observed Def5 rows. They do not define Def5 occurrence:",
        "",
        "| Feature | Def5 value IQR | Median value | Median SHAP | Fraction SHAP > 0 |",
        "|---|---:|---:|---:|---:|",
    ])
    for _, row in positive_values.iterrows():
        feature = str(row["Feature"])
        lines.append(
            f"| {feature} | {format_value(row['Value_Q25'], feature)}–{format_value(row['Value_Q75'], feature)} {UNITS[feature]} | "
            f"{format_value(row['Value_Median'], feature)} | {row['SHAP_Median']:.3f} | {row['Fraction_SHAP_Positive']:.1%} |"
        )

    suhi = transitions.loc[transitions["Feature"] == "SUHI_5K"].iloc[0]
    elev = transitions.loc[transitions["Feature"] == "Elevation_m"].iloc[0]
    lines.extend([
        "",
        "## Geographic/context findings and cautions",
        "",
        f"`SUHI_5K` is a local surface-temperature anomaly relative to rural background within 5 km, not absolute urban temperature. Its binned response is classified as **{suhi['Transition_Type']}**. The attribution does not show that urbanization causes Def5.",
        "",
        f"`Elevation_m` is classified as **{elev['Transition_Type']}**. Even if its SHAP range is small, that means elevation contributed little to this frozen model relative to other predictors in this Evaluation period; it does not mean topography lacks physical influence.",
        "",
        "`RainSum_3h` requires special care because Def5 includes an antecedent-rainfall criterion. Any apparent model-response transition can partly reflect definitional coupling with the operational target and is not independent evidence of a physical causal mechanism.",
        "",
        "ERA5 fields (`u_850`, `q_850`) and `u10` show how the frozen model used those fields. They do not supply causal meteorological attribution.",
        "",
        "## Limitations",
        "",
        "Quantile summaries describe marginal plotted relationships while other predictors vary and correlated predictors can share attribution. Binned medians can hide within-bin heterogeneity, the 95-event subset is small, static station variables repeat across station-hours, and tree-model response shapes can be discontinuous. Approximate SHAP zero-crossing ranges are model-response transitions only—not causal thresholds, rainfall thresholds, physical triggers, or statistically tested cut points.",
        "",
        "## Concise interpretation table",
        "",
        "| Feature | Observed response pattern | Approximate transition range | Direction associated with higher model output | Interpretation caution |",
        "|---|---|---|---|---|",
    ])
    for feature, pattern, transition_text, higher, caution in table_rows:
        lines.append(f"| {feature} | {pattern} | {transition_text} | {higher} | {caution} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    verify_frozen_inputs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Read global importance for frozen-source validation/provenance only.
    global_importance = pd.read_csv(GLOBAL_IMPORTANCE_PATH)
    if set(FEATURES) - set(global_importance["Feature"]):
        raise RuntimeError("Requested diagnostic feature is absent from frozen global importance output.")

    data = pd.read_parquet(SHAP_PATH)
    required = ["Datetime", "Target_Def5_tPlus1", "SHAP_Hour_sin", "SHAP_Hour_cos"]
    required += FEATURES + [f"SHAP_{feature}" for feature in FEATURES]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")
    if data[required].isna().any().any():
        raise RuntimeError("Required SHAP diagnostic columns contain missing values.")
    if len(data) != 121_296 or int(data["Target_Def5_tPlus1"].sum()) != 95:
        raise RuntimeError("Evaluation population differs from the frozen 121,296 rows / 95 positives.")

    positives = data.loc[data["Target_Def5_tPlus1"] == 1].copy()
    all_summary = pd.concat(
        [summarize_feature(data, feature, ALL_BINS) for feature in FEATURES], ignore_index=True
    )
    positive_summary = pd.concat(
        [summarize_feature(positives, feature, POSITIVE_BINS) for feature in FEATURES],
        ignore_index=True,
    )

    transitions = pd.DataFrame(
        [transition_for_feature(all_summary, feature) for feature in FEATURES],
        columns=[
            "Feature",
            "Transition_Type",
            "Approx_Lower",
            "Approx_Upper",
            "Below_Range_Direction",
            "Above_Range_Direction",
            "Notes",
        ],
    )

    positive_rows = []
    for feature in FEATURES:
        shap_col = f"SHAP_{feature}"
        values = positives[feature]
        contributions = positives[shap_col]
        positive_rows.append(
            {
                "Feature": feature,
                "Unit": UNITS[feature],
                "N": len(positives),
                "Value_Min": values.min(),
                "Value_Q25": values.quantile(0.25),
                "Value_Median": values.median(),
                "Value_Q75": values.quantile(0.75),
                "Value_Max": values.max(),
                "SHAP_Median": contributions.median(),
                "Fraction_SHAP_Positive": (contributions > 0).mean(),
            }
        )
    positive_values = pd.DataFrame(positive_rows)
    hour_summary = hour_summary_and_plot(data)

    all_summary.to_csv(
        OUT_DIR / "SHAP_Dependence_Binned_AllRows_v1.csv",
        index=False,
        float_format=CSV_FLOAT_FORMAT,
    )
    positive_summary.to_csv(
        OUT_DIR / "SHAP_Dependence_Binned_Def5Positive_v1.csv",
        index=False,
        float_format=CSV_FLOAT_FORMAT,
    )
    transitions.to_csv(
        OUT_DIR / "SHAP_Model_Response_Transitions_v1.csv",
        index=False,
        float_format=CSV_FLOAT_FORMAT,
    )
    positive_values.to_csv(
        OUT_DIR / "Def5_Positive_Feature_Value_SHAP_Summary_v1.csv",
        index=False,
        float_format=CSV_FLOAT_FORMAT,
    )
    hour_summary.to_csv(
        OUT_DIR / "SHAP_HourOfDay_Summary_v1.csv",
        index=False,
        float_format=CSV_FLOAT_FORMAT,
    )

    for idx, feature in enumerate(FEATURES):
        save_dependence_plot(data, all_summary, feature, idx)

    report = build_report(
        data,
        all_summary,
        positive_summary,
        transitions,
        positive_values,
        hour_summary,
    )
    (OUT_DIR / "SHAP_Dependence_Diagnostics_Report_v1.md").write_text(
        report, encoding="utf-8", newline="\n"
    )

    # Reproducibility/audit checks on the newly created summaries.
    all_counts = all_summary.groupby("Feature")["N"].sum()
    positive_counts = positive_summary.groupby("Feature")["N"].sum()
    if not (all_counts == len(data)).all():
        raise RuntimeError("All-row bin counts do not reconcile to the Evaluation population.")
    if not (positive_counts == len(positives)).all():
        raise RuntimeError("Positive-row bin counts do not reconcile to the 95-event population.")
    if int(hour_summary["N"].sum()) != len(data):
        raise RuntimeError("Hour summary counts do not reconcile to the Evaluation population.")
    if int(hour_summary["Actual_Def5_Count"].sum()) != len(positives):
        raise RuntimeError("Hour summary Def5 counts do not reconcile to the 95-event population.")

    print(f"Created dependence diagnostics in: {OUT_DIR}")
    print(f"Rows: {len(data):,}; Def5 positives: {len(positives):,}")
    print(transitions.to_string(index=False))


if __name__ == "__main__":
    main()
