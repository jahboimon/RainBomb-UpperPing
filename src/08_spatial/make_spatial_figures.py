"""Create final point-station maps, case figure, and Phase 8 report."""

from __future__ import annotations
import sys as _sys
from pathlib import Path as _ConfigPath
_sys.path.insert(0, str(_ConfigPath(__file__).resolve().parents[1]))
from rainbomb_paths import load_paths as _load_paths
PATHS = _load_paths()

import hashlib
import textwrap
from collections import Counter
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patheffects import withStroke


ROOT = PATHS.runtime_root
LEGACY = PATHS.upstream_root
OUT = ROOT / "03_OUTPUT/08_Spatial_Analysis"

STATION_TABLE = OUT / "Station_Evaluation_Spatial_Summary_v1.csv"
BUILT_TABLE = OUT / "BuiltUp_vs_NonBuiltUp_Spatial_Summary_v1.csv"
CASE_TABLE = OUT / "Spatial_Event_CaseStudies_v1.csv"
STATION_SHAP_LONG = ROOT / "03_OUTPUT/07_SHAP_XAI/Station_SHAP_Long_v1.csv"
STATION_SHAP_SUMMARY = ROOT / "03_OUTPUT/07_SHAP_XAI/Station_SHAP_Summary_v1.csv"
GLOBAL_SHAP = ROOT / "03_OUTPUT/07_SHAP_XAI/Global_SHAP_Importance_v1.csv"
PHASE7_REPORT = ROOT / "03_OUTPUT/07_SHAP_XAI/Phase7_SHAP_Report_v1.md"

STATION_METADATA = LEGACY / "Hourly_rain/RESETWORKPLACE/02_processed/Station_Metadata_Cleaned.csv"
BOUNDARY = LEGACY / "Station_spatial_diagnostics_v1/selected_study_area_boundary_v1.geojson"
LONG_TERM = LEGACY / "Def5_Station_Dynamics_v1.csv"

DEF5_MAP = OUT / "Station_Def5_Rate_Map_v1.png"
RECALL_MAP = OUT / "Station_Recall_Map_v1.png"
FP_MAP = OUT / "Station_FP_Rate_Map_v1.png"
MEDIAN_MAP = OUT / "Station_Median_RiskScore_Map_v1.png"
Q95_MAP = OUT / "Station_Q95_RiskScore_Map_v1.png"
DOMINANT_MAP = OUT / "Station_Dominant_SHAP_Driver_Map_v1.png"
HEATMAP = OUT / "Station_SHAP_TopFeatures_Heatmap_v1.png"
LONG_TERM_MAP = OUT / "LongTerm_Def5_Rate_Context_Map_v1.png"
CASE_FIGURE = OUT / "Spatial_Event_CaseStudies_v1.png"
REPORT = OUT / "Phase8_Spatial_Analysis_Report_v1.md"

FIGURES = [
    DEF5_MAP, RECALL_MAP, FP_MAP, MEDIAN_MAP, Q95_MAP, DOMINANT_MAP,
    HEATMAP, LONG_TERM_MAP, CASE_FIGURE,
]
OUTPUTS = FIGURES + [REPORT]

EXPECTED_STATIONS = 38
EXPECTED_ROWS = 121_296
EXPECTED_POSITIVES = 95
EXPECTED_LONG_TERM_STATIONS = 40
EXPECTED_LONG_TERM_EVENTS = 1_123
EXPECTED_HASHES = {
    STATION_METADATA: "C8AD54E3CD5A3AD1591C7C9766C1B3296A67A9A5DD8D657C4FC60BD2573396BB",
    BOUNDARY: "B8A2A12539D48F779B5ABD21DE15996AE3959C23D05503FA79B4E85E4514A84D",
    LONG_TERM: "DE73FC32DB2330A0B9336078988FB07CF15C2665F04CE5B04F6D3C2ED01F8386",
}

FONT = "DejaVu Sans"
POINT_EDGE = "#263238"
OUTCOME_COLORS = {"TP": "#18864B", "FP": "#E07A1F", "FN": "#7847A7"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def verify_inputs() -> None:
    required = [
        STATION_TABLE, BUILT_TABLE, CASE_TABLE, STATION_SHAP_LONG,
        STATION_SHAP_SUMMARY, GLOBAL_SHAP, PHASE7_REPORT, *EXPECTED_HASHES,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required frozen or Phase 8 input(s): {missing}")
    for path, expected in EXPECTED_HASHES.items():
        if sha256(path) != expected:
            raise RuntimeError(f"Frozen spatial-input SHA256 mismatch: {path}")


def load_inputs() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.DataFrame, gpd.GeoDataFrame, pd.DataFrame,
]:
    stations = pd.read_csv(STATION_TABLE)
    built = pd.read_csv(BUILT_TABLE)
    cases = pd.read_csv(CASE_TABLE, parse_dates=["Datetime", "Target_Datetime"])
    shap_long = pd.read_csv(STATION_SHAP_LONG)
    shap_summary = pd.read_csv(STATION_SHAP_SUMMARY)
    global_shap = pd.read_csv(GLOBAL_SHAP)
    metadata = pd.read_csv(STATION_METADATA)
    boundary = gpd.read_file(BOUNDARY)
    long_term = pd.read_csv(LONG_TERM)

    if len(stations) != EXPECTED_STATIONS or int(stations["Evaluation_Rows"].sum()) != EXPECTED_ROWS:
        raise RuntimeError("Phase 8 station table does not reconcile with Evaluation rows.")
    if int(stations["Actual_Def5_Count"].sum()) != EXPECTED_POSITIVES:
        raise RuntimeError("Phase 8 station table does not reconcile with Evaluation positives.")
    if len(cases) != 3 or set(cases["Outcome"]) != {"TP", "FP", "FN"}:
        raise RuntimeError("Phase 8 case table must contain exactly TP, FP, and FN.")
    if len(shap_summary) != EXPECTED_STATIONS or len(shap_long) != EXPECTED_STATIONS * 20:
        raise RuntimeError("Phase 7 station SHAP tables have unexpected coverage.")
    if len(global_shap) != 20:
        raise RuntimeError("Global SHAP ranking must contain exactly 20 predictors.")
    if len(boundary) != 1 or str(boundary.crs).upper() != "EPSG:4326" or not boundary.is_valid.all():
        raise RuntimeError("Context boundary is not the expected valid WGS 84 polygon.")
    if len(long_term) != EXPECTED_LONG_TERM_STATIONS or int(long_term["Def5_Count"].sum()) != EXPECTED_LONG_TERM_EVENTS:
        raise RuntimeError("Frozen long-term station table has unexpected coverage.")
    if set(long_term["Station"]) != set(metadata["Station"]):
        raise RuntimeError("Long-term station table and frozen metadata do not align.")
    return stations, built, cases, shap_long, shap_summary, global_shap, boundary, long_term


def add_map_context(ax: plt.Axes, boundary: gpd.GeoDataFrame) -> None:
    boundary.plot(ax=ax, facecolor="#F3F1EB", edgecolor="#666666", linewidth=1.1, zorder=0)
    xmin, ymin, xmax, ymax = boundary.total_bounds
    x_pad = (xmax - xmin) * 0.035
    y_pad = (ymax - ymin) * 0.035
    ax.set_xlim(xmin - x_pad, xmax + x_pad)
    ax.set_ylim(ymin - y_pad, ymax + y_pad)
    mid_lat = (ymin + ymax) / 2
    ax.set_aspect(1 / np.cos(np.deg2rad(mid_lat)))
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.grid(color="#FFFFFF", linewidth=0.7, alpha=0.9, zorder=-1)

    ax.annotate(
        "N", xy=(0.93, 0.92), xytext=(0.93, 0.82), xycoords="axes fraction",
        ha="center", va="center", fontsize=11, fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color="#222222", lw=1.6),
    )
    scale_km = 50
    degree_length = scale_km / (111.32 * np.cos(np.deg2rad(mid_lat)))
    x0 = xmin + (xmax - xmin) * 0.08
    y0 = ymin + (ymax - ymin) * 0.075
    ax.plot([x0, x0 + degree_length], [y0, y0], color="#222222", lw=3, solid_capstyle="butt")
    ax.plot([x0, x0], [y0 - 0.012, y0 + 0.012], color="#222222", lw=1.2)
    ax.plot([x0 + degree_length, x0 + degree_length], [y0 - 0.012, y0 + 0.012], color="#222222", lw=1.2)
    ax.text(x0 + degree_length / 2, y0 + 0.035, f"{scale_km} km (approx.)", ha="center", fontsize=8)


def save_continuous_map(
    stations: pd.DataFrame,
    boundary: gpd.GeoDataFrame,
    value_column: str,
    output: Path,
    title: str,
    subtitle: str,
    colorbar_label: str,
    cmap: str,
    source_note: str,
    vmin: float | None = None,
    vmax: float | None = None,
    missing_label: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.7, 10.2))
    add_map_context(ax, boundary)
    valid = stations[value_column].notna()
    scatter = ax.scatter(
        stations.loc[valid, "Longitude"], stations.loc[valid, "Latitude"],
        c=stations.loc[valid, value_column], cmap=cmap, vmin=vmin, vmax=vmax,
        s=100, edgecolor=POINT_EDGE, linewidth=0.7, zorder=3,
    )
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.044, pad=0.035)
    colorbar.set_label(colorbar_label)
    if (~valid).any():
        ax.scatter(
            stations.loc[~valid, "Longitude"], stations.loc[~valid, "Latitude"],
            marker="X", s=105, facecolor="#D1D5DB", edgecolor="#4B5563",
            linewidth=0.9, zorder=4, label=missing_label or "Not available",
        )
        ax.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=8)
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=16)
    ax.text(0, 1.005, subtitle, transform=ax.transAxes, fontsize=9.2, color="#4B5563")
    fig.text(0.11, 0.025, "CRS: WGS 84 (EPSG:4326). " + source_note, fontsize=7.6, color="#4B5563")
    fig.subplots_adjust(left=0.12, right=0.88, top=0.91, bottom=0.10)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_dominant_map(
    stations: pd.DataFrame, shap_summary: pd.DataFrame, boundary: gpd.GeoDataFrame
) -> None:
    mapped = stations.merge(
        shap_summary[["Station", "Dominant_SHAP_Feature", "Dominant_SHAP_MeanAbs"]],
        on="Station", validate="one_to_one",
    )
    categories = sorted(mapped["Dominant_SHAP_Feature"].unique())
    palette = dict(zip(categories, ["#2A6FBB", "#D95F02", "#228B4E", "#8C4AA8", "#C13F3F"]))
    fig, ax = plt.subplots(figsize=(8.7, 10.2))
    add_map_context(ax, boundary)
    for feature in categories:
        subset = mapped.loc[mapped["Dominant_SHAP_Feature"].eq(feature)]
        ax.scatter(
            subset["Longitude"], subset["Latitude"], s=110, color=palette[feature],
            edgecolor=POINT_EDGE, linewidth=0.75, zorder=3, label=feature,
        )
    ax.legend(title="Highest station mean |SHAP|", loc="lower right", fontsize=8.5, title_fontsize=9)
    ax.set_title("Dominant SHAP feature by station", loc="left", fontsize=15, fontweight="bold", pad=16)
    ax.text(
        0, 1.005, "Model attribution category; not a physical-cause classification",
        transform=ax.transAxes, fontsize=9.2, color="#4B5563",
    )
    fig.text(
        0.11, 0.025,
        "CRS: WGS 84 (EPSG:4326). Sources: Station_SHAP_Summary_v1.csv; frozen station metadata; project boundary.",
        fontsize=7.6, color="#4B5563",
    )
    fig.subplots_adjust(left=0.12, right=0.92, top=0.91, bottom=0.10)
    fig.savefig(DOMINANT_MAP, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_shap_heatmap(
    stations: pd.DataFrame, shap_long: pd.DataFrame, global_shap: pd.DataFrame
) -> None:
    top_features = global_shap.sort_values("Rank_Mean_Abs_SHAP").head(10)["Feature"].tolist()
    station_order = stations.sort_values(["Latitude", "Station"], ascending=[False, True])["Station"].tolist()
    matrix = (
        shap_long.loc[shap_long["Feature"].isin(top_features)]
        .pivot(index="Station", columns="Feature", values="Mean_Abs_SHAP")
        .reindex(index=station_order, columns=top_features)
    )
    if matrix.isna().any().any():
        raise RuntimeError("Station SHAP heatmap matrix contains missing station-feature values.")
    fig, ax = plt.subplots(figsize=(12.2, 11.0))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(len(top_features)), labels=top_features, rotation=42, ha="right")
    ax.set_yticks(np.arange(len(station_order)), labels=station_order, fontsize=8)
    ax.set_xlabel("Global top-10 predictors")
    ax.set_ylabel("Stations ordered north to south")
    ax.set_title("Station mean |SHAP| for global top predictors", loc="left", fontsize=15, fontweight="bold", pad=16)
    ax.text(
        0, 1.008, "Raw-margin model attribution; values are not physical effects",
        transform=ax.transAxes, fontsize=9.2, color="#4B5563",
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Station mean |SHAP| (raw margin / log-odds)")
    fig.text(
        0.10, 0.018,
        "Source: Phase 7 Station_SHAP_Long_v1.csv and Global_SHAP_Importance_v1.csv. No spatial interpolation.",
        fontsize=7.8, color="#4B5563",
    )
    fig.subplots_adjust(left=0.10, right=0.91, top=0.91, bottom=0.17)
    fig.savefig(HEATMAP, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_case_figure(
    stations: pd.DataFrame, cases: pd.DataFrame, boundary: gpd.GeoDataFrame
) -> None:
    case_order = ["TP", "FP", "FN"]
    cases = cases.set_index("Outcome").loc[case_order].reset_index()
    fig = plt.figure(figsize=(13.2, 11.2))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.18, 0.92], hspace=0.38, wspace=0.36)
    map_ax = fig.add_subplot(grid[0, :])
    add_map_context(map_ax, boundary)
    map_ax.scatter(
        stations["Longitude"], stations["Latitude"], s=34, facecolor="#D1D5DB",
        edgecolor="#6B7280", linewidth=0.5, zorder=2, label="Other Evaluation station",
    )
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor="#D1D5DB",
                      markeredgecolor="#6B7280", markersize=7, label="Other Evaluation station")]
    for outcome in case_order:
        row = cases.loc[cases["Outcome"].eq(outcome)].iloc[0]
        map_ax.scatter(
            row["Longitude"], row["Latitude"], s=150, marker="o",
            color=OUTCOME_COLORS[outcome], edgecolor="#111827", linewidth=1.0, zorder=5,
        )
        label = map_ax.annotate(
            f"{outcome}: {row['Station']}", (row["Longitude"], row["Latitude"]),
            xytext=(6, 6), textcoords="offset points", fontsize=9, fontweight="bold",
            color=OUTCOME_COLORS[outcome], zorder=6,
        )
        label.set_path_effects([withStroke(linewidth=2.5, foreground="white")])
        handles.append(Line2D([0], [0], marker="o", color="none",
                              markerfacecolor=OUTCOME_COLORS[outcome], markeredgecolor="#111827",
                              markersize=8, label=outcome))
    map_ax.legend(
        handles=handles, loc="lower left", bbox_to_anchor=(1.03, 0.02),
        fontsize=8.5, borderaxespad=0,
    )
    map_ax.set_title("Representative Evaluation 2025 outcome locations", loc="left", fontsize=14, fontweight="bold", pad=12)
    map_ax.text(
        0, 1.005, "Mechanically selected by score closest to each outcome-group median",
        transform=map_ax.transAxes, fontsize=9.2, color="#4B5563",
    )

    for column, outcome in enumerate(case_order):
        ax = fig.add_subplot(grid[1, column])
        row = cases.loc[cases["Outcome"].eq(outcome)].iloc[0]
        pairs = [(row[f"Top{rank}_Feature"], float(row[f"Top{rank}_Signed_SHAP"])) for rank in range(1, 6)]
        pairs = list(reversed(pairs))
        names = [item[0] for item in pairs]
        values = np.array([item[1] for item in pairs])
        colors = np.where(values >= 0, "#C43C39", "#2474A6")
        ax.barh(names, values, color=colors)
        ax.axvline(0, color="#374151", lw=1)
        ax.grid(axis="x", alpha=0.2)
        ax.set_title(
            f"{outcome}: {row['Station']}\nscore {row['XGB_Final_Score']:.4f}",
            fontsize=11, fontweight="bold",
        )
        ax.set_xlabel("Signed SHAP\n(raw margin / log-odds)", fontsize=8.5)
        ax.tick_params(axis="y", labelsize=8.5)
        ax.text(
            0, -0.20, pd.Timestamp(row["Target_Datetime"]).strftime("Target %Y-%m-%d %H:%M"),
            transform=ax.transAxes, fontsize=8, color="#4B5563",
        )
    fig.suptitle("Spatial event case studies", x=0.08, y=0.985, ha="left", fontsize=16, fontweight="bold")
    fig.text(
        0.08, 0.012,
        "CRS: WGS 84 (EPSG:4326). Sources: frozen Evaluation 2025, Phase 7 SHAP, station metadata, and project boundary. "
        "Red contributions raise and blue contributions lower the model raw margin.",
        fontsize=7.8, color="#4B5563",
    )
    fig.subplots_adjust(left=0.08, right=0.96, top=0.94, bottom=0.15)
    fig.savefig(CASE_FIGURE, dpi=240, bbox_inches="tight")
    plt.close(fig)


def tied_extreme(frame: pd.DataFrame, column: str, highest: bool) -> tuple[float, list[str]]:
    values = frame[column].dropna()
    extreme = float(values.max() if highest else values.min())
    stations = frame.loc[np.isclose(frame[column], extreme, equal_nan=False), "Station"].astype(str).tolist()
    return extreme, stations


def station_list(stations: list[str], limit: int = 12) -> str:
    if len(stations) <= limit:
        return ", ".join(stations)
    return ", ".join(stations[:limit]) + f", and {len(stations) - limit} more"


def write_report(
    stations: pd.DataFrame,
    built: pd.DataFrame,
    cases: pd.DataFrame,
    shap_summary: pd.DataFrame,
    global_shap: pd.DataFrame,
    long_term: pd.DataFrame,
    metadata: pd.DataFrame,
) -> str:
    status = "PASS_WITH_CAVEATS" if "PASS_WITH_CAVEATS" in PHASE7_REPORT.read_text(encoding="utf-8") else "PASS"
    high_rate, high_rate_stations = tied_extreme(stations, "Def5_Rate_per_100k_rows", True)
    low_rate, low_rate_stations = tied_extreme(stations, "Def5_Rate_per_100k_rows", False)
    estimable = stations.loc[stations["Recall"].notna()]
    high_recall, high_recall_stations = tied_extreme(estimable, "Recall", True)
    low_recall, low_recall_stations = tied_extreme(estimable, "Recall", False)
    high_fp, high_fp_stations = tied_extreme(stations, "False_Positive_Rate", True)
    low_fp, low_fp_stations = tied_extreme(stations, "False_Positive_Rate", False)
    high_median, high_median_stations = tied_extreme(stations, "Median_XGB_Score", True)
    low_median, low_median_stations = tied_extreme(stations, "Median_XGB_Score", False)
    high_q95, high_q95_stations = tied_extreme(stations, "Q95_XGB_Score", True)
    low_q95, low_q95_stations = tied_extreme(stations, "Q95_XGB_Score", False)

    long_map = long_term.merge(
        metadata[["Station", "Longitude", "Latitude"]], on="Station", validate="one_to_one"
    )
    long_map["Def5_Rate_per_100k_rows"] = long_map["Def5_Rate"] * 100_000
    high_long, high_long_stations = tied_extreme(long_map, "Def5_Rate_per_100k_rows", True)
    low_long, low_long_stations = tied_extreme(long_map, "Def5_Rate_per_100k_rows", False)

    dominant_counts = Counter(shap_summary["Dominant_SHAP_Feature"])
    dominant_text = ", ".join(
        f"`{feature}` at {count} stations" for feature, count in dominant_counts.most_common()
    )
    top10 = ", ".join(f"`{x}`" for x in global_shap.sort_values("Rank_Mean_Abs_SHAP").head(10)["Feature"])

    built_lines = []
    for row in built.itertuples(index=False):
        built_lines.append(
            f"- **{row.Group_Label}:** {row.Number_of_Stations} stations, {row.Evaluation_Rows:,} rows, "
            f"{row.Actual_Def5_Count} Def5 rows ({row.Def5_Rate_per_100k_rows:.2f} per 100,000), "
            f"median/Q95 model scores {row.Median_XGB_Score:.4f}/{row.Q95_XGB_Score:.4f}, "
            f"recall {row.Recall:.3f}, false-positive rate {row.False_Positive_Rate:.3f}, "
            f"mean/median SUHI_5K {row.Mean_SUHI_5K:.3f}/{row.Median_SUHI_5K:.3f} °C."
        )
    case_lines = []
    for outcome in ["TP", "FP", "FN"]:
        row = cases.loc[cases["Outcome"].eq(outcome)].iloc[0]
        top = ", ".join(
            f"{row[f'Top{rank}_Feature']} ({row[f'Top{rank}_Signed_SHAP']:+.4f})"
            for rank in range(1, 6)
        )
        case_lines.append(
            f"- **{outcome}:** {row['Station']}, target {pd.Timestamp(row['Target_Datetime'])}, "
            f"actual {int(row['Actual'])}, prediction {int(row['Predicted_0p5'])}, "
            f"model score {row['XGB_Final_Score']:.6f}. Top signed contributions: {top}."
        )

    report = f"""# Phase 8 — Final Spatial Analysis and Geographic Interpretation

## 1. Spatial datasets used

The analysis reused frozen sources read-only:

- Final Evaluation: `03_OUTPUT/06_Final_Model_Evaluation/Evaluation2025_Actual_vs_Predicted_v1.parquet`
- Phase 7 SHAP rows and station summaries: `03_OUTPUT/07_SHAP_XAI/`
- Model-ready static predictors: `03_OUTPUT/05_Model_Comparison/ModelReady_Seasonal_Compact20_v1.parquet`
- Protected station metadata and coordinates: `{STATION_METADATA.as_posix()}`
- Context boundary: `{BOUNDARY.as_posix()}`. This valid WGS 84 polygon is the project-preserved, repaired dissolve of `Study_area/Study_area.shp` (20 source features, original CRS EPSG:32647).
- Frozen long-term station rates: `{LONG_TERM.as_posix()}`

No source was modified.

## 2. Station coverage

Evaluation 2025 contains {int(stations['Evaluation_Rows'].sum()):,} May–October station-hours, {int(stations['Actual_Def5_Count'].sum())} actual Def5 rows, and {len(stations)} mapped stations. Frozen metadata covers all 38 Evaluation stations. The boundary covers all 40 protected station coordinates. Evaluation maps show 38 active Evaluation stations; the long-term context map shows all 40 stations present in the frozen 2012–2025 summary.

## 3. Observed Evaluation 2025 Def5 spatial pattern

**Observation:** the highest station Def5 rate was {high_rate:.2f} per 100,000 Evaluation rows at {station_list(high_rate_stations)}. The lowest was {low_rate:.2f} at {station_list(low_rate_stations)}. These are station-hour observations with station-specific denominators, not estimates for unsampled locations.

## 4. Station model-score pattern

**Model behavior:** the highest median score was {high_median:.6f} at {station_list(high_median_stations)}, and the lowest was {low_median:.6f} at {station_list(low_median_stations)}. Q95 ranged from {low_q95:.6f} at {station_list(low_q95_stations)} to {high_q95:.6f} at {station_list(high_q95_stations)}. These are XGBoost model scores, not calibrated probabilities.

## 5. TP / FP / FN and recall spatial pattern

At the frozen diagnostic threshold 0.5, station counts reconcile to TP=71, FP=24,193, TN=97,008, and FN=24. Recall is estimable at {len(estimable)} stations and not estimable at {stations['Recall'].isna().sum()} stations with no actual positives. Among estimable stations, recall ranged from {low_recall:.3f} at {station_list(low_recall_stations)} to {high_recall:.3f} at {station_list(high_recall_stations)}. A station with no actual positive is shown as **not estimable**, never as recall zero.

## 6. False-positive spatial pattern

**Model behavior:** station false-positive rate ranged from {low_fp:.4f} at {station_list(low_fp_stations)} to {high_fp:.4f} at {station_list(high_fp_stations)}. These rates describe threshold-0.5 behavior among negative Evaluation rows; the threshold remains diagnostic.

## 7. Station SHAP patterns

**Model attribution:** the heatmap displays station-level mean |SHAP| for the global top ten predictors: {top10}. Values remain on the raw-margin/log-odds attribution scale. Spatial contrast in this heatmap describes how the frozen model used predictors across sampled stations.

## 8. Dominant SHAP features

The dominant station feature is the predictor with the largest station-level mean |SHAP|. Frequencies were {dominant_text}. This category is not a physical-driver label and does not establish a causal mechanism at a station.

## 9. Built-up vs Non-built-up descriptive comparison

The frozen mapping is `BuiltUp_Binary=1` for Built-up and `0` for Non-built-up. Pooled Evaluation summaries are:

{chr(10).join(built_lines)}

This comparison is descriptive. No inferential significance test was run, and spatial association does not establish a causal relationship between land-cover context and the target. `SUHI_5K` is the local LST anomaly relative to a 5 km rural background, not absolute urban temperature.

## 10. Long-term Def5 context

The frozen `Def5_Station_Dynamics_v1.csv` was available and reused without rebuilding climatology. It contains 1,123 events and 3,554,919 evaluable station-hours across 40 stations during 2012–2025. The highest long-term rate was {high_long:.2f} per 100,000 evaluable station-hours at {station_list(high_long_stations)}; the lowest was {low_long:.2f} at {station_list(low_long_stations)}. This all-period characterization is contextual and is distinct from the May–October Evaluation 2025 performance maps.

## 11. Representative event cases

Rows were selected mechanically as the score closest to the median within TP, FP, and FN. They were not chosen for scientific attractiveness.

{chr(10).join(case_lines)}

## 12. Geographic interpretation

- **Observation** refers only to Def5 counts and denominator-adjusted rates at sampled stations.
- **Model behavior** refers to frozen XGBoost scores and threshold-0.5 errors at sampled stations.
- **Model attribution** refers to SHAP contributions to the frozen model's raw margin.

These three evidence types are kept separate. Maps show station-level values only and do not represent conditions between stations or outside the network.

## 13. Limitations

- No interpolation was performed.
- No gridded risk surface was created.
- All maps show station-level point values with an existing basin polygon used only as context.
- Station values do not represent unsampled areas.
- SHAP attribution is not causation.
- Threshold 0.5 remains diagnostic and was not optimized.
- Model scores are not calibrated probabilities.
- Recall is undefined at stations without actual positives; these stations remain visibly not estimable.
- Rare station-level event counts make rates and recall unstable, especially over one Evaluation season.
- The long-term context spans a different period and seasonal scope from Evaluation 2025 and is not a direct performance comparator.
- Phase 7 carried a small native float32 raw-margin reconstruction rounding caveat; Phase 8 reused its frozen SHAP values without alteration.

## STATUS

{status}
"""
    REPORT.write_text(report, encoding="utf-8")
    return status


def main() -> None:
    existing = [str(path) for path in OUTPUTS if path.exists()]
    if existing:
        raise RuntimeError(f"Refusing to overwrite existing Phase 8 map/report output(s): {existing}")
    verify_inputs()
    stations, built, cases, shap_long, shap_summary, global_shap, boundary, long_term = load_inputs()
    metadata = pd.read_csv(STATION_METADATA)

    source_eval = "Sources: frozen Evaluation 2025; protected station metadata; project boundary. Point values only; no interpolation."
    save_continuous_map(
        stations, boundary, "Def5_Rate_per_100k_rows", DEF5_MAP,
        "Observed Def5 rate by station — Evaluation 2025",
        "May–October 2025; denominator is each station's Evaluation rows",
        "Def5 rows per 100,000 Evaluation rows", "YlOrRd", source_eval, vmin=0,
    )
    save_continuous_map(
        stations, boundary, "Recall", RECALL_MAP,
        "Recall by station — diagnostic threshold 0.5",
        "Stations without actual positives are shown as not estimable",
        "Recall", "viridis", source_eval, vmin=0, vmax=1,
        missing_label="Not estimable (no actual Def5)",
    )
    save_continuous_map(
        stations, boundary, "False_Positive_Rate", FP_MAP,
        "False-positive rate by station — threshold 0.5",
        "False positives divided by negative Evaluation rows",
        "False-positive rate", "YlOrRd", source_eval, vmin=0,
    )
    save_continuous_map(
        stations, boundary, "Median_XGB_Score", MEDIAN_MAP,
        "Median final model score by station",
        "XGBoost model score; not a calibrated probability",
        "Median XGBoost score", "viridis", source_eval, vmin=0, vmax=1,
    )
    save_continuous_map(
        stations, boundary, "Q95_XGB_Score", Q95_MAP,
        "Q95 final model score by station",
        "95th percentile XGBoost score; not a calibrated probability",
        "Q95 XGBoost score", "viridis", source_eval, vmin=0, vmax=1,
    )
    save_dominant_map(stations, shap_summary, boundary)
    save_shap_heatmap(stations, shap_long, global_shap)

    long_map = long_term.merge(
        metadata[["Station", "Longitude", "Latitude"]], on="Station", validate="one_to_one"
    )
    long_map["Def5_Rate_per_100k_rows"] = long_map["Def5_Rate"] * 100_000
    save_continuous_map(
        long_map, boundary, "Def5_Rate_per_100k_rows", LONG_TERM_MAP,
        "Long-term Def5 rate context by station, 2012–2025",
        "Frozen all-period characterization; distinct from May–October Evaluation 2025",
        "Def5 events per 100,000 evaluable station-hours", "YlOrRd",
        "Sources: frozen Def5_Station_Dynamics_v1.csv; protected station metadata; project boundary. No interpolation.",
        vmin=0,
    )
    save_case_figure(stations, cases, boundary)
    status = write_report(stations, built, cases, shap_summary, global_shap, long_term, metadata)

    missing = [str(path) for path in OUTPUTS if not path.exists()]
    if missing:
        raise RuntimeError(f"One or more required Phase 8 outputs were not produced: {missing}")
    print(f"STATUS: {status}")
    print(f"Mapped Evaluation stations: {len(stations)}")
    print(f"Long-term context stations: {len(long_term)}")
    print("Dominant SHAP features: " + str(Counter(shap_summary["Dominant_SHAP_Feature"])))


if __name__ == "__main__":
    main()
