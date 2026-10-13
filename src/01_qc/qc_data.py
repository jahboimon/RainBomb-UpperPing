"""Phase 1 data-period QC using frozen legacy inputs only.

This script reads the authoritative dynamic rainfall panel, current RH and
temperature canonical tables, and the frozen Def5 target/evaluable base. It
writes only the requested Phase 1 outputs in RainBomb_FinalAnalysis_v2.
"""

from __future__ import annotations
import sys as _sys
from pathlib import Path as _ConfigPath
_sys.path.insert(0, str(_ConfigPath(__file__).resolve().parents[1]))
from rainbomb_paths import load_paths as _load_paths
PATHS = _load_paths()

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LEGACY = PATHS.upstream_root
ROOT = PATHS.runtime_root
OUT = ROOT / "03_OUTPUT" / "01_DataPeriod_QC"
INPUT_NOTE = ROOT / "01_INPUT_REFERENCES" / "FROZEN_INPUTS.md"

RAIN = LEGACY / "Rainfall_Canonical_2012_2025_Final_v1" / "PROCESS_Hourly_Rain_Canonical_2012_2025_v1.parquet"
RAIN_QC = LEGACY / "Rainfall_Canonical_2012_2025_Final_v1" / "PROCESS_Hourly_Rain_Canonical_2012_2025_StationYear_QC_v1.csv"
RH = LEGACY / "Humidity_Canonical_Final_v4" / "PROCESS_Hourly_Humidity_Canonical_v4.csv"
RH_QC = LEGACY / "Humidity_Canonical_Final_v4" / "humidity_canonical_v4_station_year_qc.csv"
TEMP = LEGACY / "Temperature_Canonical_Final_v1" / "PROCESS_Hourly_Temperature_Canonical_v1.csv"
TEMP_QC = LEGACY / "Temperature_Canonical_Final_v1" / "temperature_canonical_v1_station_year_qc.csv"
STATIONS = LEGACY / "Hourly_rain" / "RESETWORKPLACE" / "02_processed" / "Station_Metadata_Cleaned.csv"
DEF5_EVENTS = LEGACY / "Def5_Final_Target_2012_2025_v1" / "02_Def5_Extended_Target" / "RainBomb_Target_Def5_2012_2025_v1.parquet"
DEF5_QC = LEGACY / "Def5_Final_Target_2012_2025_v1" / "02_Def5_Extended_Target" / "Def5_StationYear_QC_v1.csv"
DEF5_T1 = LEGACY / "Def5_Final_Target_2012_2025_v1" / "04_Def5_Plus1h_Target" / "ML_Base_StationHour_Target_Def5_t1_2012_2025_v1.parquet"

SELECTED_INPUTS = [RAIN, RAIN_QC, RH, RH_QC, TEMP, TEMP_QC, STATIONS, DEF5_EVENTS, DEF5_QC, DEF5_T1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_value_csv(path: Path, value_col: str) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=["Station", "Datetime", value_col],
        parse_dates=["Datetime"],
        dtype={"Station": "category"},
        low_memory=False,
    )
    frame = frame.rename(columns={value_col: value_col.lower(), "Datetime": "datetime"})
    frame["key_present"] = True
    assert not frame.duplicated(["Station", "datetime"]).any()
    return frame


def add_year(frame: pd.DataFrame, datetime_col: str = "datetime") -> pd.DataFrame:
    frame["year"] = frame[datetime_col].dt.year.astype("int16")
    return frame


def pct(numerator: pd.Series | float, denominator: pd.Series | float):
    numerator_array = np.asarray(numerator, dtype="float64")
    denominator_array = np.asarray(denominator, dtype="float64")
    result = np.full(np.broadcast_shapes(numerator_array.shape, denominator_array.shape), np.nan, dtype="float64")
    return np.divide(numerator_array * 100.0, denominator_array, out=result, where=denominator_array > 0)


def summarize_observation_panel(panel: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = panel.groupby(group_cols, observed=True, sort=True)
    summary = grouped.agg(
        available_station_hours=("datetime", "size"),
        valid_rainfall_observations=("rain_valid", "sum"),
        missing_rainfall_observations=("rain_missing", "sum"),
        rh_valid_hours=("rh_valid", "sum"),
        rh_observational_missing_hours=("rh_observational_missing", "sum"),
        rh_structural_absence_hours=("rh_structural_absence", "sum"),
        temperature_valid_hours=("temperature_valid", "sum"),
        temperature_observational_missing_hours=("temperature_observational_missing", "sum"),
        temperature_structural_absence_hours=("temperature_structural_absence", "sum"),
        complete_predictor_station_hours=("complete_predictor", "sum"),
    ).reset_index()
    denominator = summary["available_station_hours"]
    summary["rainfall_valid_pct"] = pct(summary["valid_rainfall_observations"], denominator)
    summary["rainfall_missing_pct"] = pct(summary["missing_rainfall_observations"], denominator)
    summary["rh_valid_pct"] = pct(summary["rh_valid_hours"], denominator)
    summary["rh_observational_missing_pct"] = pct(summary["rh_observational_missing_hours"], denominator)
    summary["rh_structural_absence_pct"] = pct(summary["rh_structural_absence_hours"], denominator)
    summary["temperature_valid_pct"] = pct(summary["temperature_valid_hours"], denominator)
    summary["temperature_observational_missing_pct"] = pct(summary["temperature_observational_missing_hours"], denominator)
    summary["temperature_structural_absence_pct"] = pct(summary["temperature_structural_absence_hours"], denominator)
    summary["complete_predictor_pct"] = pct(summary["complete_predictor_station_hours"], denominator)
    return summary


def summarize_modeling(model: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = model.groupby(group_cols, observed=True, sort=True)
    result = grouped.agg(
        def5_tplus1_evaluable_station_hours=("Target_Def5_tPlus1", "size"),
        usable_station_hours_for_later_modeling=("complete_predictor", "sum"),
        usable_def5_positive_count=("usable_positive", "sum"),
    ).reset_index()
    result["usable_modeling_pct_of_def5_evaluable"] = pct(
        result["usable_station_hours_for_later_modeling"], result["def5_tplus1_evaluable_station_hours"]
    )
    return result


def summarize_def5_qc(group_cols: list[str]) -> pd.DataFrame:
    qc = pd.read_csv(DEF5_QC)
    qc = qc.rename(columns={"Station": "station", "Year": "year"})
    qc["is_active"] = qc["Source_Rows"].gt(0)
    grouped = qc.groupby(group_cols, observed=True, sort=True)
    result = grouped.agg(
        expected_station_hours_dynamic=("Source_Rows", "sum"),
        def5_evaluable_station_hours=("Def5_Eligible_Hours", "sum"),
        def5_positive_count=("Def5_Events", "sum"),
    ).reset_index()
    result["def5_positive_rate_pct"] = pct(result["def5_positive_count"], result["def5_evaluable_station_hours"])
    return result


def period_mask(series: pd.Series, start: str, end: str) -> pd.Series:
    return series.between(pd.Timestamp(start), pd.Timestamp(end), inclusive="both")


def aggregate_period(
    label: str,
    start: str,
    end: str,
    panel: pd.DataFrame,
    model: pd.DataFrame,
    events: pd.DataFrame,
) -> dict:
    obs = panel.loc[period_mask(panel["datetime"], start, end)]
    mod = model.loc[period_mask(model["Target_Datetime"], start, end)]
    evt = events.loc[period_mask(events["Datetime"], start, end)]

    available = len(obs)
    evaluable = len(mod)
    row = {
        "period": label,
        "period_start": start,
        "period_end": end,
        "stations_with_available_hours": obs["station"].nunique(),
        "available_station_hours": available,
        "valid_rainfall_observations": int(obs["rain_valid"].sum()),
        "missing_rainfall_observations": int(obs["rain_missing"].sum()),
        "rainfall_valid_pct": float(pct(obs["rain_valid"].sum(), available)),
        "rainfall_missing_pct": float(pct(obs["rain_missing"].sum(), available)),
        "rh_valid_pct": float(pct(obs["rh_valid"].sum(), available)),
        "rh_observational_missing_pct": float(pct(obs["rh_observational_missing"].sum(), available)),
        "rh_structural_absence_pct": float(pct(obs["rh_structural_absence"].sum(), available)),
        "temperature_valid_pct": float(pct(obs["temperature_valid"].sum(), available)),
        "temperature_observational_missing_pct": float(pct(obs["temperature_observational_missing"].sum(), available)),
        "temperature_structural_absence_pct": float(pct(obs["temperature_structural_absence"].sum(), available)),
        "complete_predictor_station_hours": int(obs["complete_predictor"].sum()),
        "complete_predictor_pct": float(pct(obs["complete_predictor"].sum(), available)),
        "def5_evaluable_station_hours": evaluable,
        "usable_station_hours_for_later_modeling": int(mod["complete_predictor"].sum()),
        "usable_modeling_pct_of_def5_evaluable": float(pct(mod["complete_predictor"].sum(), evaluable)),
        "def5_positive_count": int(len(evt)),
        "def5_positive_rate_pct": float(pct(len(evt), evaluable)),
    }
    return row


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D9E1E8", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def create_figures(annual: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.titlesize": 13})
    years = annual["year"]
    blue, teal, orange, purple = "#2F5597", "#279E8B", "#E07A3F", "#7A5195"

    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    for col, label, color in [
        ("rainfall_valid_pct", "Rainfall", blue),
        ("rh_valid_pct", "Relative humidity", teal),
        ("temperature_valid_pct", "Temperature", orange),
        ("complete_predictor_pct", "Rain + RH + temperature", purple),
    ]:
        ax.plot(years, annual[col], marker="o", linewidth=2, markersize=4, label=label, color=color)
    ax.axvline(2018, color="#6B7280", linestyle="--", linewidth=1, label="2018 reference")
    ax.set(title="Annual data completeness within the active rainfall network", xlabel="Year", ylabel="Valid station-hours (%)")
    ax.set_ylim(0, 103)
    ax.set_xticks(years)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    style_axes(ax)
    fig.savefig(OUT / "Annual_Data_Completeness_v1.png", dpi=220, facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    ax.bar(years, annual["active_stations"], color=blue, width=0.75)
    ax.axvline(2018, color="#6B7280", linestyle="--", linewidth=1)
    ax.set(title="Active rainfall stations by year", xlabel="Year", ylabel="Stations with source rows")
    ax.set_xticks(years)
    ax.set_ylim(0, 43)
    style_axes(ax)
    fig.savefig(OUT / "Annual_Active_Stations_v1.png", dpi=220, facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    ax.bar(years, annual["usable_station_hours_for_later_modeling"], color=teal, width=0.75)
    ax.axvline(2018, color="#6B7280", linestyle="--", linewidth=1)
    ax.set(title="Station-hours usable for later modeling", xlabel="Predictor year", ylabel="Complete predictors with evaluable Def5 at t+1")
    ax.set_xticks(years)
    ax.ticklabel_format(style="plain", axis="y")
    style_axes(ax)
    fig.savefig(OUT / "Annual_Usable_Modeling_Hours_v1.png", dpi=220, facecolor="white")
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    ax1.bar(years, annual["def5_positive_count"], color=blue, width=0.72, label="Def5 count")
    ax1.set(xlabel="Event year", ylabel="Def5 positive count", title="Annual Def5 count and evaluable-hour rate")
    ax1.set_xticks(years)
    style_axes(ax1)
    ax2 = ax1.twinx()
    ax2.plot(years, annual["def5_positive_rate_pct"], color=orange, marker="o", linewidth=2, label="Def5 rate")
    ax2.set_ylabel("Def5 positives among evaluable hours (%)")
    ax2.spines["top"].set_visible(False)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, frameon=False, loc="upper left")
    fig.savefig(OUT / "Annual_Def5_Count_or_Rate_v1.png", dpi=220, facecolor="white")
    plt.close(fig)


def fmt(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "not available"
    return f"{value:.{digits}f}%"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in SELECTED_INPUTS:
        assert path.is_file(), f"Missing frozen input: {path}"
    hashes_before = {str(path): sha256(path) for path in SELECTED_INPUTS}

    rain = pd.read_parquet(RAIN, columns=["Station", "Datetime", "Rainfall"])
    rain = rain.rename(columns={"Station": "station", "Datetime": "datetime", "Rainfall": "rainfall"})
    rain["station"] = rain["station"].astype("category")
    add_year(rain)
    assert len(rain) == 4_393_680
    assert rain["station"].nunique() == 40
    assert not rain.duplicated(["station", "datetime"]).any()

    rh = read_value_csv(RH, "Humidity").rename(columns={"Station": "station", "key_present": "rh_key_present"})
    temp = read_value_csv(TEMP, "Temperature").rename(columns={"Station": "station", "key_present": "temperature_key_present"})

    panel = rain.merge(rh, how="left", on=["station", "datetime"], validate="one_to_one")
    del rh
    panel = panel.merge(temp, how="left", on=["station", "datetime"], validate="one_to_one")
    del temp

    panel["rain_valid"] = panel["rainfall"].notna()
    panel["rain_missing"] = panel["rainfall"].isna()
    panel["rh_valid"] = panel["humidity"].notna()
    panel["rh_structural_absence"] = panel["rh_key_present"].isna()
    panel["rh_observational_missing"] = panel["rh_key_present"].eq(True) & panel["humidity"].isna()
    panel["temperature_valid"] = panel["temperature"].notna()
    panel["temperature_structural_absence"] = panel["temperature_key_present"].isna()
    panel["temperature_observational_missing"] = panel["temperature_key_present"].eq(True) & panel["temperature"].isna()
    panel["complete_predictor"] = panel[["rainfall", "humidity", "temperature"]].notna().all(axis=1)

    # Frozen Def5 t+1 base supplies the valid target/evaluable universe. No label is rebuilt.
    t1 = pd.read_parquet(DEF5_T1)
    t1 = t1.rename(columns={"Station": "station", "Datetime": "datetime"})
    t1["station"] = t1["station"].astype("category")
    model = t1.merge(
        panel[["station", "datetime", "complete_predictor"]],
        how="left",
        on=["station", "datetime"],
        validate="one_to_one",
    )
    assert model["complete_predictor"].notna().all()
    model["usable_positive"] = model["complete_predictor"] & model["Target_Def5_tPlus1"].eq(1)
    model["year"] = model["datetime"].dt.year.astype("int16")

    events = pd.read_parquet(DEF5_EVENTS)
    assert len(events) == 1_123 and events["Def5"].eq(1).all()
    events["year"] = events["Datetime"].dt.year.astype("int16")

    station_year_observed = summarize_observation_panel(panel, ["station", "year"])
    active = station_year_observed[["station", "year", "available_station_hours"]].copy()
    modeling_sy = summarize_modeling(model, ["station", "year"])
    def5_station_year = pd.read_csv(DEF5_QC).rename(columns={"Station": "station", "Year": "year"})
    station_year = def5_station_year[[
        "station", "year", "Availability_Status", "Source_Rows", "Def5_Eligible_Hours", "Def5_Events"
    ]].rename(columns={
        "Source_Rows": "expected_station_hours_dynamic",
        "Def5_Eligible_Hours": "def5_evaluable_station_hours",
        "Def5_Events": "def5_positive_count",
    })
    station_year["structurally_unavailable_station_year"] = station_year["expected_station_hours_dynamic"].eq(0)
    station_year = station_year.merge(station_year_observed, how="left", on=["station", "year"], validate="one_to_one")
    station_year = station_year.merge(modeling_sy, how="left", on=["station", "year"], validate="one_to_one")
    count_cols = [
        "def5_tplus1_evaluable_station_hours", "usable_station_hours_for_later_modeling",
        "usable_def5_positive_count", "expected_station_hours_dynamic", "available_station_hours",
        "valid_rainfall_observations", "missing_rainfall_observations", "rh_valid_hours",
        "rh_observational_missing_hours", "rh_structural_absence_hours", "temperature_valid_hours",
        "temperature_observational_missing_hours", "temperature_structural_absence_hours",
        "complete_predictor_station_hours", "def5_evaluable_station_hours", "def5_positive_count",
    ]
    station_year[count_cols] = station_year[count_cols].fillna(0).astype("int64")
    station_year["def5_positive_rate_pct"] = pct(station_year["def5_positive_count"], station_year["def5_evaluable_station_hours"])
    station_year["usable_modeling_pct_of_def5_evaluable"] = pct(
        station_year["usable_station_hours_for_later_modeling"], station_year["def5_tplus1_evaluable_station_hours"]
    )
    station_year = station_year.rename(columns={"station": "Station", "year": "Year"})

    annual = summarize_observation_panel(panel, ["year"])
    annual_active = active.groupby("year", observed=True).agg(active_stations=("station", "nunique")).reset_index()
    annual_modeling = summarize_modeling(model, ["year"])
    annual_def5 = summarize_def5_qc(["year"])
    annual = annual.merge(annual_active, on="year").merge(annual_modeling, on="year", how="left").merge(annual_def5, on="year", how="left")
    annual["full_40_station_hour_reference"] = annual["year"].map(lambda y: (8784 if y in (2012, 2016, 2020, 2024) else 8760) * 40)
    annual["structural_station_hours_absent_vs_40_reference"] = annual["full_40_station_hour_reference"] - annual["available_station_hours"]
    annual = annual.sort_values("year")

    period_rows = [
        aggregate_period("2012-2017", "2012-01-01 00:00:00", "2017-12-31 23:00:00", panel, model, events),
        aggregate_period("2013-2017 (RH-supported)", "2013-01-01 00:00:00", "2017-12-31 23:00:00", panel, model, events),
        aggregate_period("2018-2024H1", "2018-01-01 00:00:00", "2024-06-30 23:00:00", panel, model, events),
        aggregate_period("2024H2", "2024-07-01 00:00:00", "2024-12-31 23:00:00", panel, model, events),
        aggregate_period("2025", "2025-01-01 00:00:00", "2025-12-31 23:00:00", panel, model, events),
    ]
    period_comparison = pd.DataFrame(period_rows)

    # Stable, machine-readable column order.
    annual_order = [
        "year", "active_stations", "full_40_station_hour_reference", "expected_station_hours_dynamic",
        "available_station_hours", "structural_station_hours_absent_vs_40_reference",
        "valid_rainfall_observations", "missing_rainfall_observations", "rainfall_valid_pct", "rainfall_missing_pct",
        "rh_valid_hours", "rh_observational_missing_hours", "rh_structural_absence_hours", "rh_valid_pct",
        "rh_observational_missing_pct", "rh_structural_absence_pct", "temperature_valid_hours",
        "temperature_observational_missing_hours", "temperature_structural_absence_hours", "temperature_valid_pct",
        "temperature_observational_missing_pct", "temperature_structural_absence_pct",
        "complete_predictor_station_hours", "complete_predictor_pct", "def5_tplus1_evaluable_station_hours",
        "usable_station_hours_for_later_modeling", "usable_modeling_pct_of_def5_evaluable",
        "def5_evaluable_station_hours", "def5_positive_count", "def5_positive_rate_pct",
    ]
    annual = annual[annual_order]
    station_year = station_year.sort_values(["Station", "Year"])

    annual.to_csv(OUT / "DataPeriod_QC_Annual_v1.csv", index=False, float_format="%.6f")
    station_year.to_csv(OUT / "DataPeriod_QC_StationYear_v1.csv", index=False, float_format="%.6f")
    period_comparison.to_csv(OUT / "DataPeriod_QC_PeriodComparison_v1.csv", index=False, float_format="%.6f")
    create_figures(annual)

    early = period_comparison.loc[period_comparison["period"].eq("2012-2017")].iloc[0]
    early_rh_supported = period_comparison.loc[period_comparison["period"].eq("2013-2017 (RH-supported)")].iloc[0]
    later = period_comparison.loc[period_comparison["period"].eq("2018-2024H1")].iloc[0]
    # Largest annual rise is descriptive evidence, not a selected breakpoint.
    annual_eval = annual.loc[annual["year"].ge(2013)].copy()
    rise_years = {}
    for col in ["rainfall_valid_pct", "rh_valid_pct", "temperature_valid_pct", "complete_predictor_pct"]:
        diff = annual_eval.set_index("year")[col].diff()
        rise_years[col] = (int(diff.idxmax()), float(diff.max()))

    input_note = f"""# Frozen Inputs for Phase 1 Data Period QC

All paths below are read-only legacy inputs. No large source dataset was copied into the new workspace.

| Frozen input | Contents | Use in this audit | Coverage / stations |
|---|---|---|---|
| `{RAIN.as_posix()}` | Official hourly rainfall canonical with `Station`, `Datetime`, and nullable rainfall. | Defines the dynamic station-hour universe and rainfall validity. | 2012-2025; 40 stations overall; 4,393,680 dynamic-panel rows. |
| `{RAIN_QC.as_posix()}` | Frozen rainfall station-year QC and availability status. | Cross-checks annual active-station structure and observed missingness. | 2012-2025; 560 station-years including structural-zero rows. |
| `{RH.as_posix()}` | Recommended cleaned hourly relative-humidity canonical. | Measures RH validity and simultaneous predictor completeness. | 2013-2025; 40 stations overall; dynamic gaps preserved. |
| `{RH_QC.as_posix()}` | Frozen RH station-year QC. | Cross-checks RH coverage and structural gaps. | 2013-2025; 520 station-years. |
| `{TEMP.as_posix()}` | Recommended cleaned hourly temperature canonical. | Measures temperature validity and simultaneous predictor completeness. | 2012-2025; 40 stations overall. |
| `{TEMP_QC.as_posix()}` | Frozen temperature station-year QC. | Cross-checks temperature completeness. | 2012-2025; 560 station-years. |
| `{STATIONS.as_posix()}` | Protected cleaned station metadata. | Confirms the official 40-station universe and station identifiers. | 40 unique stations. |
| `{DEF5_EVENTS.as_posix()}` | Frozen Def5 positive-event keys and event attributes. | Supplies Def5 positive counts only; Def5 is not recomputed. | 2012-2025; 1,123 positive events; 40 stations overall. |
| `{DEF5_QC.as_posix()}` | Frozen Def5 station-year eligibility summary. | Supplies evaluable denominators, source-row counts, and station-year Def5 counts. | 2012-2025; 560 station-years. |
| `{DEF5_T1.as_posix()}` | Frozen station-hour base with evaluable Def5 at `t+1`. | Counts station-hours that have both complete current predictors and an evaluable later target. | 2012-2025; 3,554,919 evaluable rows. |

## Selection notes

- The official rainfall dynamic panel is the denominator. A balanced 40-station grid was not manufactured.
- RH has no frozen canonical coverage in 2012. Those hours are recorded as structural variable absence, not ordinary observed missing values.
- The frozen Def5 target and evaluable base are reused exactly. Obsolete Def1-Def4 target outputs are not used as the scientific basis.
- The large ML master and any model/Test metrics are not needed for this descriptive audit and were not used to choose a period.
"""
    INPUT_NOTE.write_text(input_note, encoding="utf-8")

    report = f"""# Data Period QC Report v1

## Scope and frozen inputs

This audit describes observational coverage from 2012 through 2025 using the frozen rainfall, RH, temperature, station metadata, and Def5 artifacts listed in `01_INPUT_REFERENCES/FROZEN_INPUTS.md`. It does not rebuild any canonical data or target.

## What was measured

For each year and station-year, the audit counts active source hours, valid and missing rainfall, valid RH, valid temperature, simultaneous completeness of rainfall + RH + temperature, frozen Def5-evaluable hours, and frozen Def5 positives. A station-hour is considered usable for later modeling only when the three current-hour predictors are all present and the frozen Def5 target at `t+1` is evaluable. This is an availability count, not a selected feature set or trained model.

## Structural station absence

The official rainfall canonical is a dynamic panel. The audit uses only station-hours that physically exist in that source. It does not create rows for the six stations without source rows in 2012-2019, the two without source rows in 2020-2021, FOP053 after 2024H1, or CHM001 in 2025. Missing values inside an active station period remain observational missingness and are never replaced with zero.

RH has no frozen canonical records for 2012. The annual and station-year tables therefore show RH structural absence separately from RH observational missingness. Temperature and RH key absence are also reported separately wherever it occurs.

## Annual pattern

Rainfall validity was {fmt(early['rainfall_valid_pct'])} in 2012-2017 and {fmt(later['rainfall_valid_pct'])} in 2018-2024H1. Simultaneous rain + RH + temperature completeness was {fmt(early['complete_predictor_pct'])} in the earlier period and {fmt(later['complete_predictor_pct'])} in the later period. The 2012 early-period result includes a complete structural absence of RH because the frozen RH canonical begins in 2013. When 2012 is excluded so both sides have RH source coverage, complete predictor availability was still only {fmt(early_rh_supported['complete_predictor_pct'])} in 2013-2017.

The active rainfall network expanded from 34 stations in 2012-2019 to 38 in 2020-2021 and 40 in 2022-2023. It then contracted to 39 in 2024H2 and 38 in 2025 because the frozen source contains no later rows for FOP053 and no 2025 rows for CHM001.

The largest positive year-to-year changes after RH coverage begins were:

- rainfall validity: {rise_years['rainfall_valid_pct'][0]} ({rise_years['rainfall_valid_pct'][1]:.1f} percentage points)
- RH validity: {rise_years['rh_valid_pct'][0]} ({rise_years['rh_valid_pct'][1]:.1f} percentage points)
- temperature validity: {rise_years['temperature_valid_pct'][0]} ({rise_years['temperature_valid_pct'][1]:.1f} percentage points)
- complete predictor availability: {rise_years['complete_predictor_pct'][0]} ({rise_years['complete_predictor_pct'][1]:.1f} percentage points)

These largest changes do not all occur in the same year, so they do not establish one objective cutoff by themselves.

## Is 2018 meaningfully different?

The later period is clearly more complete in aggregate, especially for rainfall and simultaneous predictor availability. However, the annual series should not be described as a single clean step at 2018. Rainfall improves in 2018 and again in 2019-2021, while changes in RH and temperature completeness follow their own station-year missingness patterns. The station network itself expands in 2020 and 2022. Thus, 2018 is a plausible practical reference point, but the evidence supports a gradual and multi-stage improvement rather than proving an abrupt 2018 breakpoint.

## Other plausible breakpoints

The strongest joint-data inflection is 2019: RH validity rises by {rise_years['rh_valid_pct'][1]:.1f} percentage points, temperature validity by {rise_years['temperature_valid_pct'][1]:.1f} points, and complete predictor availability by {rise_years['complete_predictor_pct'][1]:.1f} points from 2018. The year 2020 is another plausible network-consistency reference because rainfall missingness drops further and the active network expands from 34 to 38 stations. The network reaches 40 stations in 2022. These later starts improve consistency but shorten the development record and reduce the number of rare Def5 positives available for later analysis. No breakpoint is selected here.

## Limitations

- The variables have different frozen coverage. RH begins in 2013, so 2012 cannot support a complete three-predictor station-hour.
- Network growth changes both the number and identity of stations. A higher annual count is not purely a sensor-quality effect.
- Annual totals combine seasonal conditions and station-specific outages. The station-year CSV should be inspected before excluding an entire calendar period.
- Def5 rates use frozen evaluable hours, not all possible 40-station hours. Rare-event counts can vary because of meteorology as well as data availability.
- This audit checks current-hour rain, RH, and temperature availability. It does not authorize a final lag window, feature set, LULC representation, seasonal gate, or model.

## Evidence for the researcher's modeling-period decision

The researcher should review four pieces together: the annual completeness lines, station-year outliers, usable modeling hours, and retained Def5 counts/rates. A defensible choice should state the minimum acceptable completeness, whether consistent station representation or a longer historical record matters more, and how many Def5 positives remain after the chosen start date. Sensitivity across more than one plausible start year may be appropriate later, but no such modeling experiment is performed here.

## HUMAN DECISION REQUIRED

- No modeling-period cutoff has been selected.
- No model has been trained.
- No model metric has been used.
- The researcher must inspect the QC evidence and decide the modeling period.
"""
    (OUT / "DataPeriod_QC_Report_v1.md").write_text(report, encoding="utf-8")

    hashes_after = {str(path): sha256(path) for path in SELECTED_INPUTS}
    assert hashes_before == hashes_after, "A frozen input changed during the audit."

    # Output integrity assertions.
    assert annual["year"].tolist() == list(range(2012, 2026))
    assert annual["available_station_hours"].sum() == len(panel)
    assert annual["valid_rainfall_observations"].sum() == 3_713_707
    assert annual["missing_rainfall_observations"].sum() == 679_973
    assert annual["def5_positive_count"].sum() == 1_123
    assert annual["def5_evaluable_station_hours"].sum() == 3_554_919
    assert len(station_year) == 560
    assert station_year.duplicated(["Station", "Year"]).sum() == 0
    print(annual.to_string(index=False))
    print("\nPeriod comparison\n", period_comparison.to_string(index=False))
    print("\nFrozen input hashes unchanged: PASS")


if __name__ == "__main__":
    main()
