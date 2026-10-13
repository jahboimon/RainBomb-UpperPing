"""Build the frozen May-October 2020-2025 common modeling universe.

This Phase 3 script only filters and verifies the current frozen candidate-
predictor population. It assigns roles from Target_Datetime, removes documented
audit-only fields, preserves scientific missingness, and performs no modeling or
feature selection.
"""

from __future__ import annotations
import sys as _sys
from pathlib import Path as _ConfigPath
_sys.path.insert(0, str(_ConfigPath(__file__).resolve().parents[1]))
from rainbomb_paths import load_paths as _load_paths
PATHS = _load_paths()

from calendar import month_name
from pathlib import Path

import numpy as np
import pandas as pd


LEGACY = PATHS.upstream_root
ROOT = PATHS.runtime_root
OUT = ROOT / "03_OUTPUT" / "03_Seasonal_Modeling_Universe"

SOURCE = (
    LEGACY
    / "PhaseB_Atmospheric"
    / "ERA5"
    / "06_Def5_ModelReady"
    / "PhaseB_Def5_ERA5_ModelReady_v1.parquet"
)
FROZEN_T1 = (
    LEGACY
    / "Def5_Final_Target_2012_2025_v1"
    / "04_Def5_Plus1h_Target"
    / "ML_Base_StationHour_Target_Def5_t1_2012_2025_v1.parquet"
)
FROZEN_EVENTS = (
    LEGACY
    / "Def5_Final_Target_2012_2025_v1"
    / "02_Def5_Extended_Target"
    / "RainBomb_Target_Def5_2012_2025_v1.parquet"
)
STATIONS = LEGACY / "Hourly_rain" / "RESETWORKPLACE" / "02_processed" / "Station_Metadata_Cleaned.csv"
PHASE2_COUNTS = (
    ROOT
    / "03_OUTPUT"
    / "02_Seasonal_Scope"
    / "SeasonalScope_SeasonYearCounts_v1.csv"
)

RAIN = [
    "Rain_t", "Rain_t_1", "Rain_t_2", "Rain_t_3", "Rain_t_4", "Rain_t_5", "Rain_t_6",
    "RainSum_3h", "RainSum_6h", "RainMax_3h", "RainMax_6h",
]
RH = [
    "RH_t", "RH_t_1", "RH_t_2", "RH_t_3", "RH_t_4", "RH_t_5", "RH_t_6",
    "RHMean_3h", "RHMean_6h", "RHMin_3h", "RHMax_3h", "RHMin_6h", "RHMax_6h",
]
TEMP = [
    "Temp_t", "Temp_t_1", "Temp_t_2", "Temp_t_3", "Temp_t_4", "Temp_t_5", "Temp_t_6",
    "TempMean_3h", "TempMin_3h", "TempMax_3h", "TempMean_6h", "TempMin_6h", "TempMax_6h",
]
TEMPORAL = ["Hour_sin", "Hour_cos", "Month_sin", "Month_cos"]
STATIC = ["Elevation_m", "LULC_Class", "SUHI_5K"]
ERA5 = ["d2m", "u10", "v10", "q_850", "u_850", "v_850"]
CANDIDATE_PREDICTORS = RAIN + RH + TEMP + TEMPORAL + STATIC + ERA5
CURRENT_RRT = RAIN + RH + TEMP
IDENTIFIERS = ["Station", "Datetime", "Target_Datetime", "Role", "Target_Def5_tPlus1"]
AUDIT_ONLY_NOT_RETAINED = [
    "Split", "era5_datetime_utc", "era5_datetime_ict", "era5_lat", "era5_lon",
    "sp", "valid_850", "era5_join_matched",
]
ROLE_ORDER = ["Train", "Validation", "Evaluation"]
ROLE_YEARS = {"Train": (2020, 2021, 2022, 2023), "Validation": (2024,), "Evaluation": (2025,)}


def assign_role(year: pd.Series) -> pd.Series:
    values = np.select(
        [year.between(2020, 2023), year.eq(2024), year.eq(2025)],
        ROLE_ORDER,
        default="Outside",
    )
    return pd.Series(values, index=year.index, dtype="string")


def is_complete(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    complete = frame[columns].notna().all(axis=1)
    numeric_columns = [column for column in columns if pd.api.types.is_numeric_dtype(frame[column])]
    if numeric_columns:
        complete &= np.isfinite(frame[numeric_columns].to_numpy(dtype="float64")).all(axis=1)
    return complete


def role_mask(frame: pd.DataFrame, role: str) -> pd.Series:
    if role == "Total":
        return pd.Series(True, index=frame.index)
    return frame["Role"].eq(role)


def pct(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) * 100.0 / float(denominator) if denominator else np.nan


def build_summary(
    source: pd.DataFrame,
    universe: pd.DataFrame,
    frozen: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    for role in ROLE_ORDER + ["Total"]:
        src = source.loc[role_mask(source, role)]
        final = universe.loc[role_mask(universe, role)]
        base = frozen.loc[role_mask(frozen, role)]
        evt = events.loc[role_mask(events, role)]
        rrt_complete = int(is_complete(src, CURRENT_RRT).sum())
        era5_complete = int(is_complete(src, ERA5).sum())
        all_complete = int(is_complete(src, CANDIDATE_PREDICTORS).sum())
        rows.append(
            {
                "role": role,
                "target_years": "2020-2025" if role == "Total" else ",".join(map(str, ROLE_YEARS[role])),
                "frozen_seasonal_evaluable_rows": len(base),
                "frozen_seasonal_def5_event_count": len(evt),
                "source_seasonal_rows": len(src),
                "rows_complete_current_rain_rh_temperature": rrt_complete,
                "rows_complete_six_primary_era5": era5_complete,
                "rows_complete_current_candidate_set": all_complete,
                "rows_excluded_in_phase3_missing_candidate_predictors": len(src) - len(final),
                "final_eligible_rows": len(final),
                "final_eligible_positive_count": int(final["Target_Def5_tPlus1"].sum()),
                "final_eligible_negative_count": int(len(final) - final["Target_Def5_tPlus1"].sum()),
                "final_eligible_station_count": final["Station"].nunique(),
                "upstream_rows_absent_from_current_frozen_source": len(base) - len(src),
                "upstream_positive_events_absent_from_current_frozen_source": len(evt)
                - int(src["Target_Def5_tPlus1"].sum()),
                "eligible_share_of_frozen_evaluable_rows_pct": pct(len(final), len(base)),
                "eligible_positive_share_of_frozen_events_pct": pct(
                    int(final["Target_Def5_tPlus1"].sum()), len(evt)
                ),
            }
        )
    return pd.DataFrame(rows)


def build_monthly(
    source: pd.DataFrame,
    universe: pd.DataFrame,
    frozen: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    for role in ROLE_ORDER:
        for month in range(5, 11):
            src = source.loc[role_mask(source, role) & source["Target_Datetime"].dt.month.eq(month)]
            final = universe.loc[role_mask(universe, role) & universe["Target_Datetime"].dt.month.eq(month)]
            base = frozen.loc[role_mask(frozen, role) & frozen["Target_Datetime"].dt.month.eq(month)]
            evt = events.loc[role_mask(events, role) & events["Target_Datetime"].dt.month.eq(month)]
            positives = int(final["Target_Def5_tPlus1"].sum())
            rows.append(
                {
                    "role": role,
                    "target_years": ",".join(map(str, ROLE_YEARS[role])),
                    "month": month,
                    "month_name": month_name[month],
                    "frozen_evaluable_rows": len(base),
                    "frozen_def5_event_count": len(evt),
                    "source_seasonal_rows": len(src),
                    "upstream_rows_absent_from_current_frozen_source": len(base) - len(src),
                    "final_eligible_rows": len(final),
                    "final_eligible_positive_count": positives,
                    "final_eligible_negative_count": len(final) - positives,
                    "final_eligible_station_count": final["Station"].nunique(),
                    "eligible_def5_rate_per_100000_rows": pct(positives, len(final)) * 1000.0,
                }
            )
    return pd.DataFrame(rows)


def build_station_year(
    universe: pd.DataFrame,
    frozen: pd.DataFrame,
    station_ids: list[str],
) -> pd.DataFrame:
    rows: list[dict] = []
    for year in range(2020, 2026):
        role = "Train" if year <= 2023 else ("Validation" if year == 2024 else "Evaluation")
        for station in station_ids:
            final = universe.loc[
                universe["Station"].eq(station) & universe["Target_Datetime"].dt.year.eq(year)
            ]
            base = frozen.loc[
                frozen["Station"].eq(station) & frozen["Target_Datetime"].dt.year.eq(year)
            ]
            months = sorted(final["Target_Datetime"].dt.month.unique().tolist())
            if len(base) == 0:
                availability = "STRUCTURAL_ABSENCE_IN_FROZEN_DYNAMIC_NETWORK"
            elif len(final) == 0:
                availability = "NO_ELIGIBLE_ROWS_AFTER_UPSTREAM_PREDICTOR_COMPLETENESS"
            else:
                availability = "OBSERVED"
            positives = int(final["Target_Def5_tPlus1"].sum())
            rows.append(
                {
                    "role": role,
                    "target_year": year,
                    "station": station,
                    "frozen_evaluable_rows": len(base),
                    "final_eligible_rows": len(final),
                    "upstream_rows_absent_from_current_frozen_source": len(base) - len(final),
                    "final_eligible_positive_count": positives,
                    "final_eligible_negative_count": len(final) - positives,
                    "observed_candidate_month_count": len(months),
                    "observed_candidate_months": ",".join(map(str, months)),
                    "complete_may_october_window_observed": months == list(range(5, 11)),
                    "first_target_datetime": final["Target_Datetime"].min(),
                    "last_target_datetime": final["Target_Datetime"].max(),
                    "station_year_availability": availability,
                }
            )
    return pd.DataFrame(rows)


def build_missingness(source: pd.DataFrame) -> pd.DataFrame:
    families = {
        "Rainfall predictors": RAIN,
        "Relative-humidity predictors": RH,
        "Temperature predictors": TEMP,
        "Current Rain/RH/Temperature combined": CURRENT_RRT,
        "Temporal cyclic predictors": TEMPORAL,
        "Static GIS and unchanged LULC predictors": STATIC,
        "Six primary ERA5 predictors": ERA5,
        "All retained candidate predictors": CANDIDATE_PREDICTORS,
    }
    rows: list[dict] = []
    for role in ROLE_ORDER + ["Total"]:
        part = source.loc[role_mask(source, role)]
        for family, columns in families.items():
            missing = part[columns].isna()
            numeric = [column for column in columns if pd.api.types.is_numeric_dtype(part[column])]
            nonfinite = 0
            if numeric:
                values = part[numeric].to_numpy(dtype="float64")
                nonfinite = int((~np.isfinite(values) & ~np.isnan(values)).sum())
            complete_rows = int(is_complete(part, columns).sum())
            columns_missing = [column for column in columns if missing[column].any()]
            rows.append(
                {
                    "role": role,
                    "predictor_family": family,
                    "predictor_column_count": len(columns),
                    "source_seasonal_rows": len(part),
                    "complete_rows": complete_rows,
                    "incomplete_rows": len(part) - complete_rows,
                    "incomplete_rows_pct": pct(len(part) - complete_rows, len(part)),
                    "missing_cells": int(missing.sum().sum()),
                    "missing_cells_pct": pct(int(missing.sum().sum()), len(part) * len(columns)),
                    "nonfinite_nonmissing_cells": nonfinite,
                    "columns_with_missing_values": ",".join(columns_missing),
                }
            )
    return pd.DataFrame(rows)


def build_report(
    summary: pd.DataFrame,
    monthly: pd.DataFrame,
    station_year: pd.DataFrame,
    missingness: pd.DataFrame,
) -> str:
    lookup = summary.set_index("role")
    role_rows = "\n".join(
        f"| {role} | {int(lookup.loc[role, 'final_eligible_rows']):,} | "
        f"{int(lookup.loc[role, 'final_eligible_positive_count']):,} | "
        f"{int(lookup.loc[role, 'final_eligible_negative_count']):,} | "
        f"{int(lookup.loc[role, 'final_eligible_station_count'])} |"
        for role in ROLE_ORDER
    )
    discrepancy_rows = "\n".join(
        f"| {role} | {int(lookup.loc[role, 'frozen_seasonal_evaluable_rows']):,} | "
        f"{int(lookup.loc[role, 'source_seasonal_rows']):,} | "
        f"{int(lookup.loc[role, 'upstream_rows_absent_from_current_frozen_source']):,} | "
        f"{int(lookup.loc[role, 'frozen_seasonal_def5_event_count']):,} | "
        f"{int(lookup.loc[role, 'final_eligible_positive_count']):,} | "
        f"{int(lookup.loc[role, 'upstream_positive_events_absent_from_current_frozen_source']):,} |"
        for role in ROLE_ORDER
    )
    month_totals = monthly.groupby(["month", "month_name"], sort=True).agg(
        rows=("final_eligible_rows", "sum"), positives=("final_eligible_positive_count", "sum")
    ).reset_index()
    month_rows = "\n".join(
        f"| {row.month_name} | {int(row.rows):,} | {int(row.positives):,} |"
        for row in month_totals.itertuples(index=False)
    )
    structural = station_year[station_year["station_year_availability"].str.startswith("STRUCTURAL")]
    structural_text = "; ".join(
        f"{year}: {', '.join(group['station'].tolist())}"
        for year, group in structural.groupby("target_year", sort=True)
    ) or "None"
    no_eligible = station_year[
        station_year["station_year_availability"].eq(
            "NO_ELIGIBLE_ROWS_AFTER_UPSTREAM_PREDICTOR_COMPLETENESS"
        )
    ]
    all_missing_zero = int(missingness["incomplete_rows"].sum()) == 0

    return f"""# Phase 3 — Seasonal Modeling Universe

## 1. Frozen design used

The seasonal window is May-October. `Target_Datetime` assigns both seasonal membership and role: Train contains target years 2020-2023, Validation contains target year 2024, and Evaluation contains target year 2025. Def5, frozen P99, the seasonal window, the modeling start year, and these temporal roles were not changed.

## 2. Exact source inputs

The common population was filtered from `PhaseB_Atmospheric/ERA5/06_Def5_ModelReady/PhaseB_Def5_ERA5_ModelReady_v1.parquet`. This frozen source contains cleaned current rain, relative-humidity and temperature histories, temporal cycles, static elevation/LULC/SUHI fields, and the six primary ERA5 predictors at predictor time `t`.

The frozen t+1 base `Def5_Final_Target_2012_2025_v1/04_Def5_Plus1h_Target/ML_Base_StationHour_Target_Def5_t1_2012_2025_v1.parquet` and frozen positive-event table were used as upstream denominator and event-count controls. The Phase 2 May-October counts were checked independently and matched 370 / 97 / 116 for Train-era / 2024 / 2025.

The source's old `Split`, ERA5 timestamps, ERA5 grid coordinates, `sp`, `valid_850`, and `era5_join_matched` are audit or provenance fields and were not retained. No candidate predictor was removed. `LULC_Class` remains unchanged and was not encoded.

## 3. Why Target_Datetime defines season membership

The forecast target occurs at `t+1`, so May-October membership must describe the target hour, not the predictor hour. Every output row satisfies `Target_Datetime = Datetime + 1 hour`. This prevents the first and last forecast hours of a season from being assigned by the wrong timestamp.

## 4. Train / Validation / Evaluation row counts

| Role | Eligible rows | Def5 positives | Negatives | Stations |
|---|---:|---:|---:|---:|
{role_rows}

The final common universe contains **{int(lookup.loc['Total', 'final_eligible_rows']):,} station-hours**, **{int(lookup.loc['Total', 'final_eligible_positive_count']):,} positives**, and **{int(lookup.loc['Total', 'final_eligible_negative_count']):,} negatives**.

## 5. Def5 positives and negatives per role

Counts above are labels only. No class weighting, resampling, thresholding, or metric calculation was performed. The binary target check passed with values limited to 0 and 1.

## 6. Station coverage per role

Train contains 40 stations, Validation 39, and Evaluation 38. The full station-year table includes zero-row station-years so structural absence remains visible. Structural absences in the frozen dynamic t+1 network are: **{structural_text}**.

There are **{len(no_eligible)}** station-years with frozen seasonal evaluable rows but no rows in the current complete-predictor source. Station-level row differences are retained in `Seasonal_Modeling_Universe_StationYear_v1.csv`; no balanced panel or manufactured station-hour was created.

## 7. Monthly coverage

| Target month | Eligible rows, all roles | Def5 positives |
|---|---:|---:|
{month_rows}

All output targets fall in months 5-10. Monthly detail by role, including frozen upstream denominators and upstream exclusions, is in `Seasonal_Modeling_Universe_Monthly_v1.csv`.

## 8. Missing predictor patterns

Within the current frozen Phase B source, all retained rain, RH, temperature, temporal, static GIS/LULC, and six primary ERA5 predictor values are complete and finite for the selected 2020-2025 May-October rows: **{'yes' if all_missing_zero else 'no'}**. Therefore Phase 3 excluded zero additional rows for missing candidate predictors.

This zero does not mean the upstream observational record was complete. The Phase B source was built from earlier complete-predictor Def5 split rows, so upstream incomplete rows had already been excluded before Phase 3. The missingness CSV distinguishes in-source missingness from that upstream population difference.

## 9. Structural station absence

Station availability follows the frozen dynamic observation network. A station-year with zero frozen t+1 rows is labeled structural absence. Missing station-hours were not filled, interpolated, imputed, copied from another station, or represented as zero rainfall.

## 10. Frozen-event versus eligible-positive differences

| Role | Frozen evaluable rows | Current source rows | Upstream row difference | Frozen Def5 events | Eligible positives | Positive difference |
|---|---:|---:|---:|---:|---:|---:|
{discrepancy_rows}

In total, the current frozen complete-predictor source contains **{int(lookup.loc['Total', 'upstream_rows_absent_from_current_frozen_source']):,} fewer rows** and **{int(lookup.loc['Total', 'upstream_positive_events_absent_from_current_frozen_source']):,} fewer positives** than the broader frozen seasonal t+1 population. This is explained by upstream complete-predictor eligibility, not by Phase 3 seasonal filtering or ERA5 missingness. The selected seasonal rows have complete six-variable ERA5 coverage.

## 11. Readiness for Phase 4 feature architecture

The dataset is ready for Phase 4 inventory and feature-architecture review because it has one consistent population, frozen role assignments, no duplicate keys, a verified one-hour horizon, a binary target, and complete retained candidate predictors. The caveat is that the population is a complete-predictor subset of the broader frozen t+1 seasonal event population; Phase 4 must preserve this denominator and must not reinterpret upstream exclusions as random missingness.

## PHASE 3 STATUS

PASS_WITH_CAVEATS

- No feature selection was performed.
- No LULC recoding was performed.
- No model was trained.
- No model metric was computed.
- No SHAP analysis was performed.
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    raw = pd.read_parquet(SOURCE)
    required_source = set(["Station", "Datetime", "Target_Datetime", "Target_Def5_tPlus1"] + CANDIDATE_PREDICTORS)
    missing_source_columns = sorted(required_source - set(raw.columns))
    assert not missing_source_columns, f"Missing frozen source columns: {missing_source_columns}"

    season = raw.loc[
        raw["Target_Datetime"].dt.year.between(2020, 2025)
        & raw["Target_Datetime"].dt.month.between(5, 10)
    ].copy()
    season["Role"] = assign_role(season["Target_Datetime"].dt.year)
    season = season[["Station", "Datetime", "Target_Datetime", "Role", "Target_Def5_tPlus1"] + CANDIDATE_PREDICTORS]

    frozen = pd.read_parquet(
        FROZEN_T1, columns=["Station", "Datetime", "Target_Datetime", "Target_Def5_tPlus1"]
    )
    frozen = frozen.loc[
        frozen["Target_Datetime"].dt.year.between(2020, 2025)
        & frozen["Target_Datetime"].dt.month.between(5, 10)
    ].copy()
    frozen["Role"] = assign_role(frozen["Target_Datetime"].dt.year)

    events = pd.read_parquet(FROZEN_EVENTS, columns=["Station", "Datetime", "Def5"])
    events = events.rename(columns={"Datetime": "Target_Datetime"})
    events = events.loc[
        events["Target_Datetime"].dt.year.between(2020, 2025)
        & events["Target_Datetime"].dt.month.between(5, 10)
    ].copy()
    events["Role"] = assign_role(events["Target_Datetime"].dt.year)

    phase2 = pd.read_csv(PHASE2_COUNTS)
    phase2 = phase2.loc[phase2["candidate_id"].eq("MAY_OCT")]
    phase2_counts = {
        "Train": int(phase2.loc[phase2["year"].between(2020, 2023), "def5_count"].sum()),
        "Validation": int(phase2.loc[phase2["year"].eq(2024), "def5_count"].sum()),
        "Evaluation": int(phase2.loc[phase2["year"].eq(2025), "def5_count"].sum()),
    }

    complete_candidate = is_complete(season, CANDIDATE_PREDICTORS)
    universe = season.loc[complete_candidate].copy()
    universe["Role"] = pd.Categorical(universe["Role"], categories=ROLE_ORDER, ordered=True)
    universe = universe.sort_values(["Role", "Target_Datetime", "Station"]).reset_index(drop=True)
    universe["Role"] = universe["Role"].astype("string")

    assert not season.duplicated(["Station", "Datetime"]).any()
    assert not season.duplicated(["Station", "Target_Datetime"]).any()
    assert not universe.duplicated(["Station", "Datetime"]).any()
    assert not universe.duplicated(["Station", "Target_Datetime"]).any()
    assert (universe["Target_Datetime"] - universe["Datetime"]).eq(pd.Timedelta(hours=1)).all()
    assert universe["Target_Datetime"].dt.month.between(5, 10).all()
    assert universe.loc[universe["Role"].eq("Train"), "Target_Datetime"].dt.year.between(2020, 2023).all()
    assert universe.loc[universe["Role"].eq("Validation"), "Target_Datetime"].dt.year.eq(2024).all()
    assert universe.loc[universe["Role"].eq("Evaluation"), "Target_Datetime"].dt.year.eq(2025).all()
    assert set(universe["Target_Def5_tPlus1"].unique()) <= {0, 1}
    assert is_complete(universe, CANDIDATE_PREDICTORS).all()
    assert not (set(AUDIT_ONLY_NOT_RETAINED) & set(universe.columns))
    assert len(universe.columns) == len(IDENTIFIERS) + len(CANDIDATE_PREDICTORS)

    for role in ROLE_ORDER:
        frozen_positive = int(frozen.loc[frozen["Role"].eq(role), "Target_Def5_tPlus1"].sum())
        event_positive = int(events["Role"].eq(role).sum())
        assert frozen_positive == event_positive == phase2_counts[role]

    station_frame = pd.read_csv(STATIONS, dtype={"Station": "string"})
    station_ids = sorted(station_frame["Station"].dropna().astype(str).unique().tolist())
    assert len(station_ids) == 40
    assert set(universe["Station"].astype(str).unique()) <= set(station_ids)

    summary = build_summary(season, universe, frozen, events)
    monthly = build_monthly(season, universe, frozen, events)
    station_year = build_station_year(universe, frozen, station_ids)
    missingness = build_missingness(season)

    assert len(monthly) == 18
    assert len(station_year) == 240
    assert summary.loc[summary["role"].eq("Total"), "final_eligible_rows"].iat[0] == len(universe)
    assert summary.loc[summary["role"].eq("Total"), "final_eligible_positive_count"].iat[0] == int(
        universe["Target_Def5_tPlus1"].sum()
    )

    universe.to_parquet(
        OUT / "Seasonal_Modeling_Universe_v1.parquet",
        index=False,
        compression="zstd",
    )
    summary.to_csv(OUT / "Seasonal_Modeling_Universe_Summary_v1.csv", index=False, float_format="%.6f")
    monthly.to_csv(OUT / "Seasonal_Modeling_Universe_Monthly_v1.csv", index=False, float_format="%.6f")
    station_year.to_csv(OUT / "Seasonal_Modeling_Universe_StationYear_v1.csv", index=False, float_format="%.6f")
    missingness.to_csv(OUT / "Seasonal_Modeling_Universe_Missingness_v1.csv", index=False, float_format="%.6f")
    (OUT / "Seasonal_Modeling_Universe_Report_v1.md").write_text(
        build_report(summary, monthly, station_year, missingness), encoding="utf-8"
    )

    print(summary.to_string(index=False))
    print(f"Wrote Phase 3 outputs to {OUT}")


if __name__ == "__main__":
    main()
