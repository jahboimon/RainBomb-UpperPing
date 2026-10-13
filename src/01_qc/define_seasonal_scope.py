"""Define climatology-supported seasonal candidates and audit frozen Def5 consistency.

Phase 2 is descriptive only. The script reads the frozen Def5 event table and
frozen t+1 evaluable station-hour base, writes three CSV summaries, one figure,
and a Markdown report. It does not create modeling data or inspect model output.
"""

from __future__ import annotations
import sys as _sys
from pathlib import Path as _ConfigPath
_sys.path.insert(0, str(_ConfigPath(__file__).resolve().parents[1]))
from rainbomb_paths import load_paths as _load_paths
PATHS = _load_paths()

from calendar import month_abbr, month_name
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


LEGACY = PATHS.upstream_root
ROOT = PATHS.runtime_root
OUT = ROOT / "03_OUTPUT" / "02_Seasonal_Scope"

EVENTS_PATH = (
    LEGACY
    / "Def5_Final_Target_2012_2025_v1"
    / "02_Def5_Extended_Target"
    / "RainBomb_Target_Def5_2012_2025_v1.parquet"
)
EVALUABLE_PATH = (
    LEGACY
    / "Def5_Final_Target_2012_2025_v1"
    / "04_Def5_Plus1h_Target"
    / "ML_Base_StationHour_Target_Def5_t1_2012_2025_v1.parquet"
)

SOURCES = [
    {
        "key": "TMD_seasons",
        "source": "Thai Meteorological Department, Seasons of Thailand",
        "url": "https://tmd.go.th/info/%E0%B8%A4%E0%B8%94%E0%B8%81%E0%B8%B2%E0%B8%A5%E0%B8%82%E0%B8%AD%E0%B8%87%E0%B8%9B%E0%B8%A3%E0%B8%B0%E0%B9%80%E0%B8%97%E0%B8%A8%E0%B9%84%E0%B8%97%E0%B8%A2",
        "relevance": "Official national description explicitly discussing upper Thailand",
        "definition": "Approximately mid-May to mid-October",
        "type": "Monsoonal and operational climatology",
        "limitation": "Day-level onset and withdrawal vary by year; it is not an Upper Ping-only calendar-month rule.",
    },
    {
        "key": "Taesombat_Sriwongsitanon_2009",
        "source": "Taesombat & Sriwongsitanon (2009), Areal rainfall estimation using spatial interpolation techniques",
        "url": "https://www.thaiscience.info/journals/Article/SCAS/10558554.pdf",
        "relevance": "Study area is the Upper Ping River Basin",
        "definition": "May to October",
        "type": "Basin climatology / monsoonal",
        "limitation": "The season statement is contextual; the paper's main analysis concerns rainfall interpolation.",
    },
    {
        "key": "UpperPing_downscaling_2017",
        "source": "Projections of future rainfall for the Upper Ping River Basin using regression-based downscaling (2017)",
        "url": "https://senswat.eng.ku.ac.th/2024/wp-content/uploads/2020/06/Projections-of-future-rainfall-for-the-upper-Ping-River-Basin-using-regression-based-downscaling.pdf",
        "relevance": "Study area is the Upper Ping River Basin",
        "definition": "Wet season May to October; dry season November to April",
        "type": "Rainfall-based seasonal aggregation",
        "limitation": "A six-month analytical grouping, not an annual operational declaration of onset.",
    },
    {
        "key": "UpperNorthernThailand_2025",
        "source": "Analysis of long-term rainfall trend and extreme in upper northern Thailand (2025)",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12480976/",
        "relevance": "Upper northern Thailand, including the broader climatic region of Chiang Mai and Upper Ping",
        "definition": "Rainy season June to October in the study's full-month TMD-derived classification",
        "type": "Rainfall study using categorical calendar seasons",
        "limitation": "The same paper notes monsoon onset around mid-May; assigning all May to summer is a categorical simplification.",
    },
]

CANDIDATES = [
    {
        "candidate_id": "MAY_OCT",
        "candidate_label": "May-October",
        "months": tuple(range(5, 11)),
        "source_keys": "TMD_seasons; Taesombat_Sriwongsitanon_2009; UpperPing_downscaling_2017",
        "basis": "Calendar-month approximation of the official mid-May to mid-October monsoon, directly supported by two Upper Ping studies.",
    },
    {
        "candidate_id": "JUN_OCT",
        "candidate_label": "June-October",
        "months": tuple(range(6, 11)),
        "source_keys": "UpperNorthernThailand_2025",
        "basis": "Conservative full-month core used by a recent upper-northern-Thailand rainfall study; it treats May as a boundary month.",
    },
]


def safe_pct(numerator: float | int, denominator: float | int) -> float:
    return float(numerator) * 100.0 / float(denominator) if denominator else np.nan


def role_for_year(year: int) -> str:
    if 2020 <= year <= 2023:
        return "Train-era"
    if year == 2024:
        return "Validation"
    if year == 2025:
        return "Evaluation"
    return "Descriptive"


def fmt_int(value: float | int) -> str:
    return f"{int(value):,}"


def fmt_pct(value: float) -> str:
    return f"{value:.1f}%"


def build_monthly(events: pd.DataFrame, evaluable: pd.DataFrame) -> pd.DataFrame:
    event_counts = events.groupby(events["Datetime"].dt.month).size().reindex(range(1, 13), fill_value=0)
    hour_counts = evaluable.groupby(evaluable["Target_Datetime"].dt.month).size().reindex(range(1, 13), fill_value=0)
    positive_counts = (
        evaluable.groupby(evaluable["Target_Datetime"].dt.month)["Target_Def5_tPlus1"]
        .sum()
        .reindex(range(1, 13), fill_value=0)
    )
    assert event_counts.equals(positive_counts.astype(event_counts.dtype)), "Frozen event and t+1 counts disagree by month"

    total_events = int(event_counts.sum())
    monthly = pd.DataFrame(
        {
            "month": range(1, 13),
            "month_name": [month_name[m] for m in range(1, 13)],
            "def5_count": event_counts.to_numpy(dtype="int64"),
            "evaluable_station_hours": hour_counts.to_numpy(dtype="int64"),
        }
    )
    monthly["def5_rate_per_100000_evaluable_hours"] = (
        monthly["def5_count"] / monthly["evaluable_station_hours"] * 100_000.0
    )
    monthly["share_of_all_def5_events_pct"] = monthly["def5_count"] / total_events * 100.0
    monthly["cumulative_calendar_year_def5_share_pct"] = monthly["share_of_all_def5_events_pct"].cumsum()
    for candidate in CANDIDATES:
        suffix = candidate["candidate_id"].lower()
        included = monthly["month"].isin(candidate["months"])
        monthly[f"included_{suffix}"] = included
        monthly[f"cumulative_all_event_capture_{suffix}_pct"] = (
            monthly["def5_count"].where(included, 0).cumsum() / total_events * 100.0
        )
    return monthly


def build_season_year(events: pd.DataFrame, evaluable: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for candidate in CANDIDATES:
        months = candidate["months"]
        for year in range(2020, 2026):
            event_year = events[events["Datetime"].dt.year.eq(year)]
            eval_year = evaluable[evaluable["Target_Datetime"].dt.year.eq(year)]
            event_season = event_year[event_year["Datetime"].dt.month.isin(months)]
            eval_season = eval_year[eval_year["Target_Datetime"].dt.month.isin(months)]
            observed = sorted(eval_season["Target_Datetime"].dt.month.unique().tolist())
            missing = sorted(set(months) - set(observed))
            rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "candidate_label": candidate["candidate_label"],
                    "months_included": ",".join(str(m) for m in months),
                    "year": year,
                    "analysis_role": role_for_year(year),
                    "def5_count": len(event_season),
                    "annual_def5_count": len(event_year),
                    "def5_retained_from_year_pct": safe_pct(len(event_season), len(event_year)),
                    "evaluable_station_hours": len(eval_season),
                    "annual_evaluable_station_hours": len(eval_year),
                    "evaluable_hours_retained_from_year_pct": safe_pct(len(eval_season), len(eval_year)),
                    "def5_rate_per_100000_evaluable_hours": safe_pct(len(event_season), len(eval_season)) * 1000.0,
                    "stations_with_evaluable_hours": eval_season["Station"].nunique(),
                    "expected_calendar_months": len(months),
                    "observed_calendar_months": len(observed),
                    "complete_candidate_calendar_window_observed": not missing,
                    "missing_calendar_months": ",".join(str(m) for m in missing),
                }
            )
    return pd.DataFrame(rows)


def build_candidates(events: pd.DataFrame, evaluable: pd.DataFrame, season_year: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for candidate in CANDIDATES:
        months = candidate["months"]
        event_mask = events["Datetime"].dt.month.isin(months)
        eval_mask = evaluable["Target_Datetime"].dt.month.isin(months)
        candidate_years = season_year[season_year["candidate_id"].eq(candidate["candidate_id"])]
        annual_retention = []
        for year in range(2012, 2026):
            all_count = events["Datetime"].dt.year.eq(year).sum()
            selected_count = (events["Datetime"].dt.year.eq(year) & event_mask).sum()
            annual_retention.append(safe_pct(selected_count, all_count))

        def period_values(years: tuple[int, ...]) -> tuple[int, int]:
            event_period = event_mask & events["Datetime"].dt.year.isin(years)
            eval_period = eval_mask & evaluable["Target_Datetime"].dt.year.isin(years)
            return int(event_period.sum()), int(eval_period.sum())

        train_events, train_hours = period_values((2020, 2021, 2022, 2023))
        val_events, val_hours = period_values((2024,))
        eval_events, eval_hours = period_values((2025,))
        selected_events = int(event_mask.sum())
        selected_hours = int(eval_mask.sum())
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "candidate_label": candidate["candidate_label"],
                "months_included": ",".join(str(m) for m in months),
                "climatological_basis": candidate["basis"],
                "supporting_source_keys": candidate["source_keys"],
                "def5_count_2012_2025": selected_events,
                "evaluable_station_hours_2012_2025": selected_hours,
                "def5_rate_per_100000_evaluable_hours_2012_2025": safe_pct(selected_events, selected_hours) * 1000.0,
                "share_of_all_def5_events_retained_pct_2012_2025": safe_pct(selected_events, len(events)),
                "mean_annual_def5_retained_pct_2012_2025": float(np.nanmean(annual_retention)),
                "minimum_annual_def5_retained_pct_2012_2025": float(np.nanmin(annual_retention)),
                "maximum_annual_def5_retained_pct_2012_2025": float(np.nanmax(annual_retention)),
                "train_era_2020_2023_def5_count": train_events,
                "train_era_2020_2023_evaluable_station_hours": train_hours,
                "validation_2024_def5_count": val_events,
                "validation_2024_evaluable_station_hours": val_hours,
                "evaluation_2025_def5_count": eval_events,
                "evaluation_2025_evaluable_station_hours": eval_hours,
                "complete_candidate_calendar_window_observed_in_every_year_2020_2025": bool(
                    candidate_years["complete_candidate_calendar_window_observed"].all()
                ),
                "years_with_incomplete_candidate_calendar_window_2020_2025": ",".join(
                    candidate_years.loc[
                        ~candidate_years["complete_candidate_calendar_window_observed"], "year"
                    ].astype(str)
                ),
            }
        )
    return pd.DataFrame(rows)


def make_figure(monthly: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True, constrained_layout=True)
    colors = ["#d7dde5"] * 12
    colors[4] = "#f0b45f"  # May boundary month
    for index in range(5, 10):
        colors[index] = "#3f8f72"  # June-October core

    axes[0].bar(monthly["month"], monthly["def5_count"], color=colors, edgecolor="#ffffff", linewidth=0.7)
    axes[0].set_ylabel("Frozen Def5 events")
    axes[0].set_title("Monthly frozen Def5 climatology, 2012-2025")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].plot(
        monthly["month"],
        monthly["def5_rate_per_100000_evaluable_hours"],
        marker="o",
        color="#264653",
        linewidth=2,
    )
    axes[1].set_ylabel("Def5 per 100,000\nevaluable station-hours")
    axes[1].set_xlabel("Calendar month")
    axes[1].set_xticks(range(1, 13), [month_abbr[m] for m in range(1, 13)])
    axes[1].grid(axis="y", alpha=0.25)
    axes[0].legend(
        handles=[
            Patch(facecolor="#f0b45f", label="May: boundary month (May-October candidate only)"),
            Patch(facecolor="#3f8f72", label="June-October: shared core of both candidates"),
            Patch(facecolor="#d7dde5", label="Outside both candidates"),
        ],
        frameon=False,
        loc="upper left",
    )
    fig.savefig(OUT / "Monthly_Def5_Climatology_v1.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_report(monthly: pd.DataFrame, candidates: pd.DataFrame, season_year: pd.DataFrame) -> str:
    candidate_lookup = candidates.set_index("candidate_id")
    peak_count = monthly.loc[monthly["def5_count"].idxmax()]
    peak_rate = monthly.loc[monthly["def5_rate_per_100000_evaluable_hours"].idxmax()]

    source_rows = "\n".join(
        f"| [{source['source']}]({source['url']}) | {source['relevance']} | {source['definition']} | "
        f"{source['type']} | {source['limitation']} |"
        for source in SOURCES
    )
    candidate_rows = "\n".join(
        f"| {row.candidate_label} | {row.months_included} | {fmt_int(row.def5_count_2012_2025)} | "
        f"{fmt_int(row.evaluable_station_hours_2012_2025)} | "
        f"{row.def5_rate_per_100000_evaluable_hours_2012_2025:.2f} | "
        f"{fmt_pct(row.share_of_all_def5_events_retained_pct_2012_2025)} |"
        for row in candidates.itertuples(index=False)
    )
    year_rows = "\n".join(
        f"| {row.candidate_label} | {row.year} | {row.analysis_role} | {fmt_int(row.def5_count)} | "
        f"{fmt_int(row.evaluable_station_hours)} | {fmt_pct(row.def5_retained_from_year_pct)} | "
        f"{'Yes' if row.complete_candidate_calendar_window_observed else 'No'} |"
        for row in season_year.itertuples(index=False)
    )
    may = candidate_lookup.loc["MAY_OCT"]
    jun = candidate_lookup.loc["JUN_OCT"]

    return f"""# Phase 2 — Seasonal Scope Definition

## 1. Scientific question

Which calendar months are scientifically defensible rainy-season candidates for localized heavy-rain / frozen Def5 forecasting in the Upper Ping River Basin? Candidate windows were defined from external climatological evidence first. Frozen Def5 data were used only as a consistency and coverage check, never to optimize a window or a model score.

## 2. Sources used for rainy-season definition

| Source | Geographic relevance | Rainy-season definition | Definition type | Important limitation |
|---|---|---|---|---|
{source_rows}

The official TMD description places the upper-Thailand monsoon approximately from mid-May to mid-October. Two Upper Ping publications use May-October as a complete-month wet-season representation. A recent upper-northern-Thailand study instead uses June-October as a categorical rainy season while also acknowledging the mid-May monsoon onset. These sources establish the candidates before examining Def5.

## 3. Candidate rainy-season windows supported by those sources

1. **May-October** — the stronger basin-specific candidate. It is a calendar-month approximation of the TMD mid-May to mid-October monsoon and is directly used in Upper Ping studies.
2. **June-October** — a conservative full-month core supported by the recent upper-northern-Thailand seasonal classification. It deliberately treats May as a boundary month, not as a month excluded because of Def5 performance.

The evidence does not justify searching arbitrary month combinations. May-September was therefore not evaluated as a candidate.

## 4. Monthly Def5 climatology

Across 2012-2025, the frozen record contains **{fmt_int(len(monthly) and monthly['def5_count'].sum())} Def5 events** and **{fmt_int(monthly['evaluable_station_hours'].sum())} evaluable station-hours**. The largest event count occurs in **{peak_count.month_name} ({fmt_int(peak_count.def5_count)})**; the highest denominator-adjusted rate occurs in **{peak_rate.month_name} ({peak_rate.def5_rate_per_100000_evaluable_hours:.2f} per 100,000 evaluable station-hours)**. Monthly counts, denominators, rates, shares, and cumulative candidate capture are in `SeasonalScope_Monthly_Def5_v1.csv`.

The concentration from May through October is consistent with the externally defined monsoon season. This is a consistency check, not a data-driven selection rule.

## 5. Def5 capture within each candidate window

| Candidate | Months | Def5 events | Evaluable station-hours | Rate per 100,000 hours | All-event retention |
|---|---|---:|---:|---:|---:|
{candidate_rows}

Adding May changes retention from **{fmt_pct(jun.share_of_all_def5_events_retained_pct_2012_2025)}** for June-October to **{fmt_pct(may.share_of_all_def5_events_retained_pct_2012_2025)}** for May-October. That difference is descriptive; it is not evidence of predictive performance.

## 6. Seasonal Train / Validation / 2025 event counts

The labels below are descriptive time roles requested for this phase: Train-era = 2020-2023, Validation = calendar 2024, and Evaluation = calendar 2025. No fitting, threshold selection, or evaluation metric was run.

| Candidate | Year | Role | Def5 positives | Evaluable station-hours | Share of that year's Def5 retained | All candidate months observed? |
|---|---:|---|---:|---:|---:|---|
{year_rows}

For aggregate role counts, May-October retains **{fmt_int(may.train_era_2020_2023_def5_count)} / {fmt_int(may.validation_2024_def5_count)} / {fmt_int(may.evaluation_2025_def5_count)}** positives in Train-era / Validation / Evaluation, respectively. June-October retains **{fmt_int(jun.train_era_2020_2023_def5_count)} / {fmt_int(jun.validation_2024_def5_count)} / {fmt_int(jun.evaluation_2025_def5_count)}**.

Both candidate calendar windows have evaluable records in every included month for each year from 2020 through 2025. This means the calendar windows are represented; it does **not** imply a fixed station network or complete 24/7 observations at every station.

## 7. Important limitations

- The TMD onset and withdrawal are day-level and vary from year to year. A calendar-month model necessarily approximates the mid-month transitions.
- The June-October alternative is a categorical full-month convention; it should not be interpreted as evidence that May is climatologically dry.
- Def5 is a frozen project-specific localized heavy-rain definition. Its monthly distribution is not an independent definition of the meteorological rainy season.
- Evaluable station-hours come from the frozen Def5 t+1 eligibility base and reflect a dynamic observation network. They are not equivalent to complete predictor hours, and no model-ready dataset was created.
- Complete-window flags mean every candidate month has some evaluable coverage in that year, not that every station is continuously complete.
- The 2012-2025 sample contains only {fmt_int(monthly['def5_count'].sum())} Def5 events, so individual-year retention can vary substantially.

## 8. Methodological implication for the seasonal model

May-October has the most direct Upper Ping and official-monsoon support and is the broad seasonal candidate. June-October is a defensible conservative full-month core that resolves the partial-May onset differently. A later seasonal model should use exactly one researcher-frozen window and should document how the May boundary is treated. This phase supplies evidence for that decision only; it does not make the decision.

## HUMAN DECISION REQUIRED

- No rainy-season window has been frozen.
- No modeling start year has been frozen.
- No model has been trained.
- No model metric has been inspected.
- The researcher must choose the final seasonal window after reviewing climatological evidence and Def5 consistency.
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    events = pd.read_parquet(EVENTS_PATH, columns=["Station", "Datetime", "Def5"])
    evaluable = pd.read_parquet(
        EVALUABLE_PATH,
        columns=["Station", "Target_Datetime", "Target_Def5_tPlus1"],
    )
    assert events["Def5"].eq(1).all()
    assert events["Datetime"].dt.year.between(2012, 2025).all()
    assert evaluable["Target_Datetime"].dt.year.between(2012, 2025).all()
    assert int(evaluable["Target_Def5_tPlus1"].sum()) == len(events)

    monthly = build_monthly(events, evaluable)
    season_year = build_season_year(events, evaluable)
    candidates = build_candidates(events, evaluable, season_year)

    monthly.to_csv(OUT / "SeasonalScope_Monthly_Def5_v1.csv", index=False, float_format="%.6f")
    candidates.to_csv(OUT / "SeasonalScope_CandidateWindows_v1.csv", index=False, float_format="%.6f")
    season_year.to_csv(OUT / "SeasonalScope_SeasonYearCounts_v1.csv", index=False, float_format="%.6f")
    make_figure(monthly)
    (OUT / "SeasonalScope_Report_v1.md").write_text(
        build_report(monthly, candidates, season_year), encoding="utf-8"
    )

    print(f"Wrote Phase 2 outputs to {OUT}")
    print(candidates[["candidate_label", "share_of_all_def5_events_retained_pct_2012_2025"]].to_string(index=False))


if __name__ == "__main__":
    main()
