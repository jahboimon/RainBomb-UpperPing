from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull, Delaunay, distance_matrix
from pyproj import Transformer


HERE = Path(__file__).resolve().parent
DEV_END = pd.Timestamp("2024-06-30 23:00:00")
ONE_HOUR_NS = np.int64(3_600_000_000_000)
MISSING_CODES = {"-999", "999999", "9999", "-"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ensure_dirs() -> dict[str, Path]:
    dirs = {}
    for name in ["R1_zero_missing", "R2_p99_robustness", "R3_event_sensitivity", "R4_spatial_interpolation", "provenance"]:
        p = HERE / name
        p.mkdir(parents=True, exist_ok=True)
        dirs[name] = p
    return dirs


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def context_arrays(rain: pd.DataFrame):
    rain = rain.sort_values(["Station", "Datetime"], kind="mergesort").reset_index(drop=True)
    station = rain["Station"].astype(str).to_numpy()
    codes, _ = pd.factorize(station, sort=True)
    dt_ns = rain["Datetime"].to_numpy(dtype="datetime64[ns]").astype("int64")
    val = rain["Rainfall"].to_numpy(dtype=float)
    n = len(rain)
    ctx, cont = {}, {}
    for off in (-3, -2, -1, 1, 2, 3):
        a = np.full(n, np.nan); c = np.zeros(n, dtype=bool); k = abs(off)
        if off < 0:
            idx = np.arange(k, n); src = idx - k
            c[idx] = (codes[idx] == codes[src]) & ((dt_ns[idx] - dt_ns[src]) == k * ONE_HOUR_NS)
            good = idx[c[idx]]; a[good] = val[good-k]
        else:
            idx = np.arange(0, n-k); src = idx + k
            c[idx] = (codes[idx] == codes[src]) & ((dt_ns[src] - dt_ns[idx]) == k * ONE_HOUR_NS)
            good = idx[c[idx]]; a[good] = val[good+k]
        ctx[off], cont[off] = a, c
    eligible = np.isfinite(val)
    for off in (-3, -2, -1, 1, 2, 3):
        eligible &= cont[off] & np.isfinite(ctx[off])
    ante = ctx[-1] + ctx[-2] + ctx[-3]
    postmax = np.maximum.reduce([ctx[1], ctx[2], ctx[3]])
    return rain, val, ctx, eligible, ante, postmax


def zero_run_summary(group: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for station, g in group.groupby("Station", sort=True):
        g = g.sort_values("Datetime")
        z = g["Rainfall"].eq(0).to_numpy()
        dt = g["Datetime"].to_numpy(dtype="datetime64[ns]")
        run = 0; starts = []; lengths = []
        for i in range(len(g)):
            continuous = i > 0 and (dt[i] - dt[i-1]) == np.timedelta64(1, "h")
            if z[i] and (i == 0 or (continuous and z[i-1])):
                run += 1
            elif z[i]:
                run = 1
            else:
                if run: lengths.append(run)
                run = 0
        if run: lengths.append(run)
        rows.append({"Station": station, "Zero_Run_Count": len(lengths), "Max_Zero_Run_h": max(lengths, default=0),
                     "Median_Zero_Run_h": float(np.median(lengths)) if lengths else 0.0,
                     "Runs_GE_168h": int(np.sum(np.asarray(lengths) >= 168))})
    return pd.DataFrame(rows)


def exponential_variogram(h, nugget, sill, rng):
    return nugget + sill * (1.0 - np.exp(-h / np.maximum(rng, 1.0)))


def fit_variogram(xy: np.ndarray, z: np.ndarray):
    d = distance_matrix(xy, xy)
    iu = np.triu_indices(len(z), 1)
    h = d[iu]; gamma = 0.5 * (z[:, None] - z[None, :]) ** 2
    gamma = gamma[iu]
    maxd = np.quantile(h, 0.8)
    bins = np.linspace(0, maxd, 11)
    bh, bg = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (h > lo) & (h <= hi)
        if m.sum() >= 3:
            bh.append(float(h[m].mean())); bg.append(float(gamma[m].mean()))
    bh = np.asarray(bh); bg = np.asarray(bg)
    initial = [max(float(np.nanmin(bg)), 0), max(float(np.nanvar(z)), 1e-6), max(float(np.nanmedian(h)), 1000)]
    try:
        popt, _ = curve_fit(exponential_variogram, bh, bg, p0=initial, bounds=([0, 1e-9, 100], [np.inf, np.inf, np.inf]), maxfev=20000)
    except Exception:
        popt = np.asarray(initial)
    return tuple(map(float, popt)), pd.DataFrame({"Distance_m": bh, "Semivariance": bg})


def ordinary_kriging_predict(train_xy, train_z, query_xy, params):
    n = len(train_z)
    gamma = exponential_variogram(distance_matrix(train_xy, train_xy), *params)
    np.fill_diagonal(gamma, 0.0)
    A = np.zeros((n+1, n+1)); A[:n, :n] = gamma; A[:n, n] = 1; A[n, :n] = 1
    Ainv = np.linalg.pinv(A)
    out = []
    for q in query_xy:
        b = np.r_[exponential_variogram(np.linalg.norm(train_xy-q, axis=1), *params), 1.0]
        w = Ainv @ b
        out.append(float(np.dot(w[:n], train_z)))
    return np.asarray(out)


def idw_predict(train_xy, train_z, query_xy, power=2.0):
    d = distance_matrix(query_xy, train_xy)
    exact = d < 1e-9
    w = 1 / np.maximum(d, 1e-9) ** power
    pred = (w @ train_z) / w.sum(axis=1)
    for i in np.where(exact.any(axis=1))[0]: pred[i] = train_z[np.argmax(exact[i])]
    return pred


def main(root: Path) -> None:
    dirs = ensure_dirs()
    rain_path = root / "Rainfall_Canonical_2012_2025_Final_v1" / "PROCESS_Hourly_Rain_Canonical_2012_2025_v1.parquet"
    rain_csv = rain_path.with_suffix(".csv")
    p99_path = root / "RainBomb_Def4_2012_2025_v1" / "Def4_Frozen_P99_2012_2024H1_v1.csv"
    target_path = root / "Def5_Final_Target_2012_2025_v1" / "02_Def5_Extended_Target" / "RainBomb_Target_Def5_2012_2025_v1.csv"
    meta_path = root / "Hourly_rain" / "RESETWORKPLACE" / "02_processed" / "Station_Metadata_Cleaned.csv"
    raw_clean_path = root / "Hourly_rain" / "RESETWORKPLACE" / "02_processed" / "PROCESS_01_Hourly_Rain_Cleaned.csv"
    forensic_dir = root / "Rainfall_Missing_vs_Zero_Forensic_Audit_v1"
    era5_path = root / "PhaseB_Atmospheric" / "ERA5" / "05_StationHour_Canonical" / "ERA5_StationHour_2013_2025_v1.parquet"
    master_path = root / "ML_StationHour_Master_Candidate_2012_2025_v2" / "ML_StationHour_Master_Candidate_2012_2025_v2.parquet"
    protected = [rain_path, rain_csv, p99_path, target_path, meta_path, raw_clean_path, era5_path]
    for p in protected:
        if not p.is_file(): raise FileNotFoundError(p)
    hashes_before = {str(p): sha256(p) for p in protected}

    rain = pd.read_parquet(rain_path, columns=["Station", "Datetime", "Rainfall"])
    rain["Station"] = rain["Station"].astype("string").str.strip()
    rain["Datetime"] = pd.to_datetime(rain["Datetime"], errors="raise")
    rain["Rainfall"] = pd.to_numeric(rain["Rainfall"], errors="coerce")
    assert not rain.duplicated(["Station", "Datetime"]).any()
    assert rain["Station"].nunique() == 40
    assert (rain["Rainfall"].dropna() >= 0).all() and np.isfinite(rain["Rainfall"].dropna()).all()

    # R1: direct canonical audit, raw lineage evidence, zero runs, structural absences, 7-hour eligibility.
    r1 = dirs["R1_zero_missing"]
    token_counts = {k: 0 for k in MISSING_CODES}; csv_rows = 0; blank_rows = 0
    for chunk in pd.read_csv(rain_csv, dtype=str, keep_default_na=False, chunksize=300_000):
        s = chunk["Rainfall"].str.strip(); csv_rows += len(chunk); blank_rows += int(s.eq("").sum())
        for k in MISSING_CODES: token_counts[k] += int(s.eq(k).sum())
    station_year = rain.assign(Year=rain["Datetime"].dt.year).groupby(["Station", "Year"], observed=True).agg(
        Rows=("Rainfall", "size"), Valid=("Rainfall", "count"), Missing=("Rainfall", lambda s: int(s.isna().sum())),
        Zero=("Rainfall", lambda s: int(s.eq(0).sum())), Positive=("Rainfall", lambda s: int(s.gt(0).sum())),
        First=("Datetime", "min"), Last=("Datetime", "max")).reset_index()
    station_year["Expected_Between_First_Last"] = ((station_year["Last"]-station_year["First"]).dt.total_seconds()/3600+1).astype(int)
    station_year["Structural_Absent_Within_Span"] = station_year["Expected_Between_First_Last"] - station_year["Rows"]
    station_year["Missing_Pct"] = station_year["Missing"] / station_year["Rows"] * 100
    write_csv(station_year, r1 / "R1_station_year_audit.csv")
    zr = zero_run_summary(rain); write_csv(zr, r1 / "R1_zero_run_audit.csv")
    sorted_rain, values, ctx, eligible, ante, postmax = context_arrays(rain)
    p99f = pd.read_csv(p99_path); p99map = p99f.set_index("Station")["P99_Frozen"]
    frozen = sorted_rain["Station"].map(p99map).to_numpy(float)
    s0 = eligible & (values >= np.maximum(frozen, 20.0)) & (ante <= 5) & (postmax < 20)
    official = pd.read_csv(target_path, usecols=["Station", "Datetime"], parse_dates=["Datetime"])
    computed_keys = set(zip(sorted_rain.loc[s0, "Station"].astype(str), sorted_rain.loc[s0, "Datetime"]))
    official_keys = set(zip(official["Station"].astype(str), official["Datetime"]))
    target_match = computed_keys == official_keys
    r1_summary = pd.DataFrame([{
        "Canonical_Rows": len(rain), "Stations": rain["Station"].nunique(), "Numeric_Zero_Rows": int(rain["Rainfall"].eq(0).sum()),
        "Missing_Rows": int(rain["Rainfall"].isna().sum()), "Positive_Rows": int(rain["Rainfall"].gt(0).sum()),
        "CSV_Blank_Missing_Rows": blank_rows, "CSV_Sentinel_Minus999": token_counts["-999"],
        "CSV_Sentinel_999999": token_counts["999999"], "CSV_Sentinel_9999": token_counts["9999"], "CSV_Sentinel_Dash": token_counts["-"],
        "SevenHour_Evaluable_Rows": int(eligible.sum()), "S0_Events_Recomputed": int(s0.sum()), "Official_Def5_Events": len(official),
        "Exact_Official_Key_Match": target_match, "Max_Structural_Absence_Within_StationYear": int(station_year["Structural_Absent_Within_Span"].max()),
        "Conclusion": "PASS" if target_match and not any(token_counts.values()) and blank_rows == rain["Rainfall"].isna().sum() else "REVIEW"
    }]); write_csv(r1_summary, r1 / "R1_zero_missing_summary.csv")
    legacy = forensic_dir / "03_Rainfall_DataLineage_v1.csv"
    if legacy.exists():
        pd.read_csv(legacy).to_csv(r1 / "R1_legacy_lineage_readonly_snapshot.csv", index=False, encoding="utf-8-sig")

    # R2: annual station wet-hour P99, using development period only for frozen comparison.
    r2 = dirs["R2_p99_robustness"]
    dev = rain[rain["Datetime"] <= DEV_END].copy(); dev["Year"] = dev["Datetime"].dt.year
    wet = dev[dev["Rainfall"] > 0]
    annual = wet.groupby(["Station", "Year"])["Rainfall"].agg(Wet_Hours="count", Annual_P99=lambda s: s.quantile(.99, interpolation="linear")).reset_index()
    annual = annual.merge(p99f[["Station", "P99_Frozen"]], on="Station", how="left", validate="many_to_one")
    annual["Difference_mm"] = annual["Annual_P99"] - annual["P99_Frozen"]
    annual["Ratio_to_Frozen"] = annual["Annual_P99"] / annual["P99_Frozen"]
    write_csv(annual, r2 / "R2_annual_station_wethour_p99.csv")
    station_p99 = annual.groupby("Station").agg(P99_Annual_Mean=("Annual_P99","mean"), P99_Annual_SD=("Annual_P99","std"),
        P99_Annual_Min=("Annual_P99","min"), P99_Annual_Max=("Annual_P99","max"), Median_Abs_Diff=("Difference_mm",lambda s: float(np.median(np.abs(s)))),
        Median_Ratio=("Ratio_to_Frozen","median"), Years=("Year","nunique")).reset_index().merge(p99f[["Station","P99_Frozen"]],on="Station")
    station_p99["CV_Annual_P99"] = station_p99["P99_Annual_SD"] / station_p99["P99_Annual_Mean"]
    write_csv(station_p99, r2 / "R2_station_p99_variability.csv")
    fig, ax = plt.subplots(figsize=(12,7));
    for st, g in annual.groupby("Station"): ax.plot(g["Year"], g["Ratio_to_Frozen"], color="#4472C4", alpha=.18, lw=.8)
    med = annual.groupby("Year")["Ratio_to_Frozen"].median(); ax.plot(med.index, med.values, color="#C00000", lw=2.5, label="station median")
    ax.axhline(1,color="black",ls="--",lw=1); ax.set(ylabel="Annual P99 / frozen P99",xlabel="Year",title="Annual station wet-hour P99 relative to frozen 2012–2024H1 P99"); ax.legend(); fig.tight_layout(); fig.savefig(r2/"R2_annual_p99_ratio.png",dpi=220); plt.close(fig)

    # R3: small, justified sensitivity grid.
    r3 = dirs["R3_event_sensitivity"]
    # Stronger thresholds are empirical station P99.5/P99.9 over development wet hours.
    strong = wet.groupby("Station")["Rainfall"].agg(P995=lambda s:s.quantile(.995), P999=lambda s:s.quantile(.999)).reset_index().set_index("Station")
    st995 = sorted_rain["Station"].map(strong["P995"]).to_numpy(float); st999 = sorted_rain["Station"].map(strong["P999"]).to_numpy(float)
    base_intensity = values >= np.maximum(frozen,20)
    scenarios = {
        "S0_Def5": s0,
        "A1_Antecedent_LE10mm": eligible & base_intensity & (ante<=10) & (postmax<20),
        "A2_Antecedent_LE20mm": eligible & base_intensity & (ante<=20) & (postmax<20),
        "P1_PostBelowStationHeavyThreshold": eligible & base_intensity & (ante<=5) & (postmax < np.maximum(frozen,20)),
        "P2_PostRestrictionRemoved_FiniteOnly": eligible & base_intensity & (ante<=5),
        "I1_StationWetHour_P99_5": eligible & (values>=np.maximum(st995,20)) & (ante<=5) & (postmax<20),
        "I2_StationWetHour_P99_9": eligible & (values>=np.maximum(st999,20)) & (ante<=5) & (postmax<20),
    }
    base_idx = set(np.flatnonzero(s0)); summary=[]
    for name, mask in scenarios.items():
        idx=set(np.flatnonzero(mask)); inter=len(base_idx&idx); union=len(base_idx|idx)
        summary.append({"Scenario":name,"Events":len(idx),"Added_vs_S0":len(idx-base_idx),"Removed_vs_S0":len(base_idx-idx),"Overlap_with_S0":inter,"Jaccard_with_S0":inter/union if union else 1})
        ev=sorted_rain.loc[mask,["Station","Datetime","Rainfall"]].copy(); ev["Month"]=ev["Datetime"].dt.month; ev["Hour"]=ev["Datetime"].dt.hour
        write_csv(ev.groupby("Month").size().rename("Events").reset_index(),r3/f"{name}_month.csv")
        write_csv(ev.groupby("Hour").size().rename("Events").reset_index(),r3/f"{name}_hour.csv")
        station_events=ev.groupby("Station").size().rename("Events").reset_index()
        station_eval=pd.DataFrame({"Station":sorted_rain["Station"],"Evaluable":eligible}).groupby("Station")["Evaluable"].sum().reset_index(name="Evaluable_Hours")
        station_events=station_eval.merge(station_events,on="Station",how="left").fillna({"Events":0})
        station_events["Events"]=station_events["Events"].astype(int)
        station_events["Rate_per_100k_Evaluable_Hours"]=station_events["Events"]/station_events["Evaluable_Hours"]*100000
        write_csv(station_events,r3/f"{name}_station.csv")
    write_csv(pd.DataFrame(summary),r3/"R3_scenario_comparison.csv")
    write_csv(strong.reset_index(),r3/"R3_empirical_stronger_thresholds.csv")
    # Representative hydrographs: strongest S0 and strongest added-only event for each family.
    cases=[]
    for name in ["S0_Def5","A2_Antecedent_LE20mm","P2_PostRestrictionRemoved_FiniteOnly","I2_StationWetHour_P99_9"]:
        mask=scenarios[name]
        pre_holdout=sorted_rain["Datetime"].le(DEV_END).to_numpy()
        case_mask=(mask if name in {"S0_Def5","I2_StationWetHour_P99_9"} else (mask & ~s0)) & pre_holdout
        candidate=np.flatnonzero(case_mask)
        if len(candidate):
            i=candidate[np.nanargmax(values[candidate])]
            for off in range(-3,4):
                rv=values[i] if off==0 else ctx[off][i]
                cases.append({"Scenario":name,"Station":sorted_rain.at[i,"Station"],"Event_Datetime":sorted_rain.at[i,"Datetime"],"Offset_h":off,"Rainfall_mm":rv})
    cases=pd.DataFrame(cases); write_csv(cases,r3/"R3_representative_hydrographs.csv")
    if not cases.empty:
        fig,ax=plt.subplots(figsize=(10,6))
        for (sc,st,dt),g in cases.groupby(["Scenario","Station","Event_Datetime"]): ax.plot(g["Offset_h"],g["Rainfall_mm"],marker="o",label=f"{sc}: {st} {dt}")
        ax.axvline(0,color="black",ls="--"); ax.set(xlabel="Hours relative to event",ylabel="Rainfall (mm/h)",title="Representative 7-hour event windows"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(r3/"R3_representative_hydrographs.png",dpi=220); plt.close(fig)

    # R4: development-period S0 rate, IDW vs ordinary kriging, LOSO CV, surface inside convex hull only.
    r4 = dirs["R4_spatial_interpolation"]
    devmask = sorted_rain["Datetime"].le(DEV_END).to_numpy()
    rate = pd.DataFrame({"Station":sorted_rain["Station"],"Evaluable":eligible&devmask,"Event":s0&devmask}).groupby("Station").agg(Evaluable_Hours=("Evaluable","sum"),Def5_Events=("Event","sum")).reset_index()
    rate["Def5_Rate_per_100k_Evaluable_Hours"] = rate["Def5_Events"]/rate["Evaluable_Hours"]*100000
    meta=pd.read_csv(meta_path); rate=rate.merge(meta[["Station","Latitude","Longitude"]],on="Station",validate="one_to_one")
    if master_path.exists():
        elev=pq.read_table(master_path,columns=["Station","Elevation_m"]).to_pandas().groupby("Station",as_index=False)["Elevation_m"].first(); rate=rate.merge(elev,on="Station",how="left")
    tr=Transformer.from_crs(4326,32647,always_xy=True); x,y=tr.transform(rate["Longitude"].to_numpy(),rate["Latitude"].to_numpy()); rate["UTM47_X_m"]=x; rate["UTM47_Y_m"]=y
    xy=np.c_[x,y]; z=rate["Def5_Rate_per_100k_Evaluable_Hours"].to_numpy(float)
    params,vario=fit_variogram(xy,z); write_csv(vario,r4/"R4_empirical_variogram.csv")
    cv=[]
    for i in range(len(z)):
        keep=np.arange(len(z))!=i; p_idw=idw_predict(xy[keep],z[keep],xy[i:i+1])[0]
        pp,_=fit_variogram(xy[keep],z[keep]); p_ok=ordinary_kriging_predict(xy[keep],z[keep],xy[i:i+1],pp)[0]
        cv.append({"Station":rate.at[i,"Station"],"Observed_Rate":z[i],"IDW_Predicted":p_idw,"OK_Predicted":p_ok,"IDW_Error":p_idw-z[i],"OK_Error":p_ok-z[i]})
    cv=pd.DataFrame(cv); write_csv(cv,r4/"R4_LOSO_spatial_CV.csv")
    metrics=pd.DataFrame([{ "Method":m, "MAE":float(np.mean(np.abs(cv[f"{m}_Error"]))), "RMSE":float(np.sqrt(np.mean(cv[f"{m}_Error"]**2))), "Bias":float(np.mean(cv[f"{m}_Error"]))} for m in ["IDW","OK"]])
    write_csv(metrics,r4/"R4_spatial_CV_metrics.csv"); write_csv(rate,r4/"R4_station_def5_rate.csv")
    gx=np.linspace(x.min(),x.max(),120); gy=np.linspace(y.min(),y.max(),120); xx,yy=np.meshgrid(gx,gy); q=np.c_[xx.ravel(),yy.ravel()]
    inside=Delaunay(xy[ConvexHull(xy).vertices]).find_simplex(q)>=0
    nearest=distance_matrix(q,xy).min(axis=1)/1000
    idw=np.full(len(q),np.nan); ok=np.full(len(q),np.nan); idw[inside]=idw_predict(xy,z,q[inside]); ok[inside]=ordinary_kriging_predict(xy,z,q[inside],params)
    grid=pd.DataFrame({"UTM47_X_m":q[:,0],"UTM47_Y_m":q[:,1],"Inside_Station_Convex_Hull":inside,"Nearest_Station_km":nearest,"IDW_Rate":idw,"OK_Rate":ok})
    write_csv(grid,r4/"R4_interpolated_surface_grid.csv")
    fig,axs=plt.subplots(1,2,figsize=(14,7),sharex=True,sharey=True)
    for ax,pred,title in zip(axs,[idw,ok],["IDW estimated surface","Ordinary kriging estimated surface"]):
        im=ax.tricontourf(q[inside,0]/1000,q[inside,1]/1000,pred[inside],levels=14,cmap="viridis"); ax.scatter(x/1000,y/1000,c=z,cmap="viridis",edgecolor="white",s=45); ax.set_title(title); ax.set_xlabel("UTM 47N Easting (km)"); fig.colorbar(im,ax=ax,label="Def5 events / 100,000 evaluable h")
    axs[0].set_ylabel("UTM 47N Northing (km)"); fig.suptitle("Development-period Def5 rate: estimated surfaces, not observed rainfall fields"); fig.tight_layout(); fig.savefig(r4/"R4_idw_vs_ordinary_kriging.png",dpi=220); plt.close(fig)
    coverage=pd.DataFrame([{"Stations":len(rate),"Convex_Hull_Area_km2":float(ConvexHull(xy).volume/1e6),"Grid_Inside_Hull":int(inside.sum()),"Grid_Outside_Hull_Not_Interpolated":int((~inside).sum()),"Inside_Grid_P90_NearestStation_km":float(np.quantile(nearest[inside],.9)),"Variogram_Nugget":params[0],"Variogram_Sill":params[1],"Variogram_Range_m":params[2],"Preferred_By_RMSE":metrics.sort_values("RMSE").iloc[0]["Method"]}]); write_csv(coverage,r4/"R4_coverage_and_variogram_summary.csv")

    hashes_after = {str(p): sha256(p) for p in protected}
    if hashes_before != hashes_after: raise AssertionError("Protected canonical/frozen input changed during revision run")
    prov={"created_utc":datetime.now(timezone.utc).isoformat(),"workspace":str(HERE),"canonical_root":str(root),"canonical_access":"read-only",
          "python":sys.version,"platform":platform.platform(),"protected_input_hashes_before":hashes_before,"protected_input_hashes_after":hashes_after,
          "protected_inputs_unchanged":True,"rules":{"missing_is_never_zero":True,"2025_used_for_tuning":False,"frozen_outputs_written":False},
          "results":{"R1":r1_summary.iloc[0].to_dict(),"R2_annual_rows":len(annual),"R3_scenarios":summary,"R4_cv":metrics.to_dict("records")}}
    (dirs["provenance"]/"revision_run_provenance.json").write_text(json.dumps(prov,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    outputs=[p for p in HERE.rglob("*") if p.is_file() and p.name not in {"OUTPUT_SHA256SUMS.txt"}]
    (dirs["provenance"]/"OUTPUT_SHA256SUMS.txt").write_text("\n".join(f"{sha256(p)}  {p.relative_to(HERE).as_posix()}" for p in sorted(outputs))+"\n",encoding="utf-8")
    print(json.dumps({"status":"PASS","R1":r1_summary.iloc[0].to_dict(),"R4":metrics.to_dict("records")},default=str,indent=2))


if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--canonical-root",type=Path,required=True); args=ap.parse_args(); main(args.canonical_root.resolve())
