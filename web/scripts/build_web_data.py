"""Build web-only derivatives for RainBomb Atlas from frozen read-only sources.

The script never writes to the File Geodatabase, layer files, or parquet inputs.
All outputs are recreated below dist/data/derived.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from rainbomb_paths import load_paths  # noqa: E402

PATHS = load_paths()
ROOT = REPO_ROOT / "web"
GDB = PATHS.web_source_root / "data" / "source_gdb" / "ML_Visualize.gdb"
LAYER_MANIFEST = PATHS.web_source_root / "data" / "source_gdb" / "layer_manifest.txt"
PREDICTION_PARQUET = (
    PATHS.final_analysis_root
    / "03_OUTPUT/06_Final_Model_Evaluation/Evaluation2025_Actual_vs_Predicted_v1.parquet"
)
TARGET_PARQUET = (
    PATHS.upstream_root
    / "Def5_Final_Target_2012_2025_v1/02_Def5_Extended_Target/RainBomb_Target_Def5_2012_2025_v1.parquet"
)

OUT = ROOT / "data" / "derived"
VECTOR_OUT = OUT / "vector"
RASTER_OUT = OUT / "raster"
STYLE_OUT = OUT / "style"


VECTOR_SPECS = {
    "station_metadata.geojson": {
        "layer": "Station_Metadata_Point",
        "fields": [
            "Station", "Station_Name", "Latitude", "Longitude", "Province", "District",
            "Subdistrict", "Basin_Name", "Sub_Basin_Name", "Station_Type", "Region",
            "Shared_Data", "Cleaned_Rows", "First_Datetime", "Last_Datetime",
            "Data_Coverage_Note", "Elevation_m",
        ],
    },
    "station_summary_2025.geojson": {
        "layer": "GIS_Station_Model_Summary_2025_Point",
        "fields": [
            "Station", "Latitude", "Longitude", "Elevation_m", "SUHI_5K",
            "BuiltUp_Binary", "Available_Hours", "Def5_Count", "Def5_Rate_per100k",
            "Mean_XGB_Score", "Median_XGB_Score", "TN", "FP", "FN", "TP",
            "FP_Rate", "Recall",
        ],
    },
    "def5_events_2025.geojson": {
        "layer": "GIS_Def5_Events_2025_XYTableToPoint",
        "fields": [
            "Station", "Latitude", "Longitude", "Datetime", "Target_Datetime",
            "Target_Def5_tPlus1", "XGB_Final_Score", "XGB_Predicted_0p5", "Outcome_0p5",
        ],
    },
    "station_shap_2025.geojson": {
        "layer": "GIS_Station_SHAP_Long_2025_XYTableToPoint",
        "fields": [
            "Station", "Latitude", "Longitude", "Feature", "Feature_Group", "N",
            "MeanAbsSHAP", "MedianAbsSHAP", "MeanSignedSHAP",
        ],
    },
    "admin_boundaries.geojson": {
        "layer": "adm_Bound",
        "fields": ["ADM1_EN", "ADM1_TH", "ADM1_PCODE", "ADM0_EN", "ADM0_TH"],
        "simplify": 0.0002,
    },
    "upper_ping_boundary.geojson": {
        "layer": "upper_ping_Bound",
        "fields": ["Shape_Length", "Shape_Area"],
    },
    "rivers_upper_ping.geojson": {
        "layer": "HydroRIVERS_upperPing",
        "fields": [
            "HYRIV_ID", "MAIN_RIV", "LENGTH_KM", "DIS_AV_CMS", "ORD_STRA",
            "ORD_CLAS", "MBASIN_E", "SB_NAME__1_First",
        ],
        "simplify": 0.0001,
    },
}


KDE_STYLES = {
    "KDE_Rain_t_MeanAbs_R20km_C1km": (0.0025530210696160793, "teal_a"),
    "KDE_q_850_MeanAbs_R20km_C1km": (0.004480093251913786, "teal_b"),
    "KDE_u_850_MeanAbs_R20km_C1km": (0.00492524541914463, "teal_b"),
    "KDE_u10_MeanAbs_R20km_C1km": (0.001594220520928502, "teal_a"),
    "KDE_Temp_t_MeanAbs_R20km_C1km": (0.001761085819453001, "teal_b"),
}

COLOR_RAMPS = {
    "teal_a": [
        (0.00, (255, 247, 255, 0)),
        (0.25, (126, 213, 255, 64)),
        (0.50, (0, 179, 255, 128)),
        (0.75, (0, 179, 204, 166)),
        (1.00, (0, 179, 128, 204)),
    ],
    "teal_b": [
        (0.00, (247, 252, 255, 0)),
        (0.25, (64, 199, 255, 64)),
        (0.50, (25, 168, 255, 128)),
        (0.75, (0, 130, 237, 166)),
        (1.00, (0, 80, 179, 191)),
    ],
    "elevation": [
        (0.00, (158, 178, 132, 255)),
        (0.25, (199, 207, 145, 255)),
        (0.50, (230, 214, 171, 255)),
        (0.75, (194, 155, 91, 255)),
        (1.00, (126, 86, 52, 255)),
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def write_geojson(frame: gpd.GeoDataFrame, path: Path) -> None:
    frame = frame.to_crs("EPSG:4326")
    pyogrio.write_dataframe(
        frame,
        path,
        driver="GeoJSON",
        layer_options={"RFC7946": "YES", "COORDINATE_PRECISION": "6"},
    )


def source_frame(layer: str, fields: list[str]) -> gpd.GeoDataFrame:
    return pyogrio.read_dataframe(GDB, layer=layer, columns=fields)


def build_vectors(target_2025: pd.DataFrame) -> list[dict]:
    records = []
    metadata = source_frame(
        VECTOR_SPECS["station_metadata.geojson"]["layer"],
        VECTOR_SPECS["station_metadata.geojson"]["fields"],
    )
    metadata_names = pd.DataFrame(metadata[["Station", "Station_Name", "Province", "District"]])

    for filename, spec in VECTOR_SPECS.items():
        frame = source_frame(spec["layer"], spec["fields"])
        notes = "Direct geometry/attribute conversion; no scientific recalculation."

        if filename == "station_summary_2025.geojson":
            frame = frame.merge(metadata_names, on="Station", how="left")
            notes = "Joined station name/province/district by authoritative Station key."

        if filename == "def5_events_2025.geojson":
            frame["Target_Datetime_key"] = frame["Target_Datetime"].dt.tz_localize(None).astype("datetime64[ns]")
            event_meta = target_2025.rename(columns={"Datetime": "Target_Datetime_key"})
            join_fields = [
                "Station", "Target_Datetime_key", "Rainfall", "P99_Frozen",
                "Antecedent_3h_Rain", "Post1_Rain", "Post2_Rain", "Post3_Rain",
                "Post3_Max", "Def5",
            ]
            frame = frame.merge(event_meta[join_fields], on=["Station", "Target_Datetime_key"], how="left", validate="one_to_one")
            frame = frame.merge(metadata_names, on="Station", how="left")
            if frame["Rainfall"].isna().any():
                missing = frame.loc[frame["Rainfall"].isna(), ["Station", "Target_Datetime"]]
                raise RuntimeError(f"Event metadata join incomplete: {missing.to_dict('records')}")
            frame = frame.drop(columns=["Target_Datetime_key"])
            notes = "Matched 2025 frozen Def5 event metadata by Station + event hour (GDB Target_Datetime = target Datetime)."

        simplify = spec.get("simplify")
        if simplify:
            frame = frame.to_crs("EPSG:4326")
            frame.geometry = frame.geometry.simplify(simplify, preserve_topology=True)
            notes += f" Display-copy geometry simplified at {simplify} degrees with topology preserved."

        output = VECTOR_OUT / filename
        write_geojson(frame, output)
        records.append({
            "derived_file": output.relative_to(ROOT).as_posix(),
            "source": f"ML_Visualize.gdb/{spec['layer']}",
            "source_crs": str(source_frame(spec["layer"], spec["fields"][:1]).crs),
            "web_crs": "EPSG:4326 (RFC 7946)",
            "feature_count": int(len(frame)),
            "fields": [column for column in frame.columns if column != "geometry"],
            "conversion": notes,
            "size_bytes": output.stat().st_size,
            "sha256": sha256(output),
        })

    return records


def write_color_table(path: Path, maximum: float, ramp_name: str, minimum: float = 0.0) -> None:
    lines = ["nv 0 0 0 0"]
    for position, rgba in COLOR_RAMPS[ramp_name]:
        value = minimum + (maximum - minimum) * position
        lines.append(f"{value:.15g} {rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=ROOT)


def gdal_json(dataset: str) -> dict:
    result = subprocess.run(
        ["gdalinfo", "-json", dataset], check=True, cwd=ROOT,
        capture_output=True, text=True, encoding="utf-8",
    )
    return json.loads(result.stdout)


def elevation_statistics(source: str) -> dict:
    """Calculate statistics from valid source cells, never from histogram counts."""
    with rasterio.open(source) as dataset:
        values = dataset.read(1, masked=True).compressed()
        percentiles = np.percentile(values, [2, 25, 50, 75, 98])
        return {
            "unit": "metres (authoritative source layer name: Elev_m)",
            "valid_cell_count": int(values.size),
            "minimum": float(values.min()),
            "p02": float(percentiles[0]),
            "p25": float(percentiles[1]),
            "mean": float(values.mean()),
            "median": float(percentiles[2]),
            "p75": float(percentiles[3]),
            "p98": float(percentiles[4]),
            "maximum": float(values.max()),
        }


def render_elevation_rgba(cog: Path, hillshade: Path, png: Path, stats: dict) -> None:
    """Create a browser-only hypsometric tint with subtle shaded relief."""
    with rasterio.open(cog) as dataset:
        elevation = dataset.read(1).astype(np.float32)
        valid = dataset.read_masks(1) > 0
        if dataset.nodata is not None:
            valid &= elevation != dataset.nodata
    with rasterio.open(hillshade) as dataset:
        shade = dataset.read(1).astype(np.float32) / 255.0

    display_min = stats["p02"]
    display_max = stats["p98"]
    normalized = np.clip((elevation - display_min) / (display_max - display_min), 0, 1)
    stops = np.array([item[0] for item in COLOR_RAMPS["elevation"]], dtype=np.float32)
    colors = np.array([item[1][:3] for item in COLOR_RAMPS["elevation"]], dtype=np.float32)
    rgb = np.stack([np.interp(normalized, stops, colors[:, band]) for band in range(3)], axis=-1)
    relief = 0.76 + 0.24 * shade
    rgb = np.clip(rgb * relief[..., None], 0, 255).astype(np.uint8)
    alpha = np.where(valid, 255, 0).astype(np.uint8)
    Image.fromarray(np.dstack([rgb, alpha]), "RGBA").save(png, optimize=True)


def build_rasters() -> list[dict]:
    records = []
    specs = {
        **{name: {"max": max_value, "ramp": ramp} for name, (max_value, ramp) in KDE_STYLES.items()},
        "Elev_m": {"ramp": "elevation", "width": 1400},
    }

    for layer, spec in specs.items():
        safe_name = layer.lower()
        cog = RASTER_OUT / f"{safe_name}_wgs84.cog.tif"
        png = RASTER_OUT / f"{safe_name}_styled.png"
        color_table = STYLE_OUT / f"{safe_name}_color.txt"
        source = f'OpenFileGDB:"{GDB}":{layer}'
        is_elevation = layer == "Elev_m"
        stats = elevation_statistics(f"OpenFileGDB:{GDB}:{layer}") if is_elevation else None

        write_color_table(
            color_table,
            stats["p98"] if is_elevation else spec["max"],
            spec["ramp"],
            stats["p02"] if is_elevation else 0.0,
        )
        command = [
            "gdalwarp", "-overwrite", "-of", "COG", "-t_srs", "EPSG:4326",
            "-r", "bilinear" if is_elevation else "near", "-dstnodata", "-9999",
            "-co", "COMPRESS=DEFLATE",
        ]
        if spec.get("width"):
            command.extend(["-ts", str(spec["width"]), "0"])
        command.extend([source, str(cog)])
        run(command)
        hillshade = None
        if is_elevation:
            hillshade = RASTER_OUT / "elev_m_hillshade_wgs84.tif"
            run([
                "gdaldem", "hillshade", str(cog), str(hillshade), "-of", "GTiff",
                "-az", "315", "-alt", "42", "-s", "111120", "-compute_edges",
            ])
            render_elevation_rgba(cog, hillshade, png, stats)
        else:
            run(["gdaldem", "color-relief", str(cog), str(color_table), str(png), "-alpha"])
        Path(f"{png}.aux.xml").unlink(missing_ok=True)

        info = gdal_json(str(cog))
        bounds = info.get("wgs84Extent", {}).get("coordinates")
        if bounds:
            ring = bounds[0]
            west = min(point[0] for point in ring)
            east = max(point[0] for point in ring)
            south = min(point[1] for point in ring)
            north = max(point[1] for point in ring)
        else:
            corners = info["cornerCoordinates"]
            west, north = corners["upperLeft"]
            east, south = corners["lowerRight"]

        source_info = gdal_json(source)
        source_band = source_info["bands"][0]
        output_band = info["bands"][0]
        record = {
            "source": f"ML_Visualize.gdb/{layer}",
            "source_type": "File Geodatabase raster",
            "source_crs": "EPSG:32647",
            "source_size": source_info["size"],
            "source_nodata": source_band.get("noDataValue"),
            "derived_cog": cog.relative_to(ROOT).as_posix(),
            "derived_png": png.relative_to(ROOT).as_posix(),
            "style_table": color_table.relative_to(ROOT).as_posix(),
            "web_crs": "EPSG:4326",
            "web_size": info["size"],
            "web_nodata": output_band.get("noDataValue"),
            "web_bounds": [west, south, east, north],
            "resampling": (
                "bilinear display resampling for continuous elevation"
                if is_elevation else
                "nearest-neighbour display resampling, matching ArcGIS .lyrx"
            ),
            "styling": (
                "ArcGIS .lyrx minimum–maximum stretch translated to a continuous RGBA ramp"
                if not is_elevation
                else "Natural continuous hypsometric tint informed by the ArcGIS .lyrx, with a 2nd–98th percentile display stretch and 24% hillshade blend; source values remain unchanged"
            ),
            "display_min": stats["p02"] if is_elevation else 0.0,
            "display_max": stats["p98"] if is_elevation else spec["max"],
            "source_statistics": stats,
            "hillshade": (
                {
                    "derived_file": hillshade.relative_to(ROOT).as_posix(),
                    "azimuth_degrees": 315,
                    "altitude_degrees": 42,
                    "blend_strength": 0.24,
                    "size_bytes": hillshade.stat().st_size,
                    "sha256": sha256(hillshade),
                }
                if hillshade else None
            ),
            "bandwidth_m": 20000 if layer.startswith("KDE_") else None,
            "cell_size_m": 1000 if layer.startswith("KDE_") else None,
            "cog_size_bytes": cog.stat().st_size,
            "png_size_bytes": png.stat().st_size,
            "cog_sha256": sha256(cog),
            "png_sha256": sha256(png),
            "style_size_bytes": color_table.stat().st_size,
            "style_sha256": sha256(color_table),
        }
        records.append(record)

    # GDAL may create read-only metadata sidecars while inspecting the derived
    # rasters. They are not browser assets and make the web-data bundle noisy.
    for sidecar in RASTER_OUT.glob("*.aux.xml"):
        sidecar.unlink()

    return records


def build_model_products(prediction: pd.DataFrame) -> list[dict]:
    hourly = (
        prediction.groupby("Target_Datetime", sort=True)
        .agg(
            Actual_Count=("Target_Def5_tPlus1", "sum"),
            Predicted_Count=("XGB_Predicted_0p5", "sum"),
            Evaluated_Rows=("Station", "size"),
            Mean_Score=("XGB_Final_Score", "mean"),
        )
        .reset_index()
    )
    hourly_records = [
        {column: json_value(value) for column, value in row.items()}
        for row in hourly.to_dict("records")
    ]
    hourly_path = OUT / "model_hourly_2025.json"
    hourly_path.write_text(json.dumps(hourly_records, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    counts = prediction["Outcome_0p5"].value_counts().to_dict()
    tp, fn, fp, tn = (int(counts[key]) for key in ("TP", "FN", "FP", "TN"))
    metrics = {
        "evaluation_rows": int(len(prediction)),
        "evaluation_period": "May–October 2025",
        "stations": int(prediction["Station"].nunique()),
        "actual_def5": int(prediction["Target_Def5_tPlus1"].sum()),
        "predicted_positive": int(prediction["XGB_Predicted_0p5"].sum()),
        "average_precision": float(average_precision_score(prediction["Target_Def5_tPlus1"], prediction["XGB_Final_Score"])),
        "roc_auc": float(roc_auc_score(prediction["Target_Def5_tPlus1"], prediction["XGB_Final_Score"])),
        "prevalence": float(prediction["Target_Def5_tPlus1"].mean()),
        "precision_at_0_5": tp / (tp + fp),
        "recall_at_0_5": tp / (tp + fn),
        "threshold": 0.5,
        "confusion_matrix": {"TP": tp, "FN": fn, "FP": fp, "TN": tn},
        "orientation": "rows = Actual; columns = Predicted",
        "timezone": "UNKNOWN — retained without inference",
    }
    metrics_path = OUT / "model_metrics_2025.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    return [
        {
            "derived_file": hourly_path.relative_to(ROOT).as_posix(),
            "source": str(PREDICTION_PARQUET),
            "record_count": len(hourly_records),
            "conversion": "Specified hourly aggregation: sum of actual-positive and predicted-positive station rows by Target_Datetime.",
            "size_bytes": hourly_path.stat().st_size,
            "sha256": sha256(hourly_path),
        },
        {
            "derived_file": metrics_path.relative_to(ROOT).as_posix(),
            "source": str(PREDICTION_PARQUET),
            "record_count": 1,
            "conversion": "Verification metrics calculated from frozen prediction rows; no model refit or threshold change.",
            "size_bytes": metrics_path.stat().st_size,
            "sha256": sha256(metrics_path),
        },
    ]


def main() -> None:
    for directory in (VECTOR_OUT, RASTER_OUT, STYLE_OUT):
        directory.mkdir(parents=True, exist_ok=True)

    for source in (GDB, LAYER_MANIFEST, PREDICTION_PARQUET, TARGET_PARQUET):
        if not source.exists():
            raise FileNotFoundError(source)

    prediction = pd.read_parquet(PREDICTION_PARQUET)
    target = pd.read_parquet(TARGET_PARQUET)
    target_2025 = target[
        (target["Datetime"].dt.year == 2025)
        & target["Datetime"].dt.month.between(5, 10)
    ].copy()

    if len(prediction) != 121_296 or int(prediction["Target_Def5_tPlus1"].sum()) != 95:
        raise RuntimeError("Frozen Evaluation 2025 prediction parquet does not match locked row/event counts.")
    if len(target) != 1_123:
        raise RuntimeError("Frozen Def5 target parquet does not match the locked 1,123 station-hours.")

    vector_records = build_vectors(target_2025)
    tabular_records = build_model_products(prediction)
    raster_records = build_rasters()

    manifest = {
        "title": "RainBomb Atlas derived web-data manifest",
        "scope": "Evaluation 2025 only",
        "generated_by": "scripts/build_web_data.py",
        "scientific_operations": "None. Operations are format conversion, coordinate reprojection, documented display simplification/resampling, exact key joins, and requested hourly visualization aggregation.",
        "source_policy": "All sources were opened read-only; no source file was modified, renamed, moved, or deleted.",
        "timezone": "Unknown in source manifest; no timezone was inferred.",
        "vectors": vector_records,
        "rasters": raster_records,
        "tabular": tabular_records,
        "unavailable": [
            {
                "component": "Event E−6:E+6 hydrograph",
                "reason": "Full hourly context table is not present. Available event metadata provides Rain(E), frozen P99, Antecedent_3h_Rain, Post1–Post3, and Post3_Max only."
            },
            {
                "component": "Satellite imagery",
                "reason": "No authorized credential-free imagery source was supplied."
            },
            {
                "component": "2012–2024 interactive period summaries",
                "reason": "Deferred by user instruction; this pass is Evaluation 2025 only."
            }
        ]
    }
    manifest_path = OUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "vectors": len(vector_records),
        "rasters": len(raster_records),
        "tabular": len(tabular_records),
        "manifest": str(manifest_path),
    }, indent=2))


if __name__ == "__main__":
    main()
