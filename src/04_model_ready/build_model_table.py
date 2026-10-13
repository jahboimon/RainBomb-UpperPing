"""Build the frozen Phase 5 compact-20 model-ready seasonal dataset.

This script changes no Phase 3 row, target, timestamp, or temporal role.  It
only selects the frozen predictors, applies the approved binary LULC mapping,
and performs strict integrity checks before and after serialization.
"""

from __future__ import annotations
import sys as _sys
from pathlib import Path as _ConfigPath
_sys.path.insert(0, str(_ConfigPath(__file__).resolve().parents[1]))
from rainbomb_paths import load_paths as _load_paths
PATHS = _load_paths()

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = PATHS.runtime_root
SOURCE = (
    ROOT
    / "03_OUTPUT"
    / "03_Seasonal_Modeling_Universe"
    / "Seasonal_Modeling_Universe_v1.parquet"
)
OUT = ROOT / "03_OUTPUT" / "05_Model_Comparison"
OUTPUT = OUT / "ModelReady_Seasonal_Compact20_v1.parquet"

IDENTIFIERS = ["Station", "Datetime", "Target_Datetime", "Role"]
TARGET = "Target_Def5_tPlus1"
NUMERIC_PREDICTORS = [
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
    "d2m",
    "u10",
    "v10",
    "q_850",
    "u_850",
    "v_850",
]
PREDICTORS = NUMERIC_PREDICTORS[:13] + ["BuiltUp_Binary"] + NUMERIC_PREDICTORS[13:]
EXPECTED = {
    "Train": {"rows": 525_474, "positives": 312, "stations": 40},
    "Validation": {"rows": 129_671, "positives": 74, "stations": 39},
    "Evaluation": {"rows": 121_296, "positives": 95, "stations": 38},
}
EXPECTED_TOTAL_ROWS = 776_441
ALLOWED_LULC = set(range(1, 11))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_population(frame: pd.DataFrame, label: str) -> None:
    if len(frame) != EXPECTED_TOTAL_ROWS:
        raise RuntimeError(f"{label}: expected {EXPECTED_TOTAL_ROWS:,} rows, found {len(frame):,}.")
    if set(frame["Role"].unique()) != set(EXPECTED):
        raise RuntimeError(f"{label}: unexpected temporal roles {sorted(frame['Role'].unique())}.")
    for role, expected in EXPECTED.items():
        part = frame.loc[frame["Role"].eq(role)]
        observed = {
            "rows": len(part),
            "positives": int(part[TARGET].sum()),
            "stations": int(part["Station"].nunique()),
        }
        if observed != expected:
            raise RuntimeError(f"{label}: {role} mismatch; expected {expected}, observed {observed}.")


def verify_integrity(frame: pd.DataFrame, label: str) -> None:
    verify_population(frame, label)
    if frame.duplicated(["Station", "Datetime"]).any():
        raise RuntimeError(f"{label}: duplicate Station + Datetime keys found.")
    if frame.duplicated(["Station", "Target_Datetime"]).any():
        raise RuntimeError(f"{label}: duplicate Station + Target_Datetime keys found.")
    lead = frame["Target_Datetime"] - frame["Datetime"]
    if not lead.eq(pd.Timedelta(hours=1)).all():
        raise RuntimeError(f"{label}: Target_Datetime is not exactly Datetime + 1 hour throughout.")
    if not frame["Target_Datetime"].dt.month.between(5, 10).all():
        raise RuntimeError(f"{label}: target month outside May-October found.")
    if not set(frame[TARGET].unique()).issubset({0, 1}):
        raise RuntimeError(f"{label}: target is not binary.")
    if frame[TARGET].isna().any():
        raise RuntimeError(f"{label}: missing target found.")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_hash_before = sha256(SOURCE)
    source_columns = IDENTIFIERS + [TARGET] + NUMERIC_PREDICTORS + ["LULC_Class"]
    source = pd.read_parquet(SOURCE, columns=source_columns)
    verify_integrity(source, "Frozen Phase 3 source")

    observed_lulc = set(source["LULC_Class"].dropna().astype(int).unique())
    if not observed_lulc.issubset(ALLOWED_LULC) or source["LULC_Class"].isna().any():
        raise RuntimeError(f"Unexpected or missing LULC class values: {sorted(observed_lulc)}")

    compact = source[IDENTIFIERS + [TARGET] + NUMERIC_PREDICTORS + ["LULC_Class"]].copy()
    compact["BuiltUp_Binary"] = compact["LULC_Class"].eq(9).astype("int8")
    compact = compact.rename(columns={"LULC_Class": "LULC_Class_Source"})
    compact = compact[
        IDENTIFIERS + [TARGET] + PREDICTORS + ["LULC_Class_Source"]
    ]

    verify_integrity(compact, "Constructed compact dataset")
    values = compact[PREDICTORS].to_numpy(dtype="float64", copy=False)
    if not np.isfinite(values).all():
        bad = (~np.isfinite(values)).sum(axis=0)
        detail = {name: int(count) for name, count in zip(PREDICTORS, bad) if count}
        raise RuntimeError(f"Missing or non-finite compact predictors found: {detail}")
    if set(compact["BuiltUp_Binary"].unique()) != {0, 1}:
        raise RuntimeError("BuiltUp_Binary does not contain exactly the values 0 and 1.")

    source_keys = source[IDENTIFIERS + [TARGET]].reset_index(drop=True)
    compact_keys = compact[IDENTIFIERS + [TARGET]].reset_index(drop=True)
    if not source_keys.equals(compact_keys):
        raise RuntimeError("Phase 3 keys, roles, or target values changed during construction.")

    compact.to_parquet(OUTPUT, index=False, compression="zstd")
    written = pd.read_parquet(OUTPUT)
    verify_integrity(written, "Serialized compact dataset")
    if list(written.columns) != list(compact.columns):
        raise RuntimeError("Serialized compact dataset column order changed.")
    if not written[IDENTIFIERS + [TARGET]].equals(compact_keys):
        raise RuntimeError("Serialized keys, roles, or target values differ from Phase 3.")
    if not np.isfinite(written[PREDICTORS].to_numpy(dtype="float64", copy=False)).all():
        raise RuntimeError("Serialized compact dataset contains a missing or non-finite predictor.")

    source_hash_after = sha256(SOURCE)
    if source_hash_before != source_hash_after:
        raise RuntimeError("Frozen Phase 3 source hash changed during Phase 5A.")

    print(f"Integrity status: PASS")
    print(f"Rows: {len(written):,}")
    print(f"Model predictors: {len(PREDICTORS)}")
    print(f"Columns including audit LULC: {len(written.columns)}")
    for role in EXPECTED:
        part = written.loc[written["Role"].eq(role)]
        print(
            f"{role}: {len(part):,} rows / {int(part[TARGET].sum()):,} positives / "
            f"{part['Station'].nunique():,} stations"
        )
    print(f"Frozen source SHA256: {source_hash_after}")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
