from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def manifest():
    return yaml.safe_load((ROOT / "RELEASE_MANIFEST.yaml").read_text(encoding="utf-8"))["artifacts"]


def test_frozen_release_counts():
    by_id = {item["id"]: item for item in manifest()}
    assert by_id["def5_target"]["rows"] == 1_123
    assert by_id["evaluation_2025_predictions"]["rows"] == 121_296
    assert by_id["evaluation_2025_predictions"]["positives"] == 95


def test_bundled_metrics_match_release():
    with (ROOT / "outputs/tables/Evaluation2025_Metrics_v1.csv").open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert int(row["Evaluation_Rows"]) == 121_296
    assert int(row["Evaluation_Positives"]) == 95
    assert row["Evaluation_Based_Tuning_Performed"] == "False"


def test_final_model_has_53_rounds():
    model = json.loads((ROOT / "outputs/models/XGB_Final_Compact20_v1.json").read_text(encoding="utf-8"))
    trees = model["learner"]["gradient_booster"]["model"]["trees"]
    assert len(trees) == 53
