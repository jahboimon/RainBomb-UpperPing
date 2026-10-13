from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_temporal_roles_do_not_overlap():
    config = yaml.safe_load((ROOT / "config/splits.yaml").read_text(encoding="utf-8"))
    roles = {name: set(years) for name, years in config["roles"].items()}
    assert roles["Train"].isdisjoint(roles["Validation"])
    assert roles["Train"].isdisjoint(roles["Evaluation"])
    assert roles["Validation"].isdisjoint(roles["Evaluation"])
    assert roles == {"Train": {2020, 2021, 2022, 2023}, "Validation": {2024}, "Evaluation": {2025}}


def test_wet_season_is_frozen():
    config = yaml.safe_load((ROOT / "config/splits.yaml").read_text(encoding="utf-8"))
    assert config["season_months"] == [5, 6, 7, 8, 9, 10]
    assert config["time_reference"] == "Target_Datetime"
