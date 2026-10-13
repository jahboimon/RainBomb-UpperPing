from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_final_training_excludes_evaluation():
    text = (ROOT / "src/06_final_model/train_final_xgb.py").read_text(encoding="utf-8")
    assert 'filters=[("Role", "in", ["Train", "Validation"])]' in text
    assert '"Evaluation_Data_Used_In_Fit": False' in text


def test_evaluation_overwrite_guard_is_preserved():
    text = (ROOT / "src/06_final_model/evaluate_2025.py").read_text(encoding="utf-8")
    assert "refuses to run if any Evaluation output already exists" in text
    assert "Evaluation outputs already exist; refusing to score Evaluation 2025 more than once" in text


def test_feature_set_is_exactly_compact20():
    import yaml
    payload = yaml.safe_load((ROOT / "config/features.yaml").read_text(encoding="utf-8"))
    assert payload["feature_set"] == "Compact20"
    assert len(payload["predictors"]) == 20
    assert len(set(payload["predictors"])) == 20
