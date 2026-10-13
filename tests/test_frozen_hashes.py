from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_bundled_frozen_hashes():
    artifacts = yaml.safe_load((ROOT / "RELEASE_MANIFEST.yaml").read_text(encoding="utf-8"))["artifacts"]
    checked = 0
    for artifact in artifacts:
        repo_path = artifact.get("repo_path")
        if not repo_path:
            continue
        path = ROOT / repo_path
        assert path.is_file(), repo_path
        assert sha256(path) == artifact["sha256"], artifact["id"]
        checked += 1
    assert checked >= 3
