"""Central path configuration for copied Rain Bomb scripts.

Only path resolution is centralized here; scientific calculations remain in
the versioned scripts copied from the frozen analysis workspace.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ResearchPaths:
    repo_root: Path
    runtime_root: Path
    upstream_root: Path
    gis_root: Path
    final_analysis_root: Path
    web_source_root: Path
    xgb_site_packages: Path


def _resolve(value: str, base: Path) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    return expanded.resolve() if expanded.is_absolute() else (base / expanded).resolve()


def load_paths() -> ResearchPaths:
    config_path = Path(os.environ.get("RAINBOMB_PATHS", REPO_ROOT / "config" / "paths.yaml"))
    if not config_path.exists():
        config_path = REPO_ROOT / "config" / "paths.example.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    paths = payload.get("paths", {})
    return ResearchPaths(
        repo_root=REPO_ROOT,
        runtime_root=_resolve(paths.get("runtime_root", "runtime"), REPO_ROOT),
        upstream_root=_resolve(paths.get("upstream_root", "../Workplace_Code"), REPO_ROOT),
        gis_root=_resolve(paths.get("gis_root", "../Workplace_GIS"), REPO_ROOT),
        final_analysis_root=_resolve(
            paths.get("final_analysis_root", "../RainBomb_FinalAnalysis_v2"), REPO_ROOT
        ),
        web_source_root=_resolve(
            paths.get("web_source_root", "../Workplace_Code_Redesign"), REPO_ROOT
        ),
        xgb_site_packages=_resolve(
            paths.get("xgb_site_packages", "../Workplace_Code/.venv_xgb/Lib/site-packages"),
            REPO_ROOT,
        ),
    )
