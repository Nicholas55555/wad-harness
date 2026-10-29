"""Load config.yaml and derive the fixed quantities every phase shares."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
THRESH_CACHE = REPO / "cache" / "thresh"  # threshold work: train detections, labels, model caches
THRESH_RESULTS = REPO / "results" / "threshold"  # threshold work: tables, figures, dial page
CANONICAL = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]


class Cfg(dict):
    """A dict with attribute access, applied recursively."""

    def __init__(self, d=None):
        super().__init__()
        for k, v in (d or {}).items():
            self[k] = Cfg(v) if isinstance(v, dict) else v

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e


def load_config(path: str | Path | None = None, smoke: bool = False) -> Cfg:
    path = Path(path) if path else REPO / "config.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = Cfg(yaml.safe_load(f))
    cfg["smoke_mode"] = smoke
    cfg.setdefault("dataset", "dawn")
    for k in ("data", "cache", "results", "weights"):
        cfg.paths[k] = REPO / cfg.paths[k]
    if smoke:
        # Smoke runs never touch the real cache or results.
        cfg.paths["cache"] = cfg.paths.cache / "smoke"
        cfg.paths["results"] = cfg.paths.results / "smoke"
    # Where the dataset tables (images/gt) and the detections live; by default both sit in the
    # analysis cache. Run 2 points several analyses at one detection cache.
    for k in ("data_cache", "detections"):
        cfg.paths[k] = REPO / cfg.paths[k] if k in cfg.paths and not smoke else cfg.paths.cache
    return cfg


def tau_grid(cfg: Cfg) -> np.ndarray:
    """200 thresholds: log-spaced on [log_start, log_end], then linear on (log_end, lin_end]."""
    g = cfg.tau_grid
    log = np.geomspace(g.log_start, g.log_end, g.n_log)
    lin = np.linspace(g.log_end, g.lin_end, g.n_lin + 1)[1:]
    grid = np.concatenate([log, lin])
    assert np.all(np.diff(grid) > 0)
    return grid


def bins(cfg: Cfg) -> list[str]:
    return list(cfg.data.bins.keys())


def coco_to_name(cfg: Cfg) -> dict[int, str]:
    return {int(v): k for k, v in cfg.classes.items()}


def cache_path(cfg: Cfg, name: str) -> Path:
    p = cfg.paths.cache
    p.mkdir(parents=True, exist_ok=True)
    return p / name


def data_path(cfg: Cfg, name: str) -> Path:
    p = cfg.paths.data_cache
    p.mkdir(parents=True, exist_ok=True)
    return p / name


def det_path(cfg: Cfg, name: str) -> Path:
    p = cfg.paths.detections
    p.mkdir(parents=True, exist_ok=True)
    return p / name


def results_path(cfg: Cfg, name: str) -> Path:
    p = cfg.paths.results / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    tmp.replace(path)
