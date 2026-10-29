"""Run 2 (O2) and the clear-weather control (PREREG_run2.md).

config_run2.yaml lists detectors, losses and label variants; this module expands them into one
analysis per combination, each with its own cache and results folder:

  DAWN      results/run2/dawn/<detector>__<loss>__<labels>/
  BDD100K   results/run2/bdd/<detector>__<loss>/

Every analysis runs the same code as run 1 (matching, curves, CRC), plus the clear-calibrated
policy (BDD100K), the confirmed-tier / certified-miss table and the precision-recall tables.

  python -m wad.run2 detect     # detections for every detector x dataset (GPU where configured)
  python -m wad.run2 analyse    # all analyses from the cached detections
  python -m wad.run2 report     # results/run2/run2.md and figures
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

from wad.config import REPO, Cfg, load_config


def load_run2(path: str | Path | None = None) -> Cfg:
    path = Path(path) if path else REPO / "config_run2.yaml"
    with open(path, encoding="utf-8") as f:
        return Cfg(yaml.safe_load(f))


def _detector(v: Cfg, d: dict) -> None:
    v.detector.update(name=d["name"], imgsz=d["imgsz"], device=d["device"])


def dawn_cfg(r2: Cfg, det_id: str, loss: str, labels: str) -> Cfg:
    base = load_config(REPO / r2.base_config)
    v = copy.deepcopy(base)
    d = r2.detectors[det_id]
    _detector(v, d)
    tag = f"{det_id}__{loss}__{labels}"
    v.paths["data_cache"] = base.paths.cache  # run 1's images.parquet / gt.parquet
    v.paths["detections"] = REPO / d["detections"]
    v.paths["cache"] = REPO / "cache" / "run2" / "dawn" / tag
    v.paths["results"] = REPO / "results" / "run2" / "dawn" / tag
    v["match_groups"] = r2.losses[loss]
    v.data["labels"] = labels
    v["tiers"] = r2.tiers
    v["tag"], v["det_id"], v["loss"], v["dataset_name"] = tag, det_id, loss, "dawn"
    return v


def bdd_cfg(r2: Cfg, det_id: str, loss: str) -> Cfg:
    base = load_config(REPO / r2.bdd_config)
    v = copy.deepcopy(base)
    d = dict(r2.detectors[det_id])
    d["device"] = "cuda:0"  # both detectors run on the GPU for BDD100K
    _detector(v, d)
    tag = f"{det_id}__{loss}"
    v.paths["detections"] = REPO / "cache" / "bdd" / "det" / det_id
    v.paths["cache"] = REPO / "cache" / "run2" / "bdd" / tag
    v.paths["results"] = REPO / "results" / "run2" / "bdd" / tag
    v["match_groups"] = r2.losses[loss]
    v.data["labels"] = "all"
    v["tiers"] = r2.tiers
    v["tag"], v["det_id"], v["loss"], v["dataset_name"] = tag, det_id, loss, "bdd"
    return v


def detect_all(r2: Cfg, only: str | None = None) -> None:
    from wad import detect

    jobs = [("dawn", dawn_cfg(r2, d, "class", "all")) for d in r2.detectors if d != "v8s_640"]
    jobs += [("bdd", bdd_cfg(r2, d, "class")) for d in r2.bdd.detectors]
    for name, v in jobs:
        if only and only not in (name, v.det_id, f"{name}:{v.det_id}"):
            continue
        print(f"== detect {name} {v.det_id} on {v.detector.device}", flush=True)
        detect.run(v)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["detect", "analyse", "report"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--only", default=None, help="restrict to a dataset (dawn/bdd) or a detector id")
    args = ap.parse_args(argv)
    r2 = load_run2(args.config)
    if args.command == "detect":
        detect_all(r2, args.only)
    elif args.command == "analyse":
        from wad.run2_analysis import analyse_all

        analyse_all(r2, args.only)
    else:
        from wad.run2_report import write_report

        write_report(r2)


if __name__ == "__main__":
    main()
