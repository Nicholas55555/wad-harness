"""One command: raw data -> cache -> results.

  python scripts/run_all.py               run every phase whose outputs are missing
  python scripts/run_all.py --force       rerun every phase, including detection
  python scripts/run_all.py --from-cache  skip phases 1-2 (data and detection; the cache must
                                          exist) and recompute everything downstream from it
  python scripts/run_all.py --smoke       the 20-image smoke test (cache/smoke, results/smoke)

Phases: 1 data, 2 detection (+ EXIF frame check), 3 matching/curves/go-no-go, 4 CRC,
5 extras/figures/report, 6a hazard scalar (pixels only). Fixed seeds throughout; two runs give
identical CSVs. Finishes by printing the verdict and the path to results.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wad import crc, curves, data, detect, extras, figures, hazard, hazard_crc, match, report, sensitivity  # noqa: E402
from wad.config import cache_path, load_config, results_path, write_json  # noqa: E402


def done(paths) -> bool:
    return bool(paths) and all(Path(p).exists() for p in paths)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true", help="rerun every phase even if its outputs exist")
    ap.add_argument("--from-cache", action="store_true", help="skip data and detection; recompute from the cache")
    ap.add_argument("--smoke", action="store_true", help="20-image smoke test")
    ap.add_argument("--no-hazard", action="store_true", help="skip phase 6a")
    args = ap.parse_args(argv)
    os.chdir(ROOT)  # every path in the cache and config is relative to the repo root
    cfg = load_config(args.config, smoke=args.smoke)
    flags = ["--smoke"] if args.smoke else []
    if args.config:
        flags += ["--config", args.config]
    t0 = time.perf_counter()
    timing = {}

    def phase(name, outputs, fn, skip=False):
        t = time.perf_counter()
        if skip:
            print(f"[{name}] skipped (--from-cache)")
            return
        if not args.force and not args.from_cache and done(outputs):
            print(f"[{name}] outputs exist, skipping (use --force to rerun)")
            return
        print(f"[{name}] running")
        fn()
        timing[name] = round(time.perf_counter() - t, 1)

    c, r = cfg.paths.cache, cfg.paths.results
    if args.from_cache:
        need = [c / "images.parquet", c / "gt.parquet", c / "detections.parquet"]
        if not done(need):
            raise SystemExit(f"--from-cache needs {', '.join(str(p) for p in need)}; run without it first")

    phase("1 data", [c / "images.parquet", c / "gt.parquet"], lambda: data.main(flags), skip=args.from_cache)
    phase("2 detect", [c / "detections.parquet", c / "detections.meta.json"] + ([] if args.smoke else [c / "exif_check.json"]),
          lambda: detect.main(flags + (["--force"] if args.force else [])), skip=args.from_cache)
    phase("3 curves", [c / "matches_gt.parquet", r / "summary_phase3.csv", r / "go_no_go.json"],
          lambda: (match.main(flags), curves.main(flags)))
    phase("4 crc", [r / "crc_summary.csv", r / "crc_perclass.csv"], lambda: crc.main(flags))
    if not args.no_hazard:
        # h is a function of pixels only, so its cache counts as data, not as a detector output
        phase("6a hazard features", [c / "hazard.parquet"], lambda: hazard.main(flags),
              skip=args.from_cache and (c / "hazard.parquet").exists())
        phase("6a hazard bins", [r / "hazard_summary.csv"], lambda: hazard_crc.main(flags))
    phase("5 extras", [r / "extra" / "go_no_go_bootstrap.csv"], lambda: extras.main(flags))
    phase("5 figures", [r / "figures" / "fig3_main.png"], lambda: figures.main(flags))
    if not args.smoke and (c / "sensitivity" / "yolov8s_1280" / "detections.parquet").exists():
        # the imgsz-1280 detections are produced by `python -m wad.sensitivity` (~7 min); once
        # cached, everything downstream of them is recomputed here like the primary analysis
        phase("5 sensitivity (cached 1280 detections)", [r / "sensitivity" / "yolov8s_1280" / "crc_summary.csv"],
              lambda: sensitivity.run(cfg, "yolov8s_1280", detect=False))
    phase("5 report", [], lambda: report.main(flags))
    timing["total"] = round(time.perf_counter() - t0, 1)
    write_json(results_path(cfg, "run_timing.json"), timing)

    v = json.loads((r / "go_no_go.json").read_text(encoding="utf-8"))
    print()
    print(v["line"])
    print(f"results: {results_path(cfg, 'results.md')}  ({timing['total']:.0f} s)")


if __name__ == "__main__":
    main()
