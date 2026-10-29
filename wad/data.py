"""Dataset loader -> gt.parquet (one row per box) and images.parquet (one row per image).

Two datasets share one schema: DAWN (Pascal VOC, one folder per weather class; config.yaml) and
BDD100K (one JSON per image with weather and time-of-day tags; config_bdd.yaml), where a bin is
a set of (weather, time-of-day) tag pairs.

Run: python -m wad.data [--smoke] [--config config_bdd.yaml]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from wad.config import CANONICAL, REPO, Cfg, data_path, load_config, write_json
from wad.imageio import exif_orientation, load_rgb


def parse_voc(xml_path: Path) -> tuple[int, int, list[dict]]:
    root = ET.parse(xml_path).getroot()
    w = int(float(root.findtext("size/width")))
    h = int(float(root.findtext("size/height")))
    objs = []
    for o in root.findall("object"):
        b = o.find("bndbox")
        objs.append(
            dict(
                cls=o.findtext("name").strip().lower(),
                x1=float(b.findtext("xmin")),
                y1=float(b.findtext("ymin")),
                x2=float(b.findtext("xmax")),
                y2=float(b.findtext("ymax")),
                difficult=int(o.findtext("difficult") or 0),
                truncated=int(o.findtext("truncated") or 0),
            )
        )
    return w, h, objs


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def near_duplicate_pairs(images: pd.DataFrame, max_dist: int) -> list[tuple[str, str, int]]:
    """All image pairs whose 64-bit perceptual hashes differ in <= max_dist bits.

    Mirror-invariant: the distance is the smaller of hash(a) vs hash(b) and hash(a) vs
    hash(mirror(b)). DAWN contains copies whose stored pixels are identical but whose EXIF tag
    mirrors one of them, so in the annotation frame they are mirror images of each other.
    """
    ids = images.image_id.to_numpy()
    hashes = np.array([int(h, 16) for h in images.phash], dtype=np.uint64)
    mirrors = np.array([int(h, 16) for h in images.phash_mirror], dtype=np.uint64)
    bits = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    def popcount(x: np.ndarray) -> np.ndarray:
        return bits[x.view(np.uint8)].reshape(-1, 8).sum(axis=1)

    pairs = []
    for i in range(len(ids) - 1):
        d = np.minimum(popcount(np.bitwise_xor(hashes[i + 1 :], hashes[i])),
                       popcount(np.bitwise_xor(mirrors[i + 1 :], hashes[i])))
        for j in np.nonzero(d <= max_dist)[0]:
            pairs.append((ids[i], ids[i + 1 + j], int(d[j])))
    return pairs


def pixel_correlation(path_a: str, path_b: str, exif: str) -> float:
    """Mirror-aware similarity check for a hash pair: correlation of 64x64 grey images, the larger
    of (a, b) and (a, mirror(b)). 1.0 means the same picture up to resizing/re-encoding."""

    def grey(p, mirror=False):
        im = load_rgb(p, exif).convert("L")
        if mirror:
            im = ImageOps.mirror(im)
        a = np.asarray(im.resize((64, 64), Image.BILINEAR), dtype=float)
        return (a - a.mean()) / (a.std() + 1e-9)

    a = grey(path_a)
    return float(max((a * grey(path_b)).mean(), (a * grey(path_b, mirror=True)).mean()))


def components(nodes: list[str], edges: list[tuple[str, str]]) -> dict[str, int]:
    """Union-find connected components; returns node -> component index (by first member)."""
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = sorted({find(n) for n in nodes})
    index = {r: i for i, r in enumerate(roots)}
    return {n: index[find(n)] for n in nodes}


def collect_dawn(cfg: Cfg, tally: Counter):
    """(image_id, jpg, bin, (w, h) from the XML, objects) for every DAWN image."""
    for weather, folder in cfg.data.bins.items():
        ann_dir = cfg.paths.data / folder / cfg.data.ann_subdir.format(folder=folder)
        jpgs = sorted((cfg.paths.data / folder).glob("*.jpg"))
        xmls = {p.stem: p for p in ann_dir.glob("*.xml")}
        missing = [j.stem for j in jpgs if j.stem not in xmls]
        if missing:
            raise FileNotFoundError(f"{len(missing)} images in {folder} lack XML, e.g. {missing[:3]}")
        for jpg in jpgs:
            w, h, objs = parse_voc(xmls[jpg.stem])
            yield f"{weather}/{jpg.stem}", jpg, weather, (w, h), objs


def collect_bdd(cfg: Cfg, tally: Counter):
    """(image_id, jpg, bin, None, objects) for every BDD100K image whose (weather, time-of-day)
    tags fall in a configured bin. Only box2d objects are read (lanes and areas are polygons)."""
    split = cfg.data.split
    lab_dir = cfg.paths.data / "labels" / "100k" / split
    img_dir = cfg.paths.data / "images" / "100k" / split
    bin_of = {(w, t): b for b, spec in cfg.data.bins.items() for w in spec["weather"] for t in spec["timeofday"]}
    names = cfg.data.get("names_file")  # optional subset of the split (wad/thresh_net.py: BDD100K train)
    allow = {l.strip() for l in open(names, encoding="utf-8") if l.strip()} if names else None
    for p in sorted(lab_dir.glob("*.json")):
        if allow is not None and p.stem not in allow:
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        a = d.get("attributes", {})
        b = bin_of.get((a.get("weather"), a.get("timeofday")))
        if b is None:
            tally[f"excluded {a.get('weather')}/{a.get('timeofday')}"] += 1
            continue
        jpg = img_dir / f"{d['name']}.jpg"
        if not jpg.exists():
            raise FileNotFoundError(f"{jpg} missing; run scripts/get_bdd100k.py")
        objs = []
        for fr in d["frames"]:
            for o in fr["objects"]:
                if "box2d" not in o:
                    continue
                bx, at = o["box2d"], o.get("attributes", {})
                objs.append(dict(cls=cfg.data.class_map.get(o["category"], o["category"]),
                                 x1=float(bx["x1"]), y1=float(bx["y1"]), x2=float(bx["x2"]), y2=float(bx["y2"]),
                                 difficult=0, truncated=int(bool(at.get("truncated", False))),
                                 occluded=int(bool(at.get("occluded", False)))))
        yield f"{b}/{d['name']}", jpg, b, None, objs


def scan(cfg: Cfg) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    exif_policy = cfg.data.exif_orientation
    img_rows, box_rows = [], []
    dropped = Counter()
    tally = Counter()
    collect = collect_bdd if cfg.dataset == "bdd100k" else collect_dawn
    for image_id, jpg, weather, size, objs in collect(cfg, tally):
        if True:
            with Image.open(jpg) as im:
                real_w, real_h, mode = im.size[0], im.size[1], im.mode
            w, h = size if size is not None else (real_w, real_h)
            if (real_w, real_h) != (w, h):
                raise ValueError(f"{jpg}: XML size {(w, h)} != image size {(real_w, real_h)}")
            rel = jpg.relative_to(REPO).as_posix()
            kept = []
            for o in objs:
                o.setdefault("occluded", 0)
                if o["cls"] not in CANONICAL:
                    if o["cls"] in cfg.data.drop_classes:
                        dropped[o["cls"]] += 1
                        continue
                    raise ValueError(f"{jpg}: unexpected class {o['cls']!r}; add it to data.drop_classes")
                if o["x2"] <= o["x1"] or o["y2"] <= o["y1"]:
                    dropped["degenerate (zero area)"] += 1  # IoU is 0 for any prediction: an unmatchable box
                    continue
                kept.append(o)
            for o in kept:
                box_rows.append(dict(image_id=image_id, path=rel, weather=weather, width=w, height=h, **o))
            rgb = load_rgb(jpg, exif_policy)
            img_rows.append(
                dict(
                    image_id=image_id,
                    path=rel,
                    weather=weather,
                    width=w,
                    height=h,
                    n_gt=len(kept),
                    sha256=sha256_file(jpg),
                    phash=str(imagehash.phash(rgb)),
                    phash_mirror=str(imagehash.phash(ImageOps.mirror(rgb))),
                    mode=mode,
                    exif_orientation=exif_orientation(jpg),
                )
            )
    images = pd.DataFrame(img_rows).sort_values("image_id").reset_index(drop=True)
    gt = pd.DataFrame(box_rows)

    # Exact duplicates: keep the first copy (by image_id) and exclude the rest.
    images["keep"] = True
    images["duplicate_of"] = ""
    exact_groups = []
    exact_cross = []
    for sha, grp in images.groupby("sha256"):
        if len(grp) > 1:
            ids = sorted(grp.image_id)
            exact_groups.append(ids)
            if grp.weather.nunique() > 1:
                # identical pixels filed under two weathers: neither label can be trusted, drop all
                exact_cross.append(ids)
                images.loc[images.image_id.isin(ids), "keep"] = False
                images.loc[images.image_id.isin(ids), "duplicate_of"] = "cross-weather exact duplicate"
                continue
            mask = images.image_id.isin(ids[1:])
            images.loc[mask, "keep"] = False
            images.loc[mask, "duplicate_of"] = ids[0]

    # Near duplicates among the kept images: reported, and kept together when splitting.
    kept = images[images.keep]
    pairs = near_duplicate_pairs(kept, cfg.data.phash_max_hamming)
    comp = components(list(kept.image_id), [(a, b) for a, b, _ in pairs])
    images["group"] = images.image_id.map(comp).fillna(-1).astype(int)

    gt = gt[gt.image_id.isin(set(kept.image_id))].sort_values(["image_id", "x1", "y1", "x2", "y2", "cls"])
    gt = gt.reset_index(drop=True)
    gt.insert(0, "gt_id", np.arange(len(gt)))

    # Every hash pair is checked on pixels, and the full lists are written for the record.
    path_of = images.set_index("image_id").path
    dup_rows = [dict(kind="exact", a=g[0], b=b, hamming=0, rho=1.0,
                     cross_weather=g[0].split("/")[0] != b.split("/")[0]) for g in exact_groups for b in g[1:]]
    dup_rows += [dict(kind="near", a=a, b=b, hamming=d, rho=pixel_correlation(path_of[a], path_of[b], exif_policy),
                      cross_weather=a.split("/")[0] != b.split("/")[0]) for a, b, d in pairs]
    dups = pd.DataFrame(dup_rows, columns=["kind", "a", "b", "hamming", "rho", "cross_weather"])
    dups.to_csv(data_path(cfg, "duplicates.csv"), index=False, float_format="%.4f")
    near_rho = dups[dups.kind == "near"].rho

    report = dict(
        exif_policy=exif_policy,
        n_images_found=int(len(images)),
        n_images_kept=int(images.keep.sum()),
        images_per_bin_found=images.groupby("weather").size().to_dict(),
        images_per_bin_kept=images[images.keep].groupby("weather").size().to_dict(),
        exact_duplicate_groups=exact_groups,
        exact_cross_weather_dropped=exact_cross,
        near_duplicate_pairs=[dict(a=a, b=b, hamming=d) for a, b, d in pairs],
        n_near_duplicate_groups=int(sum(1 for c in Counter(comp.values()).values() if c > 1)),
        near_duplicate_pixel_check=dict(n_pairs=int(len(near_rho)), min_rho=float(near_rho.min()) if len(near_rho) else None,
                                        n_rho_ge_0_99=int((near_rho >= 0.99).sum())),
        dropped_boxes=dict(dropped),
        excluded_images=dict(tally),
        n_flagged_difficult=int(gt.difficult.sum()),
        n_flagged_truncated=int(gt.truncated.sum()),
        images_without_gt=sorted(images[images.keep & (images.n_gt == 0)].image_id),
        image_modes=images["mode"].value_counts().to_dict(),
        non_rgb_images=sorted(images[images["mode"] != "RGB"].image_id),
        exif_orientation_counts={str(k): int(v) for k, v in images.exif_orientation.value_counts().items()},
        n_distinct_sizes=int(images[["width", "height"]].drop_duplicates().shape[0]),
        width_range=[int(images.width.min()), int(images.width.max())],
        height_range=[int(images.height.min()), int(images.height.max())],
        boxes_per_bin_class=gt.groupby(["weather", "cls"]).size().unstack(fill_value=0).to_dict(orient="index"),
    )
    return images, gt, report


def smoke_subset(images: pd.DataFrame, gt: pd.DataFrame, cfg: Cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg.smoke.seed)
    picks = []
    for weather, grp in images[images.keep & (images.n_gt > 0)].groupby("weather"):
        picks += list(rng.choice(sorted(grp.image_id), size=cfg.smoke.per_bin, replace=False))
    images = images[images.image_id.isin(picks)].reset_index(drop=True)
    return images, gt[gt.image_id.isin(picks)].reset_index(drop=True)


def load_images(cfg: Cfg) -> pd.DataFrame:
    """Kept (non-duplicate) images, the only ones any downstream step sees.

    data.labels = 'clean' (run 2) also keeps one image per near-duplicate group (lowest id) and
    drops every group whose copies carry different weather labels."""
    images = pd.read_parquet(data_path(cfg, "images.parquet"))
    images = images[images.keep]
    if cfg.data.get("labels", "all") == "clean":
        mixed = images.groupby("group").weather.nunique()
        images = images[~images.group.isin(mixed[mixed > 1].index)]
        images = images.sort_values("image_id").groupby("group").head(1)
    return images.sort_values("image_id").reset_index(drop=True)


def load_gt(cfg: Cfg) -> pd.DataFrame:
    gt = pd.read_parquet(data_path(cfg, "gt.parquet"))
    if cfg.data.get("labels", "all") != "all":
        gt = gt[gt.image_id.isin(set(load_images(cfg).image_id))].reset_index(drop=True)
    return gt


def summary_table(images: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    kept = images[images.keep]
    t = gt.groupby(["weather", "cls"]).size().unstack(fill_value=0).reindex(columns=CANONICAL, fill_value=0)
    t.insert(0, "boxes", t.sum(axis=1))
    t.insert(0, "images", kept.groupby("weather").size())
    t.loc["total"] = t.sum()
    return t.astype(int)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true", help="20-image subset into cache/smoke")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, smoke=args.smoke)
    full_cfg = load_config(args.config)
    images, gt, report = scan(full_cfg)
    if args.smoke:
        images, gt = smoke_subset(images, gt, cfg)
    images.to_parquet(data_path(cfg, "images.parquet"), index=False)
    gt.to_parquet(data_path(cfg, "gt.parquet"), index=False)
    write_json(data_path(cfg, "data_report.json"), report)
    pd.set_option("display.width", 200)
    print(summary_table(images, gt).to_string())
    print(f"\nexact duplicate groups: {len(report['exact_duplicate_groups'])}  "
          f"near-duplicate pairs (phash <= {cfg.data.phash_max_hamming}): {len(report['near_duplicate_pairs'])}  "
          f"dropped boxes: {report['dropped_boxes']}  images without GT: {report['images_without_gt']}")
    print(f"wrote {data_path(cfg, 'images.parquet')} and {data_path(cfg, 'gt.parquet')}")


if __name__ == "__main__":
    main()
