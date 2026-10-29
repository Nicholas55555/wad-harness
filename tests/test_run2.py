"""Run 2 features: grouped matching, clean labels, the clear-calibrated policy, the tiers."""
import numpy as np
import pandas as pd
import pytest

from conftest import write_synthetic_cache
from wad import crc, match
from wad.config import REPO, load_config, tau_grid
from wad.curves import Cache
from wad.data import load_gt, load_images


def write_dataset(cache, images, gt, det):
    cache.mkdir(parents=True, exist_ok=True)
    im = pd.DataFrame(images)
    for k, v in dict(path="", width=100, height=100, keep=True).items():
        im[k] = im.get(k, v) if k in im else v
    im["n_gt"] = im.image_id.map(pd.DataFrame(gt).groupby("image_id").size()).fillna(0).astype(int)
    im.to_parquet(cache / "images.parquet", index=False)
    g = pd.DataFrame(gt)
    g = g.assign(gt_id=np.arange(len(g)), path="", width=100, height=100, difficult=0, truncated=0,
                 weather=g.image_id.str.split("/").str[0])
    g.to_parquet(cache / "gt.parquet", index=False)
    pd.DataFrame(det, columns=["image_id", "cls", "score", "x1", "y1", "x2", "y2"]).to_parquet(
        cache / "detections.parquet", index=False)


@pytest.fixture
def tmp_cfg(tmp_path):
    cfg = load_config()
    for k in ("cache", "data_cache", "detections"):
        cfg.paths[k] = tmp_path / "cache"
    cfg.paths["results"] = tmp_path / "results"
    return cfg


def test_grouped_matching_lets_a_truck_box_match_a_car(tmp_cfg):
    write_dataset(tmp_cfg.paths.cache,
                  images=[dict(image_id="fog/a", weather="fog", group=0)],
                  gt=[dict(image_id="fog/a", cls="car", x1=0, y1=0, x2=10, y2=10),
                      dict(image_id="fog/a", cls="person", x1=50, y1=50, x2=60, y2=70)],
                  det=[("fog/a", "truck", 0.9, 0, 0, 10, 10), ("fog/a", "car", 0.8, 50, 50, 60, 70)])
    mg, mp = match.match_all(tmp_cfg)
    assert mg.match_score.isna().all()  # class-aware: truck != car, car != person
    tmp_cfg["match_groups"] = {"vehicle": ["car", "truck", "bus"], "two_wheeler": ["bicycle", "motorcycle"],
                               "person": ["person"]}
    mg, mp = match.match_all(tmp_cfg)
    assert mg.set_index("cls").match_score.to_dict()["car"] == pytest.approx(0.9)
    assert np.isnan(mg.set_index("cls").match_score["person"])  # a car box never matches a person
    assert mp.cls.tolist() == ["truck", "car"]  # outputs keep the original classes


def test_clean_labels_keep_one_per_group_and_drop_cross_weather_groups(tmp_cfg):
    write_dataset(tmp_cfg.paths.cache,
                  images=[dict(image_id="fog/a", weather="fog", group=0), dict(image_id="fog/b", weather="fog", group=0),
                          dict(image_id="fog/c", weather="fog", group=1), dict(image_id="rain/c", weather="rain", group=1),
                          dict(image_id="rain/d", weather="rain", group=2)],
                  gt=[dict(image_id=i, cls="car", x1=0, y1=0, x2=10, y2=10) for i in ["fog/a", "fog/b", "fog/c", "rain/c", "rain/d"]],
                  det=[])
    assert len(load_images(tmp_cfg)) == 5
    tmp_cfg.data["labels"] = "clean"
    assert load_images(tmp_cfg).image_id.tolist() == ["fog/a", "rain/d"]
    assert set(load_gt(tmp_cfg).image_id) == {"fog/a", "rain/d"}


def two_bin_cache(cfg, rng):
    """'clear' images miss little; 'rain' images miss more; FPs dense at low scores."""
    images, gt, pred = [], [], []
    for b, n, p_miss in [("clear", 60, 0.05), ("rain", 60, 0.3)]:
        for i in range(n):
            iid = f"{b}/{i:03d}"
            images.append(dict(image_id=iid, weather=b))
            for k in range(3):
                s = np.nan if rng.random() < p_miss else rng.uniform(0.05 if b == "rain" else 0.3, 0.95)
                gt.append(dict(image_id=iid, cls="car", match_score=s))
            for s in rng.uniform(0.001, 0.4, size=8):
                pred.append(dict(image_id=iid, cls="car", score=s, tp=False))
    write_synthetic_cache(cfg.paths.cache, images, gt, pred)


def test_reference_policy_and_tiers(synthetic_cfg):
    cfg = synthetic_cfg
    cfg.data["bins"] = {"clear": "Clear", "rain": "Rain"}
    cfg["reference_bin"] = "clear"
    cfg["tiers"] = {"fppi_caps": [0.5, 2.0]}
    cfg.splits["n_repeats"] = 3
    two_bin_cache(cfg, np.random.default_rng(0))
    out = crc.run(cfg)
    f = out["fits"]
    ref = f[f.policy == "ref_calibrated"]
    assert set(ref.bin) == {"clear"} and len(ref) == 3 * len(cfg.alphas)
    ev = out["evals"]
    r = ev[(ev.policy == "ref_calibrated") & (ev.cls == "all")]
    # one threshold for every bin: the tau column is the same in clear and rain
    for (seed, alpha), g in r[r.bin != "all"].groupby(["seed", "alpha"]):
        assert g.tau.nunique() == 1
    # tiers: tau_c is the smallest grid tau whose calibration FPPI is <= cap, m from the formula
    t = out["tiers"]
    c = Cache(cfg)
    taus = tau_grid(cfg)
    fp = c.fp_grid(taus, "all")
    loss = c.loss_grid(taus, "all")
    for row in t.itertuples():
        cal = (out["splits"].query("seed == @row.seed").set_index("image_id").loc[c.image_ids].half == "cal").to_numpy()
        rows = cal & (c.weather == row.bin)
        cal_fppi = fp[rows].sum(0) / rows.sum()
        j = int(np.nonzero(cal_fppi <= row.fppi_cap)[0].min())
        assert row.tau_c == pytest.approx(taus[j])
        n = rows.sum()
        assert row.m_cert == pytest.approx((n * loss[rows, j].mean() + 1) / (n + 1))
    # a looser FPPI cap never gives a higher threshold
    piv = t.pivot_table(index=["seed", "bin"], columns="fppi_cap", values="tau_c")
    assert (piv[2.0] <= piv[0.5] + 1e-12).all()


def test_vectorised_near_duplicate_scan_matches_brute_force():
    from wad.data import hamming, near_duplicate_pairs

    rng = np.random.default_rng(3)
    h = [int(x) for x in rng.integers(0, 2**63, size=60, dtype=np.int64)]
    h[5] = h[2] ^ 0b1011  # planted pairs at distance 3 and 0, plus one with the top bit set
    h[9] = h[7]
    h[11] = (1 << 63) | 12345
    h[12] = h[11] ^ 1
    mirror = [int(x) for x in rng.integers(0, 2**63, size=60, dtype=np.int64)]
    mirror[30] = h[20]  # image 30 is a mirrored copy of image 20
    df = pd.DataFrame(dict(image_id=[f"x/{i:02d}" for i in range(60)], phash=[f"{v:016x}" for v in h],
                           phash_mirror=[f"{v:016x}" for v in mirror]))
    got = set(near_duplicate_pairs(df, 4))
    want = set()
    for i in range(60):
        for j in range(i + 1, 60):
            d = min(hamming(df.phash[i], df.phash[j]), hamming(df.phash[i], df.phash_mirror[j]))
            if d <= 4:
                want.add((df.image_id[i], df.image_id[j], d))
    assert got == want and len(want) >= 4


def test_collect_bdd_bins_classes_and_drops(tmp_path):
    import json
    from collections import Counter

    from PIL import Image

    from wad.data import collect_bdd

    cfg = load_config(REPO / "config_bdd.yaml")
    cfg.paths["data"] = tmp_path
    lab = tmp_path / "labels" / "100k" / "val"
    img = tmp_path / "images" / "100k" / "val"
    lab.mkdir(parents=True)
    img.mkdir(parents=True)

    def obj(cat, x1, y1, x2, y2):
        return {"category": cat, "box2d": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "attributes": {"occluded": True, "truncated": False}}

    frames = [{"objects": [obj("rider", 0, 0, 10, 20), obj("bike", 0, 10, 10, 30), obj("motor", 5, 5, 9, 9),
                           obj("traffic sign", 1, 1, 2, 2), {"category": "lane/single white", "poly2d": []},
                           obj("car", 3, 3, 3, 8)]}]
    for name, weather, tod in [("a", "rainy", "daytime"), ("b", "foggy", "daytime"), ("c", "clear", "dawn/dusk"),
                               ("d", "snowy", "night")]:
        (lab / f"{name}.json").write_text(json.dumps({"name": name, "frames": frames,
                                                      "attributes": {"weather": weather, "timeofday": tod}}))
        Image.new("RGB", (40, 30)).save(img / f"{name}.jpg")
    tally = Counter()
    got = list(collect_bdd(cfg, tally))
    assert [g[0] for g in got] == ["rain/a", "snow_night/d"]
    assert tally["excluded foggy/daytime"] == 1 and tally["excluded clear/dawn/dusk"] == 1
    classes = [o["cls"] for o in got[0][4]]
    assert classes == ["person", "bicycle", "motorcycle", "traffic sign", "car"]  # lanes skipped; drops happen in scan
    assert all(o["occluded"] == 1 for o in got[0][4])


def test_grouped_per_class_tp_counts_for_the_matched_ground_truth(tmp_cfg):
    write_dataset(tmp_cfg.paths.cache,
                  images=[dict(image_id="fog/a", weather="fog", group=0)],
                  gt=[dict(image_id="fog/a", cls="car", x1=0, y1=0, x2=10, y2=10)],
                  det=[("fog/a", "truck", 0.9, 0, 0, 10, 10)])
    tmp_cfg["match_groups"] = {"vehicle": ["car", "truck", "bus"], "two_wheeler": ["bicycle", "motorcycle"],
                               "person": ["person"]}
    match.match_all(tmp_cfg)
    c = Cache(tmp_cfg)
    assert c.pred.cls.tolist() == ["truck"] and c.pred.eval_cls.tolist() == ["car"]
    from wad.curves import average_precision, operating_curve

    m = c.pred.eval_cls.to_numpy() == "car"
    oc = operating_curve(c.pred.score.to_numpy()[m], c.pred.tp.to_numpy()[m], int(c.n_gt("car").sum()), c.n_img)
    assert average_precision(oc["precision"], oc["recall"]) == pytest.approx(1.0)
    m = c.pred.eval_cls.to_numpy() == "truck"
    assert m.sum() == 0  # the truck box no longer inflates truck TPs against zero truck ground truth
