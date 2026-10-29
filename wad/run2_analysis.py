"""Run 2 analyses (see wad/run2.py and PREREG_run2.md).

Per analysis (detector x loss x labels): matching, curves and the go/no-go table, CRC with the
tiers and (BDD100K) the clear-calibrated policy, the smallest certifiable alpha, and the
precision/recall tables that answer "does tuning matter more in hazardous than in clear weather":

  pr_table.csv        precision, recall, F1, FPPI per bin at tau = 0.25 / 0.5, at the F1-optimal
                      tau and at FPPI 0.1 / 1.0, plus max recall
  pr_vs_ref.csv       (BDD100K) bin minus clear-daytime differences at tau = 0.25 with bootstrap CIs
  gains_losses.csv    test-half change from tau = 0.25 to each calibrated policy: recall points
                      gained, precision points lost, FPPI added, false positives per recovered object

Across analyses: results/run2/cross_dataset.csv (BDD100K clear-calibrated threshold applied to
DAWN) and results/run2/dawn_variants.csv (one row per DAWN analysis and bin).
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from wad import crc, curves, extras, match
from wad.config import CANONICAL, REPO, Cfg, bins, results_path, tau_grid, write_json
from wad.curves import ALL, Cache, summarize

FF = "%.6g"
CLASSES = [ALL, "person", "car"]


# ---------------------------------------------------------------- precision / recall
def pr_table(v: Cfg, c: Cache) -> pd.DataFrame:
    taus = tau_grid(v)
    rows = []
    for cls in CLASSES:
        n_gt_img = c.n_gt(cls)
        tp_grid = c.matched_grid(taus, cls)
        fp_grid = c.fp_grid(taus, cls)
        for b in bins(v) + [ALL]:
            sel = np.ones(c.n_img, bool) if b == ALL else (c.weather == b)
            n_img, n_gt = int(sel.sum()), float(n_gt_img[sel].sum())
            if n_gt == 0:
                continue
            tp, fp = tp_grid[sel].sum(0), fp_grid[sel].sum(0)
            f1 = 2 * tp / np.maximum(2 * tp + fp + (n_gt - tp), 1)
            j = int(np.argmax(f1))

            def point(tau_name, tau, tp_, fp_):
                prec = tp_ / (tp_ + fp_) if tp_ + fp_ > 0 else np.nan
                rec = tp_ / n_gt
                return {f"{tau_name}_tau": tau, f"{tau_name}_recall": rec, f"{tau_name}_precision": prec,
                        f"{tau_name}_f1": 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0,
                        f"{tau_name}_fppi": fp_ / n_img}

            row = dict(bin=b, cls=cls, n_images=n_img, n_gt=int(n_gt), max_recall=tp[0] / n_gt)
            for t in v.default_thresholds:
                cnt = c.at_thresholds(np.full(c.n_img, float(t)), cls)
                tp_t = float((cnt["n_gt"] - cnt["n_missed"])[sel].sum())
                fp_t = float(cnt["n_fp"][sel].sum())
                row.update(point(f"t{t}", float(t), tp_t, fp_t))
            row.update(point("f1opt", float(taus[j]), float(tp[j]), float(fp[j])))
            for cap in v.fppi_budgets:
                ok = np.nonzero(fp / n_img <= cap)[0]
                k = int(ok.min()) if len(ok) else len(taus) - 1
                row.update(point(f"fppi{cap}", float(taus[k]), float(tp[k]), float(fp[k])))
            rows.append(row)
    return pd.DataFrame(rows)


def pr_vs_ref(v: Cfg, c: Cache, ref: str, tau: float = 0.25, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Each bin minus the reference bin at one threshold, with a percentile bootstrap that
    resamples images (near-duplicate groups) within each bin independently."""
    rng = np.random.default_rng(seed)
    groups = c.images.group.to_numpy()
    out = []
    for cls in CLASSES:
        cnt = c.at_thresholds(np.full(c.n_img, tau), cls)
        feats = np.c_[cnt["n_gt"], cnt["n_gt"] - cnt["n_missed"], cnt["n_fp"], np.ones(c.n_img)]

        def boot(b):
            rows = np.nonzero(c.weather == b)[0]
            g, inv = np.unique(groups[rows], return_inverse=True)
            per = np.zeros((len(g), 4))
            np.add.at(per, inv, feats[rows])
            s = per[rng.integers(0, len(g), size=(n_boot, len(g)))].sum(1)
            pt = per.sum(0)
            return pt, s

        def metrics(s):
            n_gt, tp, fp, n = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
            return dict(recall=tp / np.maximum(n_gt, 1), precision=tp / np.maximum(tp + fp, 1), fppi=fp / n)

        p_ref, s_ref = boot(ref)
        m_ref, mb_ref = metrics(p_ref), metrics(s_ref)
        for b in bins(v):
            p_b, s_b = boot(b)
            m_b, mb_b = metrics(p_b), metrics(s_b)
            row = dict(bin=b, cls=cls, tau=tau, n_images=int(p_b[3]), n_gt=int(p_b[0]))
            for k in ("recall", "precision", "fppi"):
                d = mb_b[k] - mb_ref[k]
                row.update({k: float(m_b[k]), f"{k}_ref": float(m_ref[k]), f"d_{k}": float(m_b[k] - m_ref[k]),
                            f"d_{k}_lo": float(np.percentile(d, 2.5)), f"d_{k}_hi": float(np.percentile(d, 97.5))})
            out.append(row)
    return pd.DataFrame(out)


def gains_losses(v: Cfg, evals: pd.DataFrame) -> pd.DataFrame:
    """Per split, test-half change from tau = 0.25 to each calibrated policy (all classes)."""
    e = evals[evals.cls == ALL].copy()
    e["recall"] = 1 - e.miss_rate
    e["tp"] = e.n_gt * e.recall
    e["fp"] = e.fppi * e.n_test
    e["precision"] = e.tp / (e.tp + e.fp)
    base = e[e.policy == "default_0.25"].set_index(["seed", "alpha", "bin"])
    out = []
    for pol in [p for p in e.policy.unique() if not p.startswith("default")]:
        x = e[e.policy == pol].set_index(["seed", "alpha", "bin"])
        d = x[["recall", "precision", "fppi", "tp", "fp", "image_loss"]] - base[["recall", "precision", "fppi", "tp", "fp", "image_loss"]]
        d["fp_per_recovered"] = d.fp / d.tp.where(d.tp > 0)
        d = d.reset_index().assign(policy=pol)
        out.append(d)
    d = pd.concat(out, ignore_index=True)

    def q(p):
        return lambda x: float(np.nanpercentile(x, p)) if np.isfinite(x).any() else np.nan

    g = d.groupby(["alpha", "policy", "bin"], sort=False)
    s = g.agg(n_seeds=("seed", "nunique"),
              d_recall_mean=("recall", "mean"), d_recall_lo=("recall", q(2.5)), d_recall_hi=("recall", q(97.5)),
              d_precision_mean=("precision", "mean"), d_precision_lo=("precision", q(2.5)),
              d_precision_hi=("precision", q(97.5)),
              d_fppi_mean=("fppi", "mean"), d_fppi_lo=("fppi", q(2.5)), d_fppi_hi=("fppi", q(97.5)),
              d_image_loss_mean=("image_loss", "mean"), sum_dfp=("fp", "sum"), sum_dtp=("tp", "sum"),
              n_seeds_recall_gain=("tp", lambda x: int((x > 0).sum()))).reset_index()
    # false positives added per object recovered: ratio of sums over splits, undefined if recall falls
    s["fp_per_recovered"] = np.where(s.sum_dtp > 0, s.sum_dfp / s.sum_dtp.where(s.sum_dtp > 0), np.nan)
    return s.drop(columns=["sum_dfp", "sum_dtp"])


def operating_point_boot(v: Cfg, c: Cache, ref: str | None, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Q2 with intervals: per bin, the tau giving FPPI 0.1 / 1.0 and the F1-optimal tau (all classes),
    with a percentile bootstrap over images (near-duplicate groups) within the bin, and each bin
    minus the reference bin."""
    rng = np.random.default_rng(seed)
    taus = tau_grid(v)
    tp, fp, ng = c.matched_grid(taus, ALL), c.fp_grid(taus, ALL), c.n_gt(ALL)
    groups = c.images.group.to_numpy()

    def stats(T, F, G, n):
        out = {}
        for cap in v.fppi_budgets:
            ok = F / n <= cap
            out[f"tau_fppi{cap}"] = taus[np.argmax(ok, axis=-1)] if ok.ndim > 1 else taus[int(np.argmax(ok))]
        f1 = 2 * T / np.maximum(2 * T + F + (G[..., None] if np.ndim(G) else G) - T, 1)
        out["tau_f1opt"] = taus[np.argmax(f1, axis=-1)]
        return out

    boots, points = {}, {}
    for b in bins(v):
        rows = np.nonzero(c.weather == b)[0]
        g, inv = np.unique(groups[rows], return_inverse=True)
        T = np.zeros((len(g), len(taus)))
        F = np.zeros_like(T)
        G = np.zeros(len(g))
        N = np.zeros(len(g))
        np.add.at(T, inv, tp[rows])
        np.add.at(F, inv, fp[rows])
        np.add.at(G, inv, ng[rows])
        np.add.at(N, inv, 1)
        points[b] = stats(T.sum(0), F.sum(0), G.sum(), N.sum())
        idx = rng.integers(0, len(g), size=(n_boot, len(g)))
        bs = {k: [] for k in points[b]}
        for r in range(n_boot):
            st = stats(T[idx[r]].sum(0), F[idx[r]].sum(0), G[idx[r]].sum(), N[idx[r]].sum())
            for k in bs:
                bs[k].append(st[k])
        boots[b] = {k: np.array(x) for k, x in bs.items()}
    out = []
    for b in bins(v):
        row = dict(bin=b)
        for k, x in boots[b].items():
            row.update({k: float(points[b][k]), f"{k}_lo": float(np.percentile(x, 2.5)), f"{k}_hi": float(np.percentile(x, 97.5))})
            if ref:
                d = x - boots[ref][k]
                row.update({f"d_{k}": float(points[b][k] - points[ref][k]), f"d_{k}_lo": float(np.percentile(d, 2.5)),
                            f"d_{k}_hi": float(np.percentile(d, 97.5))})
        out.append(row)
    return pd.DataFrame(out)


HEIGHT_EDGES = [0, 10, 20, 40, 80, np.inf]  # ground-truth box height in original pixels


def pr_vs_ref_sizeadj(v: Cfg, c: Cache, ref: str, tau: float = 0.25, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """EXPLORATORY (added after review; not pre-registered). Recall at tau standardised to the
    reference bin's ground-truth height mix: each bin's per-height-band recall reweighted by the
    reference bin's share of boxes per band. Removes the object-size difference between bins."""
    from wad.data import load_gt

    rng = np.random.default_rng(seed)
    gt = load_gt(v).set_index("gt_id")
    hgt = (gt.y2 - gt.y1).reindex(c.gt.gt_id).to_numpy()
    band = np.digitize(hgt, HEIGHT_EDGES[1:-1])
    nb = len(HEIGHT_EDGES) - 1
    hit = c.gt_score >= tau
    groups = c.images.group.to_numpy()

    def per_group(b):
        rows = np.nonzero(c.weather == b)[0]
        g, inv = np.unique(groups[rows], return_inverse=True)
        pos = {r: i for i, r in enumerate(rows)}
        sel = np.isin(c.gt_img, rows)
        gi = np.array([inv[pos[r]] for r in c.gt_img[sel]])
        n = np.zeros((len(g), nb))
        h = np.zeros((len(g), nb))
        np.add.at(n, (gi, band[sel]), 1)
        np.add.at(h, (gi, band[sel]), hit[sel])
        return n, h

    def std_recall(n, h, w):
        r = h.sum(0) / np.maximum(n.sum(0), 1)
        return float((w * r).sum())

    n_ref, h_ref = per_group(ref)
    w = n_ref.sum(0) / n_ref.sum()
    idx_ref = rng.integers(0, len(n_ref), size=(n_boot, len(n_ref)))
    ref_boot = np.array([std_recall(n_ref[i], h_ref[i], w) for i in idx_ref])
    ref_pt = std_recall(n_ref, h_ref, w)
    out = []
    for b in bins(v):
        n, h = per_group(b)
        idx = rng.integers(0, len(n), size=(n_boot, len(n)))
        bt = np.array([std_recall(n[i], h[i], w) for i in idx])
        raw = float(h.sum() / max(n.sum(), 1))
        pt = std_recall(n, h, w)
        d = bt - ref_boot
        out.append(dict(bin=b, tau=tau, recall_raw=raw, recall_std=pt, recall_ref=ref_pt, d_recall_std=pt - ref_pt,
                        d_recall_std_lo=float(np.percentile(d, 2.5)), d_recall_std_hi=float(np.percentile(d, 97.5)),
                        **{f"share_band{k}": float(n.sum(0)[k] / max(n.sum(), 1)) for k in range(nb)},
                        **{f"recall_band{k}": float(h.sum(0)[k] / max(n.sum(0)[k], 1)) for k in range(nb)}))
    return pd.DataFrame(out)


def split_intervals(evals: pd.DataFrame, fits: pd.DataFrame) -> pd.DataFrame:
    """Q3 with intervals over the split repeats: per policy and bin, the held-out miss fraction and
    FPPI (mean, 2.5/97.5 percentiles), share of splits above alpha, and the share of fits that were
    infeasible; plus clear-calibrated minus per-bin differences."""
    e = evals[evals.cls == ALL]
    rows = []
    for (a, pol, b), g in e.groupby(["alpha", "policy", "bin"], sort=False):
        rows.append(dict(alpha=a, policy=pol, bin=b, n_seeds=g.seed.nunique(),
                         loss_mean=g.image_loss.mean(), loss_lo=g.image_loss.quantile(0.025),
                         loss_hi=g.image_loss.quantile(0.975), fppi_mean=g.fppi.mean(),
                         fppi_lo=g.fppi.quantile(0.025), fppi_hi=g.fppi.quantile(0.975),
                         exceed_frac=float((g.image_loss > a).mean())))
    s = pd.DataFrame(rows)
    inf = fits.groupby(["alpha", "policy", "bin"]).infeasible.mean().rename("infeasible_frac").reset_index()
    s = s.merge(inf, on=["alpha", "policy", "bin"], how="left")
    for pol in ["global", "ref_calibrated"]:  # one fit applies to every bin
        fi = fits[fits.policy == pol].groupby("alpha").infeasible.mean()
        m = s.policy == pol
        s.loc[m, "infeasible_frac"] = s.loc[m, "alpha"].map(fi)
    if "ref_calibrated" in set(e.policy):
        x = e[e.policy == "ref_calibrated"].set_index(["seed", "alpha", "bin"])
        o = e[e.policy == "oracle_bin"].set_index(["seed", "alpha", "bin"])
        d = (x[["image_loss", "fppi"]] - o[["image_loss", "fppi"]]).reset_index()
        dd = d.groupby(["alpha", "bin"]).agg(d_loss_mean=("image_loss", "mean"), d_loss_lo=("image_loss", lambda z: z.quantile(0.025)),
                                             d_loss_hi=("image_loss", lambda z: z.quantile(0.975)),
                                             d_fppi_mean=("fppi", "mean"), d_fppi_lo=("fppi", lambda z: z.quantile(0.025)),
                                             d_fppi_hi=("fppi", lambda z: z.quantile(0.975))).reset_index()
        s = s.merge(dd.assign(policy="ref_calibrated"), on=["alpha", "policy", "bin"], how="left")
    return s


def gpu_cpu_check(n: int = 12) -> dict:
    """Do this torch build's CPU detections reproduce run 1's cached CPU detections bit for bit?"""
    import copy

    import torch

    from wad.config import load_config
    from wad.data import load_images
    from wad.detect import load_model, predict_one

    cfg = load_config()
    m = load_model(cfg)
    det = pd.read_parquet(REPO / "cache" / "detections.parquet")
    same = 0
    ids = list(load_images(cfg).head(n).itertuples())
    for r in ids:
        d, _ = predict_one(m, copy.deepcopy(cfg), r.image_id, r.path, "cpu")
        ref = det[det.image_id == r.image_id]
        key = ["score", "x1", "y1", "x2", "y2"]
        a = d.sort_values(key, ascending=False, kind="stable")[key].to_numpy()
        b = ref.sort_values(key, ascending=False, kind="stable")[key].to_numpy()
        same += int(a.shape == b.shape and np.array_equal(a, b))
    out = dict(torch=torch.__version__, cuda=torch.version.cuda, n_images=len(ids), bit_identical=same)
    (REPO / "results" / "run2").mkdir(parents=True, exist_ok=True)
    write_json(REPO / "results" / "run2" / "gpu_cpu_check.json", out)
    return out


# ---------------------------------------------------------------- one analysis
def analyse(v: Cfg) -> dict:
    print(f"== analyse {v.dataset_name} {v.tag}", flush=True)
    match.match_all(v)
    cur, summ, verdict = curves.compute(v)
    cur.to_csv(results_path(v, "curves.csv"), index=False, float_format=FF)
    summ.to_csv(results_path(v, "summary_phase3.csv"), index=False, float_format=FF)
    write_json(results_path(v, "go_no_go.json"), verdict)
    out = crc.run(v)
    crc.write(v, out)
    extras.min_feasible_alpha(v).to_csv(results_path(v, "min_feasible_alpha.csv"), index=False, float_format=FF)
    c = Cache(v)
    pr_table(v, c).to_csv(results_path(v, "pr_table.csv"), index=False, float_format=FF)
    gains_losses(v, out["evals"]).to_csv(results_path(v, "gains_losses.csv"), index=False, float_format=FF)
    ref = v.get("reference_bin")
    if ref:
        pr_vs_ref(v, c, ref).to_csv(results_path(v, "pr_vs_ref.csv"), index=False, float_format=FF)
        pr_vs_ref_sizeadj(v, c, ref).to_csv(results_path(v, "pr_vs_ref_sizeadj_EXPLORATORY.csv"), index=False,
                                             float_format=FF)
    operating_point_boot(v, c, ref).to_csv(results_path(v, "operating_point_boot.csv"), index=False, float_format=FF)
    # pre-registered robustness: the same procedure over 200 split repeats
    v200 = copy.deepcopy(v)
    v200.splits["n_repeats"] = 200
    o200 = crc.run(v200)
    o200["summary"].to_csv(results_path(v, "seeds200/crc_summary.csv"), index=False, float_format=FF)
    if "tiers_summary" in o200:
        o200["tiers_summary"].to_csv(results_path(v, "seeds200/tiers_summary.csv"), index=False, float_format=FF)
    split_intervals(o200["evals"], o200["fits"]).to_csv(results_path(v, "seeds200/policy_intervals.csv"), index=False,
                                                         float_format=FF)
    gains_losses(v, o200["evals"]).to_csv(results_path(v, "seeds200/gains_losses.csv"), index=False, float_format=FF)
    if v.dataset_name == "dawn":
        df, s = extras.go_no_go_bootstrap(v)
        df.to_csv(results_path(v, "go_no_go_bootstrap.csv"), index=False, float_format=FF)
        write_json(results_path(v, "go_no_go_bootstrap.json"), s)
    extras.misses_by_size(v).to_csv(results_path(v, "misses_by_size.csv"), index=False, float_format=FF)
    # EXPLORATORY (not pre-registered): every policy over a grid of budgets, so the policies can be
    # compared where the pre-registered alphas are infeasible
    extras.alpha_sweep(v).to_csv(results_path(v, "alpha_sweep_EXPLORATORY.csv"), index=False, float_format=FF)
    print("   " + verdict["line"], flush=True)
    return dict(verdict=verdict, summary=summ, crc=out)


# ---------------------------------------------------------------- cross-dataset and roll-ups
def cross_dataset(r2: Cfg) -> pd.DataFrame:
    """The BDD100K clear-daytime threshold (ref_calibrated, per split and alpha) applied to every
    DAWN image with the same detector and loss. Descriptive: the datasets differ in more than
    weather."""
    from wad.run2 import bdd_cfg, dawn_cfg

    rows = []
    for det in r2.bdd.detectors:
        for loss in r2.bdd.losses:
            vb = bdd_cfg(r2, det, loss)
            fits_p = results_path(vb, "crc_fits.csv")
            if not fits_p.exists():
                continue
            fits = pd.read_csv(fits_p)
            fits = fits[fits.policy == "ref_calibrated"]
            vd = dawn_cfg(r2, det, loss, "clean")
            if not (vd.paths.cache / "matches_gt.parquet").exists():
                continue
            c = Cache(vd)
            for f in fits.itertuples():
                for cls in [ALL]:
                    cnt = c.at_thresholds(np.full(c.n_img, f.tau_hat), cls)
                    for b in bins(vd) + [ALL]:
                        r = np.arange(c.n_img) if b == ALL else np.nonzero(c.weather == b)[0]
                        s = summarize(cnt, r, per_class=False)
                        tp = s["n_gt"] * (1 - s["miss_rate"])
                        fp = s["fppi"] * s["n_images"]
                        rows.append(dict(detector=det, loss=loss, seed=f.seed, alpha=f.alpha, tau_bdd_clear=f.tau_hat,
                                         bdd_fit_infeasible=bool(f.infeasible),
                                         dawn_bin=b, image_loss=s["image_loss"], recall=1 - s["miss_rate"],
                                         precision=tp / (tp + fp) if tp + fp > 0 else np.nan, fppi=s["fppi"],
                                         n_images=s["n_images"]))
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    s = d.groupby(["detector", "loss", "alpha", "dawn_bin"], sort=False).agg(
        n_seeds=("seed", "nunique"), tau_bdd_clear_mean=("tau_bdd_clear", "mean"),
        tau_bdd_clear_median=("tau_bdd_clear", "median"), bdd_fit_infeasible_frac=("bdd_fit_infeasible", "mean"),
        image_loss_mean=("image_loss", "mean"), recall_mean=("recall", "mean"),
        precision_mean=("precision", "mean"), fppi_mean=("fppi", "mean"), n_images=("n_images", "first")).reset_index()
    # image-level bootstrap of the DAWN miss fraction at the median BDD tau (the DAWN images are
    # fixed, so the spread over BDD splits alone says nothing about DAWN sampling error)
    rng = np.random.default_rng(0)
    lo, hi, p_exc = [], [], []
    caches = {}
    for r in s.itertuples():
        key = (r.detector, r.loss)
        if key not in caches:
            caches[key] = Cache(dawn_cfg(r2, r.detector, r.loss, "clean"))
        c = caches[key]
        cnt = c.at_thresholds(np.full(c.n_img, r.tau_bdd_clear_median), ALL)
        rows = np.arange(c.n_img) if r.dawn_bin == ALL else np.nonzero(c.weather == r.dawn_bin)[0]
        li = np.where(cnt["n_gt"][rows] > 0, cnt["n_missed"][rows] / np.maximum(cnt["n_gt"][rows], 1), 0.0)
        bs = li[rng.integers(0, len(li), size=(2000, len(li)))].mean(1)
        lo.append(float(np.percentile(bs, 2.5)))
        hi.append(float(np.percentile(bs, 97.5)))
        p_exc.append(float((bs > r.alpha).mean()))
    s["image_loss_at_median_lo"], s["image_loss_at_median_hi"], s["p_loss_above_alpha"] = lo, hi, p_exc
    return s


def dawn_variants(r2: Cfg) -> pd.DataFrame:
    from wad.run2 import dawn_cfg

    rows = []
    for det in r2.detectors:
        for loss in r2.losses:
            for labels in r2.labels:
                v = dawn_cfg(r2, det, loss, labels)
                p = results_path(v, "crc_summary.csv")
                if not p.exists():
                    continue
                s3 = pd.read_csv(results_path(v, "summary_phase3.csv"))
                cs = pd.read_csv(p)
                ma = pd.read_csv(results_path(v, "min_feasible_alpha.csv")).set_index("bin")
                gv = pd.read_json(results_path(v, "go_no_go.json"), typ="series")
                for b in bins(v) + [ALL]:
                    x = s3[(s3.bin == b) & (s3.cls == ALL)].iloc[0]
                    row = dict(detector=det, loss=loss, labels=labels, bin=b, n_images=int(x.n_images),
                               n_gt=int(x.n_gt), max_recall=x.max_recall, recall_at_025=x["recall_at_0.25"],
                               lamr=x.lamr, map50=x.map50, min_alpha=ma.loc[b, "min_alpha_mean"],
                               verdict=gv["verdict"])
                    for a in v.alphas:
                        o = cs[(cs.alpha == a) & (cs.policy == "oracle_bin") & (cs.cls == ALL) & (cs.bin == b)].iloc[0]
                        g = cs[(cs.alpha == a) & (cs.policy == "global") & (cs.cls == ALL) & (cs.bin == b)].iloc[0]
                        row.update({f"oracle_loss_a{a}": o.image_loss_mean, f"oracle_fppi_a{a}": o.fppi_mean,
                                    f"oracle_infeasible_a{a}": o.infeasible_frac, f"oracle_exceed_a{a}": o.exceed_frac,
                                    f"global_loss_a{a}": g.image_loss_mean, f"global_fppi_a{a}": g.fppi_mean,
                                    f"global_exceed_a{a}": g.exceed_frac})
                    rows.append(row)
    return pd.DataFrame(rows)


def analyse_all(r2: Cfg, only: str | None = None) -> None:
    from wad.run2 import bdd_cfg, dawn_cfg

    jobs = []
    for det, d in r2.detectors.items():
        if not (REPO / d["detections"] / "detections.parquet").exists():
            print(f"skip dawn {det}: no detections yet")
            continue
        jobs += [dawn_cfg(r2, det, loss, labels) for loss in r2.losses for labels in r2.labels]
    for det in r2.bdd.detectors:
        if not (REPO / "cache" / "bdd" / "det" / det / "detections.parquet").exists():
            print(f"skip bdd {det}: no detections yet")
            continue
        jobs += [bdd_cfg(r2, det, loss) for loss in r2.bdd.losses]
    for v in jobs:
        if only and only not in (v.dataset_name, v.det_id, v.tag):
            continue
        analyse(v)
    out = REPO / "results" / "run2"
    out.mkdir(parents=True, exist_ok=True)
    dv = dawn_variants(r2)
    if not dv.empty:
        dv.to_csv(out / "dawn_variants.csv", index=False, float_format=FF)
    cd = cross_dataset(r2)
    if not cd.empty:
        cd.to_csv(out / "cross_dataset.csv", index=False, float_format=FF)
