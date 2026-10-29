"""Conformal risk control per weather bin: splits, calibration, evaluation (phase 4).

Loss: L_i(tau) = fraction of image i's ground-truth boxes with no matched detection at score
>= tau (0 for images without ground truth). Bounded by B = 1 and non-decreasing in tau.

Rule, on n calibration images with mean loss R(tau):
    tau_hat = max { tau in grid : n/(n+1) * R(tau) + 1/(n+1) <= alpha }
If no grid point qualifies, tau_hat = conf floor and the fit is flagged infeasible: the
guarantee then needs the loss to vanish at the floor, which it does not when the detector
never proposes some objects.

Splits are by image, stratified by weather, and keep each near-duplicate group (wad.data) on
one side, so calibration and test never share an image or a re-encoded copy of one.

Run: python -m wad.crc [--smoke] [--variant near_dup_dropped]
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pandas as pd

from wad.config import CANONICAL, Cfg, bins, load_config, results_path, tau_grid, write_json
from wad.curves import ALL, Cache, summarize

POLICIES = ["oracle_bin", "global"]


# ---------------------------------------------------------------- calibration rule
def crc_bound(risk: np.ndarray, n: int) -> np.ndarray:
    return n / (n + 1) * risk + 1 / (n + 1)


def fit_threshold(losses: np.ndarray, taus: np.ndarray, alpha: float, floor: float) -> dict:
    """Largest grid threshold whose conformal bound on the calibration losses is <= alpha.

    losses: (n_images, n_taus) per-image losses on the grid, each row non-decreasing.
    """
    n = losses.shape[0]
    if n == 0:
        return dict(tau_hat=floor, infeasible=True, n_cal=0, cal_risk=np.nan, cal_bound=np.nan)
    risk = losses.mean(axis=0)
    bound = crc_bound(risk, n)
    # n/(n+1)*R + 1/(n+1) <= alpha  <=>  sum(L) + 1 <= alpha*(n+1); the sum form plus a 1e-9
    # tolerance keeps exact ties at alpha feasible despite floating-point rounding.
    ok = np.nonzero(losses.sum(axis=0) + 1 <= alpha * (n + 1) + 1e-9)[0]
    if len(ok) == 0:
        j = int(np.argmin(np.abs(taus - floor)))
        return dict(tau_hat=float(floor), infeasible=True, n_cal=n, cal_risk=float(risk[j]), cal_bound=float(bound[j]))
    j = int(ok.max())
    return dict(tau_hat=float(taus[j]), infeasible=False, n_cal=n, cal_risk=float(risk[j]), cal_bound=float(bound[j]))


# ---------------------------------------------------------------- splits
def split_once(images: pd.DataFrame, frac: float, seed: int) -> np.ndarray:
    """Boolean calibration mask. Stratified by weather; near-duplicate groups stay together.

    Groups are visited in a random order and each goes to calibration while its weather bins
    still have room (target = floor(n_bin * frac)), otherwise to test.
    """
    rng = np.random.default_rng(seed)
    target = {b: int(np.floor(n * frac)) for b, n in images.weather.value_counts().items()}
    members = images.groupby("group").weather.apply(list)
    have = Counter()
    cal_groups = set()
    for gid in rng.permutation(np.array(sorted(members.index))):
        need = Counter(members[gid])
        if all(have[b] + k <= target[b] for b, k in need.items()):
            cal_groups.add(gid)
            have.update(need)
    return images.group.isin(cal_groups).to_numpy()


def make_splits(cfg: Cfg, images: pd.DataFrame) -> pd.DataFrame:
    s = cfg.splits
    out = []
    for r in range(s.n_repeats):
        seed = s.seed + r
        cal = split_once(images, s.calibration_fraction, seed)
        out.append(pd.DataFrame(dict(seed=seed, image_id=images.image_id, weather=images.weather,
                                     group=images.group, half=np.where(cal, "cal", "test"))))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- variants
def restrict(c: Cache, variant: str) -> Cache:
    """Primary analysis keeps every non-exact-duplicate image. The sensitivity variant keeps one
    image per near-duplicate group and drops groups whose copies carry different weather labels."""
    if variant == "primary":
        return c
    if variant != "near_dup_dropped":
        raise ValueError(variant)
    im = c.images
    mixed = im.groupby("group").weather.nunique()
    keep = im[~im.group.isin(mixed[mixed > 1].index)].sort_values("image_id").groupby("group").head(1)
    keep_ids = set(keep.image_id)
    c2 = Cache.__new__(Cache)
    c2.__dict__.update(c.__dict__)
    c2.images = im[im.image_id.isin(keep_ids)].reset_index(drop=True)
    c2.image_ids = c2.images.image_id.to_numpy()
    c2.idx = {k: i for i, k in enumerate(c2.image_ids)}
    c2.weather = c2.images.weather.to_numpy()
    c2.n_img = len(c2.image_ids)
    gm = c.gt.image_id.isin(keep_ids).to_numpy()
    c2.gt = c.gt[gm].reset_index(drop=True)
    c2.gt_img = c2.gt.image_id.map(c2.idx).to_numpy()
    c2.gt_cls, c2.gt_score = c.gt_cls[gm], c.gt_score[gm]
    fm = np.isin(c.image_ids[c.fp_img], list(keep_ids))
    c2.fp_img = np.array([c2.idx[i] for i in c.image_ids[c.fp_img[fm]]], dtype=int)
    c2.fp_cls, c2.fp_score = c.fp_cls[fm], c.fp_score[fm]
    c2.pred = c.pred[c.pred.image_id.isin(keep_ids)].reset_index(drop=True)
    return c2


# ---------------------------------------------------------------- phase 4 driver
def run(cfg: Cfg, variant: str = "primary") -> dict[str, pd.DataFrame]:
    c = restrict(Cache(cfg), variant)
    taus = tau_grid(cfg)
    floor = cfg.detector.conf_floor
    splits = make_splits(cfg, c.images)
    loss_all = c.loss_grid(taus, ALL)
    loss_cls = {k: c.loss_grid(taus, k) for k in CANONICAL}
    has_cls = {k: c.n_gt(k) > 0 for k in CANONICAL}
    bin_names = bins(cfg)
    fits, evals, pc_fits, pc_evals = [], [], [], []

    for seed, sp in splits.groupby("seed", sort=True):
        sp = sp.set_index("image_id").loc[c.image_ids]
        cal = (sp.half == "cal").to_numpy()
        test = ~cal
        for alpha in cfg.alphas:
            tau_img = {}
            # oracle-bin policy: one threshold per true weather bin
            t = np.full(c.n_img, np.nan)
            for b in bin_names:
                rows = cal & (c.weather == b)
                f = fit_threshold(loss_all[rows], taus, alpha, floor)
                fits.append(dict(seed=seed, alpha=alpha, policy="oracle_bin", bin=b, **f))
                t[c.weather == b] = f["tau_hat"]
            tau_img["oracle_bin"] = t
            # global policy: one threshold on every calibration image pooled
            f = fit_threshold(loss_all[cal], taus, alpha, floor)
            fits.append(dict(seed=seed, alpha=alpha, policy="global", bin=ALL, **f))
            tau_img["global"] = np.full(c.n_img, f["tau_hat"])
            # reference-bin policy (run 2, BDD100K): calibrated on the clear-weather bin only,
            # applied to every bin -- what a threshold tuned in good weather does elsewhere
            ref = cfg.get("reference_bin")
            if ref:
                f = fit_threshold(loss_all[cal & (c.weather == ref)], taus, alpha, floor)
                fits.append(dict(seed=seed, alpha=alpha, policy="ref_calibrated", bin=ref, **f))
                tau_img["ref_calibrated"] = np.full(c.n_img, f["tau_hat"])
            for d in cfg.default_thresholds:
                tau_img[f"default_{d}"] = np.full(c.n_img, float(d))
            # evaluate every policy on the test half, per bin and per class
            for policy, ti in tau_img.items():
                for cls in [ALL] + CANONICAL:
                    counts = c.at_thresholds(ti, cls)
                    for b in bin_names + [ALL]:
                        rows = np.nonzero(test & ((c.weather == b) if b != ALL else True))[0]
                        s = summarize(counts, rows, per_class=cls != ALL)
                        evals.append(dict(seed=seed, alpha=alpha, policy=policy, bin=b, cls=cls,
                                          image_loss=s["image_loss"], miss_rate=s["miss_rate"], fppi=s["fppi"],
                                          n_test=s["n_images"], n_loss_images=s["n_loss_images"], n_gt=s["n_gt"],
                                          tau=float(np.unique(ti[rows])[0]) if len(np.unique(ti[rows])) == 1 else np.nan))

        # per-class budgets (O4 preview): oracle bin x class, loss over images containing the class
        t_pc = {k: np.full(c.n_img, np.nan) for k in CANONICAL}
        for k in CANONICAL:
            a = cfg.perclass_alphas.get(k, cfg.perclass_alphas["default"])
            for b in bin_names:
                rows = cal & (c.weather == b) & has_cls[k]
                f = fit_threshold(loss_cls[k][rows], taus, a, floor)
                pc_fits.append(dict(seed=seed, alpha=a, bin=b, cls=k, **f))
                t_pc[k][c.weather == b] = f["tau_hat"]
            counts = c.at_thresholds(t_pc[k], k)
            for b in bin_names + [ALL]:
                rows = np.nonzero(test & ((c.weather == b) if b != ALL else True))[0]
                s = summarize(counts, rows, per_class=True)
                pc_evals.append(dict(seed=seed, alpha=a, bin=b, cls=k, image_loss=s["image_loss"],
                                     miss_rate=s["miss_rate"], fppi=s["fppi"], n_test=s["n_images"],
                                     n_loss_images=s["n_loss_images"], n_gt=s["n_gt"]))

    fits, evals = pd.DataFrame(fits), pd.DataFrame(evals)
    pc_fits, pc_evals = pd.DataFrame(pc_fits), pd.DataFrame(pc_evals)
    out = dict(splits=splits, fits=fits, evals=evals, summary=summarize_seeds(cfg, fits, evals),
               perclass=summarize_perclass(pc_fits, pc_evals), perclass_fits=pc_fits, perclass_evals=pc_evals)
    if cfg.get("tiers"):
        out["tiers"] = tiers(cfg, c, splits, taus, loss_all)
        out["tiers_summary"] = summarize_tiers(out["tiers"])
    return out


def tiers(cfg: Cfg, c: Cache, splits: pd.DataFrame, taus: np.ndarray, loss_all: np.ndarray) -> pd.DataFrame:
    """Confirmed tier (proposal slide 5): per bin, tau_c = smallest grid threshold whose
    calibration FPPI is <= cap, and the certified miss rate at it, m = (n R(tau_c) + 1) / (n + 1)
    on the calibration half. tau_c is chosen from calibration false positives, so m is a
    calibration-based bound; the held-out loss at tau_c is reported against it."""
    fp = c.fp_grid(taus, ALL)
    out = []
    for seed, sp in splits.groupby("seed", sort=True):
        cal = (sp.set_index("image_id").loc[c.image_ids].half == "cal").to_numpy()
        test = ~cal
        for b in bins(cfg):
            rows = cal & (c.weather == b)
            n = int(rows.sum())
            if n == 0:
                continue
            cal_fppi = fp[rows].sum(axis=0) / n  # non-increasing in tau
            risk = loss_all[rows].mean(axis=0)
            trows = np.nonzero(test & (c.weather == b))[0]
            for cap in cfg.tiers["fppi_caps"]:
                ok = np.nonzero(cal_fppi <= cap)[0]
                j = int(ok.min()) if len(ok) else len(taus) - 1
                s = summarize(c.at_thresholds(np.full(c.n_img, taus[j]), ALL), trows, per_class=False)
                m = (n * risk[j] + 1) / (n + 1)
                out.append(dict(seed=seed, fppi_cap=cap, bin=b, tau_c=float(taus[j]), n_cal=n,
                                cal_fppi=float(cal_fppi[j]), cal_risk=float(risk[j]), m_cert=float(m),
                                test_image_loss=s["image_loss"], test_miss_rate=s["miss_rate"],
                                test_fppi=s["fppi"], n_test=s["n_images"],
                                exceeds_m=bool(s["image_loss"] > m)))
    return pd.DataFrame(out)


def summarize_tiers(t: pd.DataFrame) -> pd.DataFrame:
    return t.groupby(["fppi_cap", "bin"], sort=False).agg(
        n_seeds=("seed", "nunique"), tau_c_mean=("tau_c", "mean"), tau_c_std=("tau_c", "std"),
        m_cert_mean=("m_cert", "mean"), m_cert_std=("m_cert", "std"),
        test_image_loss_mean=("test_image_loss", "mean"), test_image_loss_std=("test_image_loss", "std"),
        test_fppi_mean=("test_fppi", "mean"), test_fppi_std=("test_fppi", "std"),
        exceed_m_frac=("exceeds_m", "mean"), n_cal_mean=("n_cal", "mean"), n_test_mean=("n_test", "mean")).reset_index()


def summarize_seeds(cfg: Cfg, fits: pd.DataFrame, evals: pd.DataFrame) -> pd.DataFrame:
    keys = ["alpha", "policy", "bin", "cls"]
    e = evals.copy()
    e["exceeds"] = (e.image_loss > e.alpha).astype(float).where(e.image_loss.notna())
    e["excess"] = (e.image_loss - e.alpha).clip(lower=0)
    g = e.groupby(keys, sort=False)
    s = g.agg(n_seeds=("image_loss", "count"),
              image_loss_mean=("image_loss", "mean"), image_loss_std=("image_loss", "std"),
              miss_rate_mean=("miss_rate", "mean"), miss_rate_std=("miss_rate", "std"),
              fppi_mean=("fppi", "mean"), fppi_std=("fppi", "std"),
              exceed_frac=("exceeds", "mean"), max_excess=("excess", "max"),
              n_test_mean=("n_test", "mean"), n_loss_images_mean=("n_loss_images", "mean"),
              n_gt_mean=("n_gt", "mean"), tau_mean=("tau", "mean"), tau_std=("tau", "std")).reset_index()
    # threshold and feasibility from the fits (oracle per bin; global applies to every bin)
    f = fits.copy()
    fo = f[f.policy == "oracle_bin"].groupby(["alpha", "bin"]).agg(
        tau_hat_mean=("tau_hat", "mean"), tau_hat_std=("tau_hat", "std"),
        infeasible_frac=("infeasible", "mean"), n_cal_mean=("n_cal", "mean")).reset_index().assign(policy="oracle_bin")
    fo_all = f[f.policy == "oracle_bin"].groupby(["alpha"]).agg(
        infeasible_frac=("infeasible", "mean"), n_cal_mean=("n_cal", "sum")).reset_index()
    fo_all = fo_all.assign(bin=ALL, policy="oracle_bin", n_cal_mean=fo_all.n_cal_mean / f.seed.nunique(),
                           tau_hat_mean=np.nan, tau_hat_std=np.nan)
    fo = pd.concat([fo, fo_all], ignore_index=True)
    s = s.merge(fo, on=["alpha", "policy", "bin"], how="left")
    for pol in ["global", "ref_calibrated"]:
        fg = f[f.policy == pol].groupby(["alpha"]).agg(
            tau_hat_mean=("tau_hat", "mean"), tau_hat_std=("tau_hat", "std"),
            infeasible_frac=("infeasible", "mean"), n_cal_mean=("n_cal", "mean"))
        if fg.empty:
            continue
        for col in ["tau_hat_mean", "tau_hat_std", "infeasible_frac", "n_cal_mean"]:
            m = s.policy == pol
            s.loc[m, col] = s.loc[m, "alpha"].map(fg[col])
    for d in cfg.default_thresholds:
        m = s.policy == f"default_{d}"
        s.loc[m, "tau_hat_mean"], s.loc[m, "tau_hat_std"] = float(d), 0.0
    return s


def summarize_perclass(pc_fits: pd.DataFrame, pc_evals: pd.DataFrame) -> pd.DataFrame:
    e = pc_evals.copy()
    e["exceeds"] = (e.image_loss > e.alpha).astype(float).where(e.image_loss.notna())
    s = e.groupby(["bin", "cls", "alpha"], sort=False).agg(
        n_seeds=("image_loss", "count"),
        image_loss_mean=("image_loss", "mean"), image_loss_std=("image_loss", "std"),
        miss_rate_mean=("miss_rate", "mean"), miss_rate_std=("miss_rate", "std"),
        fppi_mean=("fppi", "mean"), fppi_std=("fppi", "std"),
        exceed_frac=("exceeds", "mean"), n_loss_images_mean=("n_loss_images", "mean"),
        n_gt_mean=("n_gt", "mean")).reset_index()
    f = pc_fits.groupby(["bin", "cls"]).agg(tau_hat_mean=("tau_hat", "mean"), tau_hat_std=("tau_hat", "std"),
                                            infeasible_frac=("infeasible", "mean"),
                                            n_cal_mean=("n_cal", "mean")).reset_index()
    return s.merge(f, on=["bin", "cls"], how="left")


def write(cfg: Cfg, out: dict[str, pd.DataFrame], variant: str = "primary") -> None:
    sub = "" if variant == "primary" else f"sensitivity/{variant}/"
    ff = "%.6g"
    out["splits"].to_csv(results_path(cfg, f"{sub}splits.csv"), index=False)
    out["fits"].to_csv(results_path(cfg, f"{sub}crc_fits.csv"), index=False, float_format=ff)
    out["evals"].to_csv(results_path(cfg, f"{sub}crc_eval.csv"), index=False, float_format=ff)
    out["summary"].to_csv(results_path(cfg, f"{sub}crc_summary.csv"), index=False, float_format=ff)
    out["perclass"].to_csv(results_path(cfg, f"{sub}crc_perclass.csv"), index=False, float_format=ff)
    out["perclass_fits"].to_csv(results_path(cfg, f"{sub}crc_perclass_fits.csv"), index=False, float_format=ff)
    if "tiers" in out:
        out["tiers"].to_csv(results_path(cfg, f"{sub}tiers.csv"), index=False, float_format=ff)
        out["tiers_summary"].to_csv(results_path(cfg, f"{sub}tiers_summary.csv"), index=False, float_format=ff)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--variant", default="primary", choices=["primary", "near_dup_dropped"])
    args = ap.parse_args(argv)
    cfg = load_config(args.config, smoke=args.smoke)
    out = run(cfg, args.variant)
    write(cfg, out, args.variant)
    s = out["summary"]
    pd.set_option("display.width", 250)
    show = s[(s.cls == ALL)][["alpha", "policy", "bin", "image_loss_mean", "image_loss_std", "fppi_mean", "fppi_std",
                              "tau_hat_mean", "tau_hat_std", "exceed_frac", "infeasible_frac", "n_cal_mean", "n_test_mean"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
