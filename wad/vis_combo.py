"""EXPLORATORY (not pre-registered; added after the v2/v3 results): does the trained estimator add
information to the simple pixel score v1? Score = mean of the within-dataset rank of v1 and of v3
(equal weights, no fitting, no labels), compared with v1 and v3 alone using the same evaluation
(wad/vis_eval.py, 200 splits, alpha = 0.2).

Also (exploratory): Spearman within day and within night for v1, v3 and v3's light level alone, to
separate 'is it night' from 'how hard is this image'.

Run: python -m wad.vis_combo   -> results/vis3/exploratory_combo_{policies,correlations}.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from wad import vis_eval as ev
from wad.curves import ALL, Cache


def main():
    from wad.visibility import load_h

    ev.configure(3)
    out = ev.OUT
    h1_all = load_h()
    pols, corr = [], []
    for ds, key, v in ev.configs():
        c = Cache(v)
        h1 = h1_all.loc[c.image_ids].to_numpy()
        h3 = -pd.read_parquet(ev.CACHE / f"pred_{key}.parquet").set_index("image_id").loc[c.image_ids].log_veff.to_numpy()
        combo = (rankdata(h1) + rankdata(h3)) / (2 * len(h1))
        n = c.n_gt(ALL)
        found = np.bincount(c.gt_img, weights=c.gt_score >= 0.25, minlength=c.n_img)
        has = n > 0
        miss = 1 - found[has] / n[has]
        k3 = -pd.read_parquet(ev.CACHE / f"pred_{key}.parquet").set_index("image_id").loc[c.image_ids].log_k.to_numpy()
        night = pd.Series(c.weather).str.endswith("night").to_numpy()
        subsets = [("all images", has)] + ([("day only", has & ~night), ("night only", has & night)] if night.any() else [])
        for sub, m in subsets:
            mm = m[has]  # restrict the per-image miss fractions (defined on images with objects)
            for name, h in (("v1", h1), ("v3", h3), ("v3 light only (-log k)", k3), ("rank mean of v1 and v3", combo)):
                corr.append(dict(dataset=ds, images=sub, score=name, n=int(m.sum()),
                                 spearman_miss025=spearmanr(h[m], miss[mm])[0]))
        ev.V = "v1+v3 rank mean"
        pt, _ = ev.policies_table(ds, v, c, h1, combo, {"v3": h3})
        # fixed physical bands are meaningless for a rank score; per-condition thirds are not needed here
        pols.append(pt[~pt.group.str.contains("exploratory") & ~pt.policy.str.contains("fixed bands")])
    pd.concat(pols).to_csv(out / "exploratory_combo_policies.csv", index=False, float_format="%.6g")
    pd.DataFrame(corr).to_csv(out / "exploratory_combo_correlations.csv", index=False, float_format="%.6g")
    print(pd.DataFrame(corr).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
