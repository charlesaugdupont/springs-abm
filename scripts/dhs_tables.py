# scripts/dhs_tables.py
"""
Tabulates the Ghana DHS 2014 and 2022 recode files for the household economy and the DHS targets.

For Ghana, the Eastern Region and the DHS clusters within 30 km of Akuse
(cluster coordinates from the SPRINGS compiled survey file ghana_data.csv,
joined on cluster number), all survey-weighted:

* households: size, under-fives, urban share, national wealth quintile, and
  drinking-water source, sanitation and handwashing by quintile;
* the age structure of de jure household members;
* children under five with diarrhoea in the last two weeks: advice or
  treatment sought, ORS and zinc, with cluster-bootstrap 95% intervals
  (clusters resampled within strata), and two-week diarrhoea prevalence;
* women 15-49: health insurance by quintile, work and cash earnings.

Its output is the source of config.EconomyConfig.quintile_shares and of the
Eastern Region WASH gradient behind water_source_by_quintile and
sanitation_by_quintile, of ECON_CARE_TARGETS in
experiments/calibration/history_matching.py, and of
DHS_DIARRHOEA_2WK_BY_QUINTILE in experiments/calibration/validate_calibration.py.

The DHS recode files are not redistributable and are not in this repository:
request them from the DHS Program (https://dhsprogram.com) and pass the folder
that holds them, laid out as in FILES below (ghana_data.csv is read from the
same folder unless --clusters-csv is given).

Usage (run as a module, like the other scripts):
    python -m scripts.dhs_tables --data-dir "SPRINGS DATA" --out-dir outputs/dhs
Writes dhs_tables.txt to --out-dir and prints the same tables.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

AKUSE = (6.0993, 0.12821)
FILES = {
    2022: dict(svy="GHA2022DHS", hr="GHS 2022/GHHR8CDT/GHHR8CFL.DTA", pr="GHS 2022/GHPR8CDT/GHPR8CFL.DTA",
               kr="GHS 2022/GHKR8CDT/GHKR8CFL.DTA", ir="GHS 2022/GHIR8CDT/GHIR8CFL.DTA"),
    2014: dict(svy="GHA2014DHS", hr="GHS 2014/GHHR72DT/GHHR72FL.DTA", pr="GHS 2014/GHPR72DT/GHPR72FL.DTA",
               kr="GHS 2014/GHKR72DT/GHKR72FL.DTA", ir="GHS 2014/GHIR72DT/GHIR72FL.DTA"),
}


def read(data_dir: Path, path, cols):
    lab = pd.read_stata(data_dir / path, iterator=True).variable_labels()
    return pd.read_stata(data_dir / path, columns=[c for c in cols if c in lab], convert_categoricals=False)


def local_clusters(clusters_csv: Path, svy):
    g = pd.read_csv(clusters_csv, usecols=["survey", "Cluster", "Latitude", "Longitude"])
    g = g[g.survey == svy].drop_duplicates("Cluster")
    la, lo = np.radians(g.Latitude), np.radians(g.Longitude)
    la0, lo0 = np.radians(AKUSE[0]), np.radians(AKUSE[1])
    d = 2 * 6371 * np.arcsin(np.sqrt(np.sin((la - la0) / 2) ** 2 + np.cos(la0) * np.cos(la) * np.sin((lo - lo0) / 2) ** 2))
    return set(g.Cluster.str[-5:].astype(int)[d <= 30])


def wmean(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float)
    ok = ~np.isnan(x)
    return float((x[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else np.nan


def boot_ci(df, val, wcol, clcol, stcol, B, rng):
    """Weighted mean of val with a cluster bootstrap (clusters resampled within strata)."""
    est = wmean(df[val], df[wcol])
    groups = [g for _, g in df.groupby(stcol)]
    clusters = [{c: gg for c, gg in g.groupby(clcol)} for g in groups]
    reps = []
    for _ in range(B):
        parts = []
        for cl in clusters:
            keys = list(cl)
            for k in rng.choice(len(keys), len(keys), replace=True):
                parts.append(cl[keys[k]])
        b = pd.concat(parts)
        reps.append(wmean(b[val], b[wcol]))
    lo, hi = np.nanpercentile(reps, [2.5, 97.5])
    return est, lo, hi, len(df)


def areas(df, regcol, clcol, local):
    return {"Ghana": df, "Eastern": df[df[regcol] == 5], "Near Akuse (30 km)": df[df[clcol].isin(local)]}


WATER = {"piped": [10, 11, 12, 13, 14], "ground": [20, 21, 30, 31, 32], "packaged": [71, 72],
         "surface": [43], "other": [40, 41, 42, 51, 61, 62, 96]}
SAN = {"sewer": [11], "improved on-site": [12, 13, 16, 21, 22, 41], "unimproved": [14, 15, 23, 42, 43, 96],
       "open defecation": [30, 31]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data-dir", required=True, type=Path,
                    help="folder holding the DHS recode files, laid out as in FILES")
    ap.add_argument("--clusters-csv", type=Path, default=None,
                    help="SPRINGS compiled survey file with cluster coordinates (default: DATA_DIR/ghana_data.csv)")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs") / "dhs")
    args = ap.parse_args()
    data_dir = args.data_dir
    clusters_csv = args.clusters_csv or data_dir / "ghana_data.csv"
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(1)

    lines = []
    P = lambda s="": lines.append(s)
    for year, F in FILES.items():
        local = local_clusters(clusters_csv, F["svy"])
        P(f"\n######## DHS {year}: {len(local)} clusters within 30 km of Akuse\n")

        # ---------------- households (HR)
        hr = read(data_dir, F["hr"], ["hv001", "hv002", "hv005", "hv021", "hv022", "hv024", "hv025", "hv270", "hv009", "hv014",
                                      "hv201", "hv205", "hv225", "hv230a", "hv230b", "hv232"])
        hr["w"] = hr.hv005 / 1e6
        hr["has_u5"] = (hr.hv014 > 0).astype(float)
        hr["basic_hw"] = (hr.hv230a.isin([1, 2]) & (hr.hv230b == 1) & (hr.hv232 == 1)).astype(float)
        hr.loc[~hr.hv230a.isin([1, 2, 3]), "basic_hw"] = np.nan          # not observed for other reasons: missing
        for k, codes in WATER.items():
            hr[f"w_{k}"] = hr.hv201.isin(codes).astype(float)
        for k, codes in SAN.items():
            hr[f"s_{k}"] = hr.hv205.isin(codes).astype(float)
        P("Households")
        P(f"{'area':22s} {'n':>6s} {'mean size':>9s} {'1':>5s} {'2':>5s} {'3':>5s} {'4':>5s} {'5':>5s} {'6+':>5s} {'with u5':>8s} {'u5|any':>7s} {'urban':>6s}")
        for a, d in areas(hr, "hv024", "hv001", local).items():
            sz = d.hv009.clip(upper=6)
            dist = [wmean((sz == k).astype(float), d.w) for k in range(1, 7)]
            P(f"{a:22s} {len(d):6d} {wmean(d.hv009, d.w):9.2f} " + " ".join(f"{x:5.2f}" for x in dist)
              + f" {wmean(d.has_u5, d.w):8.2f} {wmean(d.hv014[d.hv014 > 0], d.w[d.hv014 > 0]):7.2f} {wmean((d.hv025 == 1).astype(float), d.w):6.2f}")
        P("\nHouseholds by national wealth quintile (share of households)")
        for a, d in areas(hr, "hv024", "hv001", local).items():
            P(f"  {a:22s} " + " ".join(f"{wmean((d.hv270 == q).astype(float), d.w):6.3f}" for q in range(1, 6)))
        for a in ("Eastern", "Near Akuse (30 km)"):
            d = areas(hr, "hv024", "hv001", local)[a]
            P(f"\nWASH by wealth quintile, {a} (household shares; n per quintile)")
            cols = [f"w_{k}" for k in WATER] + [f"s_{k}" for k in SAN] + ["basic_hw"]
            P(f"  {'quintile':8s} {'n':>5s} " + " ".join(f"{c[2:][:9]:>9s}" for c in cols))
            for q in list(range(1, 6)) + ["all"]:
                dq = d if q == "all" else d[d.hv270 == q]
                P(f"  {str(q):8s} {len(dq):5d} " + " ".join(f"{wmean(dq[c], dq.w):9.3f}" for c in cols))

        # ---------------- people (PR), de jure
        pr = read(data_dir, F["pr"], ["hv001", "hv005", "hv024", "hv102", "hv105"])
        pr = pr[(pr.hv102 == 1) & (pr.hv105 < 98)]
        pr["w"] = pr.hv005 / 1e6
        P("\nAge structure, de jure household members (share of persons)")
        P(f"  {'area':22s} {'n':>6s} {'0-4':>6s} {'5-14':>6s} {'15-64':>6s} {'65+':>6s}")
        for a, d in areas(pr, "hv024", "hv001", local).items():
            bands = [(0, 4), (5, 14), (15, 64), (65, 200)]
            P(f"  {a:22s} {len(d):6d} " + " ".join(f"{wmean(d.hv105.between(lo, hi).astype(float), d.w):6.3f}" for lo, hi in bands))

        # ---------------- children (KR): diarrhoea and care-seeking
        lab = pd.read_stata(data_dir / F["kr"], iterator=True).variable_labels()
        srcs = [c for c in lab if c.startswith("h12") and c not in ("h12y", "h12z") and not lab[c].startswith("na -")]
        kr = read(data_dir, F["kr"], ["v001", "v005", "v021", "v022", "v024", "v025", "v190", "v481", "b5", "h11", "h13", "h13b", "h15e"] + srcs)
        kr = kr[kr.b5 == 1].copy()
        kr["w"] = kr.v005 / 1e6
        kr["diar"] = kr.h11.isin([1, 2]).astype(float)
        kr.loc[kr.h11 == 8, "diar"] = np.nan
        d2 = kr[kr.diar == 1].copy()
        pharm = ["h12k"]
        nonmed = [c for c in srcs if any(s in lab[c].lower() for s in ("shop", "traditional", "drug peddler"))
                  or lab[c].lower().strip() == "diarrhea: other"]
        facility = [c for c in srcs if c not in pharm + nonmed]
        d2["any_src"] = (d2[srcs] == 1).any(axis=1).astype(float)
        d2["facility"] = (d2[facility] == 1).any(axis=1).astype(float)
        d2["facility_or_pharm"] = (d2[facility + pharm] == 1).any(axis=1).astype(float)
        d2["ors"] = (d2.h13.isin([1, 2]) | (d2.get("h13b", 0) == 1)).astype(float)
        d2["zinc"] = (d2.h15e == 1).astype(float)
        P(f"\nChildren under 5 with diarrhoea in the last 2 weeks: advice or treatment (weighted %, 95% bootstrap CI)")
        P(f"  non-medical sources counted: {nonmed}; pharmacy: {pharm}")
        P(f"  {'group':34s} {'n':>5s} {'any source':>20s} {'facility/provider':>20s} {'fac. or pharmacy':>18s} {'ORS':>6s} {'zinc':>6s} {'diarr. prev':>11s}")
        groups = {**{f"{a}": d for a, d in areas(d2, "v024", "v001", local).items()}}
        for q in range(1, 6):
            groups[f"Ghana, quintile {q}"] = d2[d2.v190 == q]
        for q in range(1, 6):
            groups[f"Eastern, quintile {q}"] = d2[(d2.v024 == 5) & (d2.v190 == q)]
        allkids = {**{a: d for a, d in areas(kr, "v024", "v001", local).items()},
                   **{f"Ghana, quintile {q}": kr[kr.v190 == q] for q in range(1, 6)},
                   **{f"Eastern, quintile {q}": kr[(kr.v024 == 5) & (kr.v190 == q)] for q in range(1, 6)}}
        for name, d in groups.items():
            if len(d) < 5:
                P(f"  {name:34s} {len(d):5d}  (too few)"); continue
            B = 1000 if not name.startswith("Ghana") or name == "Ghana" else 300
            e1, l1, h1, n = boot_ci(d, "any_src", "w", "v001", "v022", B, rng)
            e2, l2, h2, _ = boot_ci(d, "facility", "w", "v001", "v022", B, rng) if name in ("Ghana", "Eastern", "Near Akuse (30 km)") else (wmean(d.facility, d.w), np.nan, np.nan, n)
            prev = wmean(allkids[name].diar, allkids[name].w)
            P(f"  {name:34s} {n:5d} {100*e1:6.1f} ({100*l1:4.1f}-{100*h1:4.1f}) {100*e2:6.1f} ({100*l2:4.1f}-{100*h2:4.1f}) "
              f"{100*wmean(d.facility_or_pharm, d.w):18.1f} {100*wmean(d.ors, d.w):6.1f} {100*wmean(d.zinc, d.w):6.1f} {100*prev:11.1f}")

        # ---------------- women (IR): insurance and work
        ir = read(data_dir, F["ir"], ["v001", "v005", "v024", "v190", "v481", "v714", "v717", "v741"])
        ir["w"] = ir.v005 / 1e6
        P("\nWomen 15-49: health insurance (v481) by wealth quintile, and work")
        for a, d in areas(ir, "v024", "v001", local).items():
            ins = [wmean((d.v481[d.v190 == q] == 1).astype(float), d.w[d.v190 == q]) for q in range(1, 6)]
            working = wmean((d.v714 == 1).astype(float), d.w)
            cash = wmean(d.v741.isin([1, 2]).astype(float)[d.v714 == 1], d.w[d.v714 == 1])
            occ = d[d.v714 == 1].groupby("v717").w.sum() / d[d.v714 == 1].w.sum()
            P(f"  {a:22s} n={len(d):5d} insured {wmean((d.v481 == 1).astype(float), d.w):.3f}; by quintile "
              + " ".join(f"{x:.2f}" for x in ins) + f"; working {working:.2f}; paid in cash (of working) {cash:.2f}")
            P("     occupation of working women: " + ", ".join(f"{int(k)}:{v:.2f}" for k, v in occ.items() if v >= 0.03))

    (args.out_dir / "dhs_tables.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
