#!/usr/bin/env python3
"""
make_mock_measured.py -- regenerate ground_truth_placeholder.csv, the mock of measured_friction.csv.

The mock keeps the same columns as the real file (measured_friction_interface.md): the 21
original columns, then the appended ones from aggregate_pairs.NEW_COLUMNS.  When the file
exists, its first 21 columns are copied byte for byte (scene.py and saved results depend on
those values) and only the appended columns are (re)generated, from a seeded RNG, so the
output is deterministic.  Every row's `sources` says PLACEHOLDER: none of these numbers is a
friction estimate.  Without an existing file, the 21 columns are built from the pair scheme
(M = 9(surface-1) + object, V01-V03) with mock values.

Usage:
    python3 make_mock_measured.py                       # rewrites ground_truth_placeholder.csv in place
    python3 make_mock_measured.py --out mock.csv        # elsewhere
"""
import argparse, csv, math, os
import numpy as np
from aggregate_pairs import BASE_COLUMNS, NEW_COLUMNS

OBJECTS = {"O1": "PLA (3D printed)", "O2": "ABS sheet (extruded)", "O3": "furniture felt", "O4": "neoprene rubber 60A",
           "O5": "corrugated cardboard", "O6": "cast acrylic", "O7": "oak, grain along travel", "O8": "glazed ceramic tile",
           "O9": "stainless steel 304", "O7b": "oak, grain across travel"}
SURFACES = {"S1": "white melamine shelf board", "S2": "birch plywood, sanded", "S3": "painted MDF (satin)",
            "S4": "oak board, grain along travel", "S5": "ABS sheet, smooth face"}
PLACEHOLDER = "PLACEHOLDER synthetic value, not a friction estimate"
REASONS = ["lateral force", "yaw", "short slide", "touched paddle"]

def pair_scheme():
    """(pair_id, object_id, surface_id) for M01-M27 and the validation pairs."""
    pairs = [(f"M{9*(s-1)+o:02d}", f"O{o}", f"S{s}") for s in (1, 2, 3) for o in range(1, 10)]
    return pairs + [("V01", "O9", "S5"), ("V02", "O7", "S4"), ("V03", "O7b", "S4")]

def base_rows_from_scheme(rng):
    """The 21 original columns with mock values, used only when no placeholder file exists."""
    rows = []
    for pid, oid, sid in pair_scheme():
        mu = float(np.clip(math.exp(rng.normal(math.log(0.25), 0.4)), 0.12, 0.48))
        s = 0.100
        rows.append([pid, oid, sid, OBJECTS[oid], SURFACES[sid], f"{mu:.3f}", f"{s:.3f}",
                     f"{mu*math.exp(-1.645*s):.3f}", f"{mu*math.exp(1.645*s):.3f}", PLACEHOLDER,
                     "", "", "", "", "", "", "1", "", "", "", ""])
    return rows

def mock_new_columns(mu, rng):
    """Deterministic mock values for the appended columns, in NEW_COLUMNS order."""
    mu_v2d = mu*(1 + rng.normal(0, 0.04))
    n_rej = int(rng.poisson(0.8))
    reasons = {}
    for r in rng.choice(REASONS, n_rej, p=[0.5, 0.25, 0.15, 0.10]):
        reasons[r] = reasons.get(r, 0) + 1
    vals = {"mu_v2d": f"{mu_v2d:.3f}", "v2d_ratio_mean": f"{mu_v2d/mu:.3f}", "v2d_ratio_sd": f"{rng.uniform(0.03, 0.10):.3f}",
            "mu_slope_per_mps": f"{rng.normal(0, 0.02):.4f}",
            "strip_scale_mm": f"{rng.choice([100, 100, 100, 120, 140, 160, 200]):.0f}",
            "profile_repeatability": f"{rng.uniform(0.3, 0.9):.2f}", "n_rejected": str(n_rej),
            "reject_reasons": ";".join(f"{k}:{n}" for k, n in sorted(reasons.items(), key=lambda kv: -kv[1])),
            "s_lane": f"{abs(rng.normal(0.04, 0.015)):.3f}"}
    return [vals[c] for c in NEW_COLUMNS]

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="ground_truth_placeholder.csv", help="existing mock whose 21 columns are kept")
    ap.add_argument("--out", default="ground_truth_placeholder.csv")
    ap.add_argument("--seed", type=int, default=20261108)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    if os.path.exists(a.src):
        with open(a.src, newline="") as f:
            header, *rows = list(csv.reader(f))
        n = len(BASE_COLUMNS)
        if header[:n] != BASE_COLUMNS:
            raise SystemExit(f"{a.src}: first {n} columns differ from the contract: {header[:n]}")
        rows = [r[:n] for r in rows if r]           # drop any earlier appended columns, keep the 21 verbatim
        print(f"kept the {n} original columns of {len(rows)} rows from {a.src}")
    else:
        rows = base_rows_from_scheme(rng)
        print(f"no {a.src}: built {len(rows)} rows from the pair scheme")
    out = [r + mock_new_columns(float(r[BASE_COLUMNS.index("mu_k_nominal")]), rng) for r in rows]
    with open(a.out, "w", newline="") as f:        # csv defaults: CRLF rows, quotes only where needed, as before
        w = csv.writer(f); w.writerow(BASE_COLUMNS + NEW_COLUMNS); w.writerows(out)
    print(f"wrote {a.out}: {len(out)} rows, {len(BASE_COLUMNS)} + {len(NEW_COLUMNS)} columns (mock, seed {a.seed})")

if __name__ == "__main__":
    main()
