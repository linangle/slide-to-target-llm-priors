#!/usr/bin/env python3
"""
trial_summary.py -- compare run folders made by run_trial.py: the landing RMSE by practice slides and method, with the
miss measured both ways (D4), and each run's loss, braking jerk, board and start gap.

    python3 evaluation/trial_summary.py results/side_base/trials_nov20/* [--condition nominal] [--same-as FOLDER] [--out summary.txt]

--same-as FOLDER   also report whether the first folder's results.csv equals FOLDER's (the wrapper changes nothing by itself)
"""
import argparse, csv, json, math
from pathlib import Path

METHODS = ("generic", "lookup", "llm_raw", "oracle")

def rmse(v):
    v = list(v)
    return math.sqrt(sum(x*x for x in v)/len(v)) if v else float("nan")

def load(folder):
    folder = Path(folder)
    rows = list(csv.DictReader((folder/"both_measures.csv").open()))
    for r in rows:
        for k in ("target_x", "peak_speed", "x_rest", "error_x_mm", "error_xy_mm", "miss_rest_mm", "miss_slide_mm", "gap_mm", "accel_limit_fraction"):
            r[k] = float(r[k])
        r["slides"] = int(r["slides"])
    return dict(name=folder.name, rows=rows, settings=json.loads((folder/"settings.json").read_text()),
                override=json.loads((folder/"trial_override.json").read_text()), status=json.loads((folder/"status.json").read_text()))

def table(runs, condition, key, title):
    L = [title, f"{'run':28} " + " ".join(f"{'n=' + str(n) + ' ' + m:>14}" for n in (0, 1, 2) for m in METHODS[:3]) + f" {'oracle':>8}"]
    for run in runs:
        rows = [r for r in run["rows"] if r["condition"] == condition]
        cells = [rmse(r[key] for r in rows if r["slides"] == n and r["method"] == m) for n in (0, 1, 2) for m in METHODS[:3]]
        L.append(f"{run['name']:28} " + " ".join(f"{c:14.2f}" for c in cells) + f" {rmse(r[key] for r in rows if r['method'] == 'oracle'):8.2f}")
    return L

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folders", nargs="+", type=Path)
    p.add_argument("--condition", default="nominal")
    p.add_argument("--same-as", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    runs = [load(f) for f in a.folders if (f/"both_measures.csv").exists()]
    L = ["Trial runs (evaluation/run_trial.py): loss, braking jerk, board, start gap; landing RMSE in mm", ""]
    for run in runs:
        s, o, rows = run["settings"], run["override"], run["rows"]
        L.append(f"{run['name']}: loss {o['loss']}; J {s['jerk']:g}; board {o['board']}; start gap {o['start_gap']}; cap {s['peak_cap']}, practice "
                 f"{s['practice_peak']}, targets {s['targets']}; {run['status']['status']}, {len(rows)} rows, invalid {run['status']['invalid_rows']}; "
                 f"scored pushes {min(r['peak_speed'] for r in rows):.3f}-{max(r['peak_speed'] for r in rows):.3f} m/s, "
                 f"acceleration at most {max(r['accel_limit_fraction'] for r in rows):.3f} of the limit")
    if a.same_as is not None and runs:
        same = (a.folders[0]/"results.csv").read_bytes() == (a.same_as/"results.csv").read_bytes()
        L += ["", f"{a.folders[0].name}/results.csv is {'identical to' if same else 'DIFFERENT from'} {a.same_as}/results.csv"]
    L += [""] + table(runs, a.condition, "error_xy_mm", f"Planar landing RMSE, {a.condition} noise (the evaluator's score)")
    L += [""] + table(runs, a.condition, "miss_rest_mm", f"D4 (a): miss measured from the rest position on the board, along x, {a.condition} noise")
    L += [""] + table(runs, a.condition, "miss_slide_mm", f"D4 (b): miss measured as the slide from the FK release point, {a.condition} noise")
    L += ["", "Difference between the two measures, miss_slide - miss_rest, all rows (mm): mean, SD, largest"]
    for run in runs:
        d = [r["miss_slide_mm"] - r["miss_rest_mm"] for r in run["rows"]]
        m = sum(d)/len(d)
        L.append(f"{run['name']:28} {m:+.3f}  {math.sqrt(sum((x - m)**2 for x in d)/len(d)):.3f}  {max(abs(x) for x in d):.3f}")
    L += ["", "Signed mean miss with no practice, rest position, nominal noise (mm): generic / lookup / LLM, by target"]
    for run in runs:
        rows = [r for r in run["rows"] if r["condition"] == a.condition and r["slides"] == 0]
        cells = []
        for t in sorted({r["target_x"] for r in rows}):
            cells.append(f"{t:g} m: " + " / ".join(f"{sum(r['miss_rest_mm'] for r in rows if r['method'] == m and r['target_x'] == t)/max(1, sum(1 for r in rows if r['method'] == m and r['target_x'] == t)):+.1f}" for m in METHODS[:3]))
        L.append(f"{run['name']:28} " + ";  ".join(cells))
    text = "\n".join(L) + "\n"
    print(text, end="")
    if a.out:
        a.out.write_text(text)

if __name__ == "__main__":
    main()
