"""Validate and summarize the ten fixed-settings arm pilot runs. No API or Drake.

Run from the repository: python evaluation/summarize_arm_pilot.py
Outputs pilot_summary/summary.txt and aggregate/per-pair CSVs.
Coverage describes the fitted/truncated grid posterior, NOT original LLM intervals.
"""
import argparse
import csv
import json
import math
from pathlib import Path
from collections import defaultdict
from paths import RESULTS

METHODS = ('generic', 'lookup', 'llm_raw', 'oracle')
PAIRS = ('Y01', 'Y02', 'Y03')


def read_runs(root):
    all_rows = []
    reference = None
    for seed in range(10):
        folder = root / f'arm_pilot_seed_{seed}'
        config = json.loads((folder/'settings.json').read_text())
        if config.get('seed') != seed or sorted(config.get('pairs', [])) != list(PAIRS):
            raise ValueError(f'{folder}: unexpected seed or pair list')
        comparable = {k: v for k, v in config.items() if k != 'seed'}
        if reference is None:
            reference = comparable
        elif comparable != reference:
            raise ValueError(f'{folder}: settings or source/cache hashes differ between runs')
        with (folder/'results.csv').open(newline='') as f:
            rows = list(csv.DictReader(f))
        expected = {(pair, method, n) for pair in PAIRS for method in METHODS for n in range(3)}
        seen = set()
        for row in rows:
            key = (row['pair'], row['method'], int(row['slides']))
            if key not in expected or key in seen or int(row['seed']) != seed:
                raise ValueError(f'{folder}: duplicate/unexpected row {key}')
            seen.add(key)
            if row['valid'] != 'True':
                raise ValueError(f'{folder}: invalid rollout {key}; not silently excluded')
            for name in ('mu_true', 'mu_median', 'mu_low', 'mu_high', 'peak_speed',
                         'error_x_mm', 'error_xy_mm', 'q_limit_fraction',
                         'v_limit_fraction', 'ff_torque_limit_fraction'):
                row[name] = float(row[name])
                if not math.isfinite(row[name]):
                    raise ValueError(f'{folder}: nonfinite {name} for {key}')
            if not 0 < row['mu_low'] <= row['mu_median'] <= row['mu_high']:
                raise ValueError(f'{folder}: invalid posterior interval for {key}')
            if any(row[k] > 1 for k in ('q_limit_fraction','v_limit_fraction','ff_torque_limit_fraction')):
                raise ValueError(f'{folder}: joint-limit violation for {key}')
            if row['peak_speed'] > config['peak_cap'] + 1e-8:
                raise ValueError(f'{folder}: speed-cap violation for {key}')
            if row['error_xy_mm'] < abs(row['error_x_mm']) - 1e-7:
                raise ValueError(f'{folder}: inconsistent planar and along-push errors')
            if row['at_cap'] not in ('True','False'):
                raise ValueError(f'{folder}: malformed cap flag')
            row['slides'], row['seed'] = key[2], seed
            all_rows.append(row)
        if seen != expected:
            raise ValueError(f'{folder}: incomplete run; missing {expected-seen}')
    return all_rows


def summarize(rows):
    count = len(rows)
    oracle = rows[0]['method'] == 'oracle'
    mean = lambda values: sum(values)/count
    return dict(n=count,
                planar_rmse_mm=math.sqrt(mean([r['error_xy_mm']**2 for r in rows])),
                mean_planar_error_mm=mean([r['error_xy_mm'] for r in rows]),
                signed_x_bias_mm=mean([r['error_x_mm'] for r in rows]),
                friction_mae=mean([abs(r['mu_median']-r['mu_true']) for r in rows]),
                posterior_coverage_pct='' if oracle else 100*mean([r['mu_low'] <= r['mu_true'] <= r['mu_high'] for r in rows]),
                mean_interval_width='' if oracle else mean([r['mu_high']-r['mu_low'] for r in rows]),
                at_cap_pct=100*mean([r['at_cap']=='True' for r in rows]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=RESULTS/'pilot_v1')
    parser.add_argument('--out', type=Path, default=RESULTS/'pilot_v1'/'pilot_summary')
    args = parser.parse_args()
    rows = read_runs(args.root)  # Validate everything before creating outputs.
    groups = defaultdict(list)
    by_pair = defaultdict(list)
    for r in rows:
        groups[(r['method'],r['slides'])].append(r)
        by_pair[(r['pair'],r['method'],r['slides'])].append(r)
    aggregate = [dict(method=m,slides=n,**summarize(groups[(m,n)])) for n in range(3) for m in METHODS]
    per_pair = [dict(pair=p,method=m,slides=n,**summarize(by_pair[(p,m,n)])) for p in PAIRS for n in range(3) for m in METHODS]
    lines = ['VALIDATED: 10 complete runs, 360 valid scored shots, identical settings/source hashes.',
             'Equal weighting: 3 pairs x 10 noise seeds per method/practice count.',
             '', 'method    slides   RMSE mm   mu MAE   posterior coverage   mean width   at cap']
    for r in aggregate:
        cov = 'N/A' if r['method']=='oracle' else f"{r['posterior_coverage_pct']:.1f}%"
        width = 'N/A' if r['method']=='oracle' else f"{r['mean_interval_width']:.4f}"
        lines.append(f"{r['method']:9s} {r['slides']:6d} {r['planar_rmse_mm']:9.3f} "
                     f"{r['friction_mae']:8.4f} {cov:>20s} {width:>12s} {r['at_cap_pct']:7.1f}%")
    lines += ['', 'Within-pair noise sensitivity (planar RMSE, mm):',
              'pair slides   generic    lookup   llm_raw    oracle']
    for p in PAIRS:
        for n in range(3):
            vals = [summarize(by_pair[(p,m,n)])['planar_rmse_mm'] for m in METHODS]
            lines.append(f'{p} {n:6d} ' + ' '.join(f'{v:9.3f}' for v in vals))
    lines += ['', 'Interpretation limits:',
              '- Coverage is for the fitted/truncated grid posterior, not the original LLM interval.',
              '- Zero-practice and oracle repetitions are deterministic duplicates, not independent evidence.',
              '- Ten seeds vary observation noise; they do not create ten new material pairs.',
              '- Only three material pairs, including two with the same simulator friction.',
              '- Pilot priors, ideal release timing, and noisy-state sensing remain provisional.',
              '- Reported joint checks cover planned motion and feed-forward torque, not total actuator torque.']
    args.out.mkdir(parents=True, exist_ok=False)
    for name, data in [('aggregate.csv',aggregate),('per_pair.csv',per_pair)]:
        with (args.out/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    text='\n'.join(lines)+'\n'
    (args.out/'summary.txt').write_text(text)
    print(text)
    print(f'Saved summaries to {args.out}')


if __name__ == '__main__':
    main()
