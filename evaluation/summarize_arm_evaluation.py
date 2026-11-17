"""Summarize one complete v3 evaluation folder, keeping noise conditions separate."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from run_arm_evaluation import VERSION, VERSIONS, METHODS, expected_keys


def load(folder):
    config=json.loads((folder/'settings.json').read_text())
    status=json.loads((folder/'status.json').read_text())
    if config['version'] not in VERSIONS or status['status']!='complete':
        raise ValueError('Requires a complete v3 or v4-layout run; inspect status.json/preflight.json for failures')
    with (folder/'results.csv').open(newline='') as f: rows=list(csv.DictReader(f))
    seen=set()
    for r in rows:
        r['seed'],r['slides']=int(r['seed']),int(r['slides'])
        for field in ('target_x','mu_true','mu_median','mu_low','mu_high','peak_speed','x_rest','y_rest',
                      'error_x_mm','error_xy_mm','q_limit_fraction','v_limit_fraction','ff_torque_limit_fraction'):
            r[field]=float(r[field])
            if not math.isfinite(r[field]): raise ValueError(f'Nonfinite {field}')
        key=(r['pair'],r['condition'],r['seed'],r['target_x'],r['method'],r['slides'])
        if key in seen: raise ValueError(f'Duplicate row {key}')
        seen.add(key)
        if r['valid']!='True': raise ValueError(f'Invalid shot {key}; not silently excluded')
        if not 0<r['mu_low']<=r['mu_median']<=r['mu_high']: raise ValueError('Invalid posterior interval')
        if not 0<=r['peak_speed']<=config['peak_cap']+1e-8: raise ValueError('Speed cap violation')
        if any(r[k]>1 for k in ('q_limit_fraction','v_limit_fraction','ff_torque_limit_fraction')):
            raise ValueError('Reported planned joint-limit violation')
        if r['at_cap'] not in ('True','False'): raise ValueError('Malformed cap flag')
        if not math.isclose(r['error_xy_mm'],1000*math.hypot(r['x_rest']-r['target_x'],r['y_rest']),abs_tol=1e-6):
            raise ValueError('Score inconsistent with saved positions')
    if seen!=expected_keys(config) or len(rows)!=status['rows_written']:
        raise ValueError('Incomplete/unexpected evaluation matrix')
    return config,rows


def metrics(rows):
    # One belief per pair/seed/method/n/condition. Targets reuse it, so remove
    # target repetitions from friction metrics and verify they are identical.
    beliefs={}
    for r in rows:
        key=(r['pair'],r['condition'],r['seed'],r['method'],r['slides'])
        vals=tuple(r[k] for k in ('mu_true','mu_median','mu_low','mu_high'))
        if key in beliefs and beliefs[key]!=vals: raise ValueError('Belief changed across targets')
        beliefs[key]=vals
    n=len(rows); b=list(beliefs.values())
    oracle=rows[0]['method']=='oracle'
    return dict(scored_rows=n,belief_rows=len(b),
        planar_rmse_mm=math.sqrt(sum(r['error_xy_mm']**2 for r in rows)/n),
        signed_x_bias_mm=sum(r['error_x_mm'] for r in rows)/n,
        friction_mae=sum(abs(m-t) for t,m,lo,hi in b)/len(b),
        coverage_pct='' if oracle else 100*sum(lo<=t<=hi for t,m,lo,hi in b)/len(b),
        mean_width='' if oracle else sum(hi-lo for t,m,lo,hi in b)/len(b),
        at_cap_pct=100*sum(r['at_cap']=='True' for r in rows)/n)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path)
    p.add_argument('--out',type=Path)
    a=p.parse_args(); config,rows=load(a.folder)
    out=a.out or a.folder/'summary'
    groups=defaultdict(list); per_pair=defaultdict(list); per_target=defaultdict(list)
    for r in rows:
        key=(r['condition'],r['slides'],r['method'])
        groups[key].append(r);per_pair[(r['pair'],*key)].append(r);per_target[(r['target_x'],*key)].append(r)
    aggregate=[dict(condition=c,slides=n,method=m,**metrics(groups[(c,n,m)]))
               for c in config['conditions'] for n in range(3) for m in METHODS]
    pair_rows=[dict(pair=key[0],condition=key[1],slides=key[2],method=key[3],**metrics(rs)) for key,rs in sorted(per_pair.items())]
    target_rows=[dict(target_x=key[0],condition=key[1],slides=key[2],method=key[3],**metrics(rs)) for key,rs in sorted(per_target.items())]
    out.mkdir(parents=True,exist_ok=False)
    for name,data in [('aggregate.csv',aggregate),('per_pair.csv',pair_rows),('per_target.csv',target_rows)]:
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
    lines=[f'VALIDATED: {len(rows)} scored rows; complete selected matrix; all validity flags pass.',
           'Conditions reported separately; friction metrics count each shared belief once across targets.']
    arm=config.get('arm')   # absent in runs saved before 16 Nov: the v3 protocol (origin, 0.45 m/s, 12 m/s^2 brake cap)
    if arm:
        accel=[r['accel_limit_fraction'] for r in rows if r.get('accel_limit_fraction') not in (None,'')]
        lines.append(f"Arm: layout {arm['layout']}, base {arm['base'][:2]}, brake cap {arm['brake_cap']:g} m/s^2; peak cap {config['peak_cap']} m/s "
                     f"({arm['peak_cap_rule']}); practice {config['practice_peak']} m/s; targets {config['targets']} ({arm['targets_rule']}); "
                     f"limits required: {', '.join(arm['checked_limits'])}"
                     +(f"; peak planned acceleration / limit over the scored pushes: {max(map(float,accel)):.2f}" if accel else ''))
    for c in config['conditions']:
        lines+=['',f'CONDITION: {c}', 'method    slides   RMSE mm    mu MAE   coverage   width   at cap']
        for r in aggregate:
            if r['condition']!=c: continue
            cov='N/A' if r['coverage_pct']=='' else f"{r['coverage_pct']:.1f}%"
            width='N/A' if r['mean_width']=='' else f"{r['mean_width']:.4f}"
            lines.append(f"{r['method']:9s} {r['slides']:6d} {r['planar_rmse_mm']:9.3f} {r['friction_mae']:9.4f} {cov:>9s} {width:>7s} {r['at_cap_pct']:7.1f}%")
    lines+=['','Limits: three previously inspected published pairs; seeds are not new materials.',
            'No-added-noise is one deterministic diagnostic with nominal likelihood SDs.',
            'Coverage is of fitted grid posteriors, not original LLM intervals.',
            'Repeated zero-practice/oracle rows and reused rollouts are not independent evidence.',
            'Simulator-state sensing and ideal contact-event timing; no rendered camera or calibration.',
            'Joint checks cover planned position/velocity and feed-forward torque'+(', and planned acceleration.' if arm and 'a' in arm['checked_limits'] else ' only.')]
    report='\n'.join(lines)+'\n';(out/'summary.txt').write_text(report);print(report)
    print(f'Saved summaries to {out}')


if __name__=='__main__': main()
