"""Version 3 evaluation infrastructure: multiple targets and observation-noise levels.

No API calls. Uses existing published-pair cache. Positions come from Drake state,
not camera perception; release timing uses the ideal contact event. Dataset remains
three exploratory pairs. Old runners/results are untouched. Deterministic scored
rollouts with the exact same (pair, speed) are reused within a run.

Arm setup (16 Nov). With no options the run is the v3 protocol as saved: base at the
origin, 0.45 m/s cap and practice speed, targets 0.60/0.62/0.64 m. The options below
try another setup without changing that default; settings.json records which one ran,
and such runs belong in their own results folder, never pooled with the v3 checks.
  --layout side          the iiwa beside the board (arm.LAYOUTS), or --base-x/--base-y
  --a-brake 4            cap on the paddle's braking deceleration, m/s^2 (default 12)
  --check-accel          also require the planned joint accelerations within the SDF limits
  --peak-cap auto        the layout's fastest push within the checked joint limits
                         (arm.limited_peak_speed) instead of a hand-set number
  --practice-peak auto   the fastest practice push, at most the cap, that keeps the slide
                         on the board for the generic prior's slipperiest 5% (decide.practice_speed)
  --targets auto         targets re-derived from the cap (arm.target_grid): from x0 + 5 cm to
                         what the highest-friction pair reaches at 90% of the cap
  --clip-unreachable     a target the cap cannot reach is pushed at the cap and scored as a
                         miss (the proposal's rule) instead of stopping the run
"""
import argparse
import csv
import functools
import hashlib
import itertools
import json
import math
from pathlib import Path
from paths import locate

import numpy as np
from belief import theta_grid, logprior, posterior, loglik_lookup, lookup_observation, update_with_slides
from llm_prior import build_prompt, cached_response, parse_response
from demo_published_comparison import OBJECT, SURFACES, raw_llm_logprior
from run_arm_comparison_v2 import PRIOR_MEDIAN, PRIOR_LOG_SD, execute

ROOT = None   # None: inputs and sources come from the repository's folders (paths.locate).  The offline tests set it to a
              # folder that holds a copy of every one of them.
def _file(name):
    return ROOT/name if ROOT is not None else locate(name)
VERSION = 'multi_target_noise_v3'
METHODS = ('generic', 'lookup', 'llm_raw', 'oracle')
NOISE = {'low': .5, 'nominal': 1., 'high': 2., 'no_added': 0.}
FIELDS = ['pair','condition','seed','target_x','method','slides','mu_true','mu_median',
          'mu_low','mu_high','peak_speed','at_cap','x_rest','y_rest','error_x_mm',
          'error_xy_mm','q_limit_fraction','v_limit_fraction','ff_torque_limit_fraction',
          'valid','reused_rollout','accel_limit_fraction']
X0, JERK, V3_CAP = .55, 100., .45


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def condition_seeds(condition, seeds):
    # The no-added-noise diagnostic is deterministic and runs exactly once.
    return [-1] if condition == 'no_added' else seeds


def expected_keys(config):
    return {(p,c,s,t,m,n) for p in config['pairs'] for c in config['conditions']
            for s in condition_seeds(c,config['seeds']) for t in config['targets']
            for m in METHODS for n in range(3)}


def make_observation(result, pusher, rng, condition):
    from observe import Obs
    from physics import OBJECT_SIZE
    scale = NOISE[condition]
    # Keep strictly positive nominal likelihood SDs in the no-added diagnostic.
    likelihood_scale = scale if scale else 1.
    tr, t0 = result['traj'], result['t_force_zero']
    moving = np.flatnonzero((tr['t'] > t0) & (np.abs(tr['vx']) > .005))
    if not len(moving):
        raise ValueError('No measurable free slide')
    ts = np.arange(1, int((tr['t'][moving[-1]]-t0)*60)+1)/60.
    if len(ts) < 2:
        raise ValueError('Fewer than two free-slide frames')
    # Draw in the same order as v2: frames, FK, rest. Re-seeding per condition
    # shares standard-normal draws across noise levels; slides advance the RNG.
    z_frame, z_fk, z_rest = rng.standard_normal(len(ts)), rng.standard_normal(), rng.standard_normal()
    x = np.interp(t0+ts,tr['t'],tr['x']) + .002*scale*z_frame
    q = float(np.interp(t0,tr['t'],tr['q']))
    fk = float(pusher.face_x(q)+OBJECT_SIZE[0]/2)+.0005*scale*z_fk
    rest = result['x_rest']+.0005*scale*z_rest
    return Obs(ts,x,float(rest),float(fk),.002*likelihood_scale,
               .0005*likelihood_scale,.0005*likelihood_scale)


def obs_record(obs):
    return dict(t=obs.t.tolist(), x=obs.x.tolist(), x_rest=obs.x_rest, x_fk=obs.x_fk,
                sigma_f=obs.sigma_f,sigma_rest=obs.sigma_rest,sigma_fk=obs.sigma_fk)


def arm_setup(args, truth, generic_post):
    """The arm's layout, speed cap, practice speed and targets for this run. Defaults reproduce v3
    and, like the rest of a configuration check, need no Drake: arm.py is imported only for the options."""
    BASE, A_BRAKE = (0.,0.,0.), 12.       # arm.BASE, arm.A_BRAKE
    layout = getattr(args,'layout',None)
    bx, by = getattr(args,'base_x',None), getattr(args,'base_y',None)
    base = (bx or 0., by or 0., 0.) if bx is not None or by is not None else None
    if layout:
        from arm import layout_base
        base = layout_base(layout)
    a_brake = getattr(args,'a_brake',None)
    check = ('q','v','tau','a') if getattr(args,'check_accel',False) else ('q','v','tau')
    cap = getattr(args,'peak_cap',V3_CAP)
    cap_rule = 'fixed'
    if cap == 'auto':
        from arm import limited_peak_speed
        which = tuple(k for k in check if k != 'q')
        cap = math.floor(1000*limited_peak_speed(JERK,X0,which=which,base=tuple(base or BASE),a_max=a_brake or A_BRAKE))/1000
        cap_rule = f"arm.limited_peak_speed within {'/'.join(which)} limits, floored to the mm/s"
        if cap <= 0:
            raise ValueError('No push is within the checked joint limits for this layout and brake cap')
    cap = float(cap)
    practice = getattr(args,'practice_peak',None)
    practice_rule = 'fixed' if practice is not None else 'equal to the peak cap'
    if practice == 'auto':
        from decide import Release, practice_speed, slide_room
        practice = math.floor(1000*practice_speed(cap,slide_room(X0),generic_post,Release(J=JERK)))/1000
        practice_rule = "decide.practice_speed: on the board for the generic prior's 5% quantile, at most the cap"
    practice = cap if practice is None else float(practice)
    targets = list(getattr(args,'targets',[.60,.62,.64]))
    targets_rule = 'given'
    if targets == ['auto']:
        from arm import target_grid
        mu_hi = max(truth[p] for p in SURFACES)
        targets = target_grid(cap,mu_hi,getattr(args,'n_targets',3),JERK,X0)
        targets_rule = f'arm.target_grid: x0 + 5 cm to the reach of mu = {mu_hi} (the highest among {"/".join(SURFACES)}) at 90% of the cap'
    targets = [float(t) for t in targets]
    options = {}
    if base is not None: options['base'] = tuple(base)
    if a_brake is not None: options['a_brake'] = float(a_brake)
    if check != ('q','v','tau'): options['check'] = check
    record = dict(layout=layout or ('custom' if base is not None else 'origin'),base=list(base or BASE),
                  brake_cap=float(a_brake or A_BRAKE),checked_limits=list(check),peak_cap_rule=cap_rule,
                  practice_rule=practice_rule,targets_rule=targets_rule,
                  clip_unreachable=bool(getattr(args,'clip_unreachable',False)))
    return dict(cap=cap,practice=practice,targets=targets,options=options,record=record)


def checked_execute(mu,u,**options):
    r,p,limits,valid = execute(mu,u,**options)
    valid = bool(valid and all(math.isfinite(float(r[k])) for k in ('x_rest','y_rest'))
                 and all(math.isfinite(v) for v in limits.values()))
    return r,p,limits,valid


def run(args, executor=checked_execute, planner=None):
    pairs = list(SURFACES) if args.pair == 'all' else [args.pair]
    cache = _file('llm_cache_published_pilot.json')
    with _file('ground_truth_published.csv').open(newline='') as f:
        truth = {r['pair_id']:float(r['mu_k_nominal']) for r in csv.DictReader(f)}
    g = theta_grid(args.grid_points,math.log(.005),math.log(5.))
    generic = logprior(g,math.log(PRIOR_MEDIAN),PRIOR_LOG_SD)
    initials = {}
    for pair in pairs:
        prompt = build_prompt(dict(description_object=OBJECT,description_surface=SURFACES[pair]))
        raw = cached_response(prompt,cache)
        if raw is None:
            raise ValueError(f'Cache miss for {pair}; no request made')
        table = lookup_observation(pair,str(_file('friction_lookup.csv')))
        initials[pair] = dict(generic=generic,
            lookup=generic.copy() if table is None else generic+loglik_lookup(g,*table),
            llm_raw=raw_llm_logprior(g,parse_response(raw)))
    setup = arm_setup(args,truth,posterior(g,generic))
    cap, practice_peak, args.targets = setup['cap'], setup['practice'], setup['targets']
    if executor is checked_execute and setup['options']:
        executor = functools.partial(checked_execute,**setup['options'])
    plan_options = dict(clip=True) if setup['record']['clip_unreachable'] else {}
    config = dict(version=VERSION,status='infrastructure_for_exploratory_pairs',
                  pairs=pairs,conditions=args.noise,seeds=args.seeds,targets=args.targets,
                  x0=X0,peak_cap=cap,practice_peak=practice_peak,jerk=JERK,fps=60,arm=setup['record'],
                  noise_scales={c:NOISE[c] for c in args.noise},nominal_sd=[.002,.0005,.0005],
                  no_added_likelihood='nominal positive SDs; zero injected noise; seed label -1',
                  grid_points=args.grid_points,grid_mu_bounds=[.005,5.],
                  generic_median=PRIOR_MEDIAN,generic_log_sd=PRIOR_LOG_SD,
                  llm='raw approximate lognormal, uncalibrated',
                  observations='noisy simulator positions, ideal contact-event release timing',
                  random_sharing='same standard-normal draws across noise levels; same observations across methods/targets',
                  limits='planned q, velocity, feed-forward torque; not total actuator torque',
                  rollout_reuse='deterministic state, exact pair and float speed; no approximate cache rounding')
    files = ['run_arm_evaluation.py','run_arm_comparison_v2.py','demo_published_comparison.py',
             'llm_prior.py','belief.py','observe.py','arm.py','push.py','physics.py','scene.py',
             'drake_config.py','sim_config.yaml','ground_truth_published.csv','friction_lookup.csv',
             'llm_cache_published_pilot.json']
    config['sha256']={name:hashlib.sha256(_file(name).read_bytes()).hexdigest() for name in files}
    from importlib.metadata import version, PackageNotFoundError
    config['packages']={}
    for name in ('numpy','scipy','drake','manipulation','openai'):
        try: config['packages'][name]=version(name)
        except PackageNotFoundError: config['packages'][name]='not installed'
    config['expected_rows']=len(expected_keys(config))
    if args.check_only:
        p=posterior(g,generic)
        print(f"Configuration/cache check passed: {config['expected_rows']} scored rows planned.")
        print(f"Default median={p['mu_median']:.5f}, interval=[{p['mu_q05']:.5f}, {p['mu_q95']:.5f}]")
        print(f"Arm: {setup['record']}; peak cap {cap} m/s, practice {practice_peak} m/s, targets {args.targets}")
        print('No Drake feasibility check performed. No files written or API requests made.')
        return
    if planner is None:
        from arm import plan_peak_speed
        planner=plan_peak_speed
    args.out.mkdir(parents=True,exist_ok=False)
    write_json(args.out/'settings.json',config)
    status=dict(status='running',rows_written=0,expected_rows=config['expected_rows'],invalid_rows=0)
    write_json(args.out/'status.json',status)
    physics_cache={}
    def shot(pair,u):
        key=(pair,float(u))
        reused=key in physics_cache
        if not reused:
            r,_,lim,valid=executor(truth[pair],u)
            physics_cache[key]=(dict(x_rest=r['x_rest'],y_rest=r['y_rest']),lim,valid)
        return (*physics_cache[key],reused)
    preflight=[]
    try:
        oracle={}
        # Check all selected targets, without inspecting estimated-method scores.
        for pair,target in itertools.product(pairs,args.targets):
            try:
                u=planner(target,truth[pair],J=JERK,x0=X0,u_max=cap,**plan_options)
                r,lim,valid,_=shot(pair,u)
                item=dict(pair=pair,target_x=target,peak_speed=u,valid=valid,limits=lim)
                oracle[(pair,target)]=u
            except Exception as exc:
                item=dict(pair=pair,target_x=target,valid=False,error=f'{type(exc).__name__}: {exc}')
            preflight.append(item)
        write_json(args.out/'preflight.json',preflight)
        if any(not r['valid'] for r in preflight):
            raise RuntimeError('Feasibility preflight failed; see preflight.json. No cells silently excluded.')
        with (args.out/'results.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n')
            writer.writeheader()
            for pair in pairs:
                practice=[]
                practice_limits=[]
                for i in range(2):
                    r,p,lim,valid=executor(truth[pair],practice_peak)
                    practice_limits.append(dict(slide=i+1,valid=valid,limits=lim))
                    write_json(args.out/f'{pair}_practice_validity.json',practice_limits)
                    if not valid:
                        raise RuntimeError(f'{pair}: invalid practice rollout {i+1}')
                    practice.append((r,p))
                for condition in args.noise:
                    for seed in condition_seeds(condition,args.seeds):
                        rng=np.random.default_rng(np.random.SeedSequence([max(seed,0),int(pair[1:])]))
                        observations=[make_observation(r,p,rng,condition) for r,p in practice]
                        obs_id=f'{pair}_{condition}_seed{seed}'
                        write_json(args.out/f'{obs_id}_observations.json',[obs_record(o) for o in observations])
                        # Beliefs are updated once per pair/condition/seed/n, not per target.
                        for n in range(3):
                            posts={m:update_with_slides(g,lp,observations[:n],model='C') for m,lp in initials[pair].items()}
                            for target,method in itertools.product(args.targets,METHODS):
                                post=posts.get(method)
                                u=oracle[(pair,target)] if method=='oracle' else planner(target,post,J=JERK,x0=X0,u_max=cap)
                                if not math.isfinite(u) or not 0<=u<=cap+1e-8:
                                    raise ValueError('Planner violated peak-speed bounds')
                                r,lim,valid,reused=shot(pair,u)
                                dx,dy=r['x_rest']-target,r['y_rest']
                                row=dict(pair=pair,condition=condition,seed=seed,target_x=target,method=method,slides=n,
                                    mu_true=truth[pair],mu_median=truth[pair] if post is None else post['mu_median'],
                                    mu_low=truth[pair] if post is None else post['mu_q05'],
                                    mu_high=truth[pair] if post is None else post['mu_q95'],peak_speed=u,at_cap=u>=cap-1e-4,
                                    x_rest=r['x_rest'],y_rest=dy,error_x_mm=1000*dx,error_xy_mm=1000*math.hypot(dx,dy),
                                    q_limit_fraction=lim['q'],v_limit_fraction=lim['v'],ff_torque_limit_fraction=lim['tau'],
                                    valid=valid,reused_rollout=reused,accel_limit_fraction=lim.get('a',''))
                                writer.writerow(row);f.flush();status['rows_written']+=1
                                if not valid:
                                    status['invalid_rows']+=1
                                    raise RuntimeError('Invalid scored shot recorded; stopping, not dropping it from metrics')
                        print(f'{obs_id}: finished {len(args.targets)*12} scored rows',flush=True)
        if status['rows_written']!=config['expected_rows']:
            raise RuntimeError('Unexpected row count')
        status.update(status='complete',unique_scored_rollouts=len(physics_cache))
    except Exception as exc:
        status.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(args.out/'status.json',status)
    print(f"COMPLETE: {status['rows_written']} rows, {len(physics_cache)} unique scored rollouts; saved to {args.out}")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pair',choices=[*SURFACES,'all'],default='Y01')
    p.add_argument('--targets',nargs='+',default=['0.60','0.62','0.64'],help="world-x positions, or 'auto' (see the module docstring)")
    p.add_argument('--n-targets',type=int,default=3,help='how many targets --targets auto makes')
    p.add_argument('--layout',choices=['origin','side'],default=None)
    p.add_argument('--base-x',type=float,default=None);p.add_argument('--base-y',type=float,default=None)
    p.add_argument('--a-brake',type=float,default=None,help='brake cap in m/s^2 (default: the arm\'s 12)')
    p.add_argument('--check-accel',action='store_true')
    p.add_argument('--peak-cap',default=str(V3_CAP),help="m/s, or 'auto'")
    p.add_argument('--practice-peak',default=None,help="m/s, or 'auto'; default: the peak cap")
    p.add_argument('--clip-unreachable',action='store_true')
    p.add_argument('--noise',nargs='+',choices=list(NOISE),default=list(NOISE))
    p.add_argument('--seeds',nargs='+',type=int,default=[0])
    p.add_argument('--grid-points',type=int,default=2000)
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--out',type=Path,default=Path('arm_eval_v3'))
    a=p.parse_args()
    if a.grid_points<500 or any(s<0 for s in a.seeds): p.error('Use >=500 grid points and nonnegative seeds')
    for name,values in [('targets',a.targets),('noise',a.noise),('seeds',a.seeds)]:
        if len(values)!=len(set(values)): p.error(f'Duplicate {name}')
    if a.targets!=['auto']:
        try: a.targets=[float(t) for t in a.targets]
        except ValueError: p.error("Targets are world-x positions in m, or the single word 'auto'")
        if any(not math.isfinite(t) or t<=X0 for t in a.targets): p.error('Targets must be finite and beyond x0=.55')
    for name in ('peak_cap','practice_peak'):
        value=getattr(a,name)
        if value not in (None,'auto'):
            try: setattr(a,name,float(value))
            except ValueError: p.error(f"--{name.replace('_','-')} is a speed in m/s or 'auto'")
            if not 0<getattr(a,name)<=5: p.error('Speeds must be positive')
    if a.a_brake is not None and a.a_brake<=0: p.error('--a-brake must be positive')
    run(a)


if __name__=='__main__': main()
