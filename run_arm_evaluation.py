"""Version 3 evaluation infrastructure: multiple targets and observation-noise levels.

No API calls. Uses existing published-pair cache. Positions come from Drake state,
not camera perception; release timing uses the ideal contact event. Dataset remains
three exploratory pairs. Old runners/results are untouched. Deterministic scored
rollouts with the exact same (pair, speed) are reused within a run.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from belief import theta_grid, logprior, posterior, loglik_lookup, lookup_observation, update_with_slides
from llm_prior import build_prompt, cached_response, parse_response
from demo_published_comparison import OBJECT, SURFACES, raw_llm_logprior
from run_arm_comparison_v2 import PRIOR_MEDIAN, PRIOR_LOG_SD, execute

ROOT = Path(__file__).resolve().parent
VERSION = 'multi_target_noise_v3'
METHODS = ('generic', 'lookup', 'llm_raw', 'oracle')
NOISE = {'low': .5, 'nominal': 1., 'high': 2., 'no_added': 0.}
FIELDS = ['pair','condition','seed','target_x','method','slides','mu_true','mu_median',
          'mu_low','mu_high','peak_speed','at_cap','x_rest','y_rest','error_x_mm',
          'error_xy_mm','q_limit_fraction','v_limit_fraction','ff_torque_limit_fraction',
          'valid','reused_rollout']


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


def checked_execute(mu,u):
    r,p,limits,valid = execute(mu,u)
    valid = bool(valid and all(math.isfinite(float(r[k])) for k in ('x_rest','y_rest'))
                 and all(math.isfinite(v) for v in limits.values()))
    return r,p,limits,valid


def run(args, executor=checked_execute, planner=None):
    pairs = list(SURFACES) if args.pair == 'all' else [args.pair]
    cache = ROOT/'llm_cache_published_pilot.json'
    with (ROOT/'ground_truth_published.csv').open(newline='') as f:
        truth = {r['pair_id']:float(r['mu_k_nominal']) for r in csv.DictReader(f)}
    g = theta_grid(args.grid_points,math.log(.005),math.log(5.))
    generic = logprior(g,math.log(PRIOR_MEDIAN),PRIOR_LOG_SD)
    initials = {}
    for pair in pairs:
        prompt = build_prompt(dict(description_object=OBJECT,description_surface=SURFACES[pair]))
        raw = cached_response(prompt,cache)
        if raw is None:
            raise ValueError(f'Cache miss for {pair}; no request made')
        table = lookup_observation(pair,str(ROOT/'friction_lookup.csv'))
        initials[pair] = dict(generic=generic,
            lookup=generic.copy() if table is None else generic+loglik_lookup(g,*table),
            llm_raw=raw_llm_logprior(g,parse_response(raw)))
    config = dict(version=VERSION,status='infrastructure_for_exploratory_pairs',
                  pairs=pairs,conditions=args.noise,seeds=args.seeds,targets=args.targets,
                  x0=.55,peak_cap=.45,practice_peak=.45,jerk=100.,fps=60,
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
    config['sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}
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
                u=planner(target,truth[pair],J=100,x0=.55,u_max=.45)
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
                    r,p,lim,valid=executor(truth[pair],.45)
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
                                u=oracle[(pair,target)] if method=='oracle' else planner(target,post,J=100,x0=.55,u_max=.45)
                                if not math.isfinite(u) or not 0<=u<=.45+1e-8:
                                    raise ValueError('Planner violated peak-speed bounds')
                                r,lim,valid,reused=shot(pair,u)
                                dx,dy=r['x_rest']-target,r['y_rest']
                                row=dict(pair=pair,condition=condition,seed=seed,target_x=target,method=method,slides=n,
                                    mu_true=truth[pair],mu_median=truth[pair] if post is None else post['mu_median'],
                                    mu_low=truth[pair] if post is None else post['mu_q05'],
                                    mu_high=truth[pair] if post is None else post['mu_q95'],peak_speed=u,at_cap=u>=.45-1e-4,
                                    x_rest=r['x_rest'],y_rest=dy,error_x_mm=1000*dx,error_xy_mm=1000*math.hypot(dx,dy),
                                    q_limit_fraction=lim['q'],v_limit_fraction=lim['v'],ff_torque_limit_fraction=lim['tau'],
                                    valid=valid,reused_rollout=reused)
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
    p.add_argument('--targets',nargs='+',type=float,default=[.60,.62,.64])
    p.add_argument('--noise',nargs='+',choices=list(NOISE),default=list(NOISE))
    p.add_argument('--seeds',nargs='+',type=int,default=[0])
    p.add_argument('--grid-points',type=int,default=2000)
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--out',type=Path,default=Path('arm_eval_v3'))
    a=p.parse_args()
    if a.grid_points<500 or any(s<0 for s in a.seeds): p.error('Use >=500 grid points and nonnegative seeds')
    for name,values in [('targets',a.targets),('noise',a.noise),('seeds',a.seeds)]:
        if len(values)!=len(set(values)): p.error(f'Duplicate {name}')
    if any(not math.isfinite(t) or t<=.55 for t in a.targets): p.error('Targets must be finite and beyond x0=.55')
    run(a)


if __name__=='__main__': main()
