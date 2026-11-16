"""Version 2 arm evaluation candidate: proposal-width generic prior.

No API calls. First run --check-only, then --pair Y01.
The original pilot runner and results remain unchanged.
Prior: lognormal matching 0.08 and 1.1 as untruncated 5th/95th percentiles;
median sqrt(0.08*1.1) = 0.29665, approximately the proposal's 0.3.
All methods share an expanded [0.005, 5] friction grid.
This is a configuration candidate, not a frozen final experiment.

Practice and scored shots run in Drake. Observations are sampled simulator
positions plus Gaussian noise, NOT camera perception. Release timing uses the
simulator contact-force event (ideal timing assumption). The generic prior and
raw-LLM approximation remain pilot settings. Single-process execution only.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PRIOR_MEDIAN = math.sqrt(.08 * 1.1)
PRIOR_LOG_SD = math.log(1.1/.08)/(2*1.6448536269514722)


def make_observation(result, pusher, rng):
    from observe import Obs
    from physics import OBJECT_SIZE
    tr = result['traj']
    t0 = result['t_force_zero']
    # Observe moving frames only; onset is contact-based, not computed from mu.
    moving = np.flatnonzero((tr['t'] > t0) & (np.abs(tr['vx']) > 0.005))
    if not len(moving):
        raise ValueError('No measurable free slide after contact separation')
    ts = np.arange(1, int((tr['t'][moving[-1]] - t0) * 60) + 1) / 60.0
    if len(ts) < 2:
        raise ValueError('Fewer than two free-slide frames; review practice settings')
    positions = np.interp(t0 + ts, tr['t'], tr['x']) + rng.normal(0, .002, len(ts))
    q0 = float(np.interp(t0, tr['t'], tr['q']))
    fk = float(pusher.face_x(q0) + OBJECT_SIZE[0]/2) + rng.normal(0, .0005)
    rest = result['x_rest'] + rng.normal(0, .0005)
    return Obs(ts, positions, rest, fk, .002, .0005, .0005)


def execute(mu, speed):
    from arm import ArmPusher, joint_limit_report
    from push import push_slide
    pusher = ArmPusher()
    result = push_slide(mu, speed, 100.0, x0=.55, pusher=pusher, keep_traj=True)
    limits = joint_limit_report(pusher.traj)
    maxima = {k: float(np.max(limits[k])) for k in ('q', 'v', 'tau')}
    valid = all(v <= 1.0 for v in maxima.values()) and bool(result['separated'])
    # These limits describe the planned trajectory and feed-forward torque,
    # not a full certificate of actual actuator torque including feedback.
    return result, pusher, maxima, valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair', choices=['Y01', 'Y02', 'Y03', 'all'], default='Y01')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--check-only', action='store_true', help='Validate cache and print priors; no Drake or output files')
    parser.add_argument('--grid-points', type=int, default=2000)
    parser.add_argument('--out', type=Path, default=Path('arm_comparison_v2'))
    args = parser.parse_args()
    from belief import theta_grid, logprior, loglik_lookup, lookup_observation, update_with_slides, posterior
    from llm_prior import build_prompt, cached_response, parse_response
    from demo_published_comparison import OBJECT, SURFACES, raw_llm_logprior

    pairs = list(SURFACES) if args.pair == 'all' else [args.pair]
    cache_path = ROOT / 'llm_cache_published_pilot.json'
    with (ROOT / 'ground_truth_published.csv').open(newline='') as f:
        truth = {r['pair_id']: float(r['mu_k_nominal']) for r in csv.DictReader(f)}
    estimates = {}
    for pair in pairs:
        prompt = build_prompt({'description_object': OBJECT, 'description_surface': SURFACES[pair]})
        raw = cached_response(prompt, cache_path)
        if raw is None:
            raise ValueError(f'Missing cached response: {pair}. No API request was made.')
        estimates[pair] = parse_response(raw)
    if args.grid_points < 500:
        raise ValueError('Use at least 500 grid points')
    grid = theta_grid(n=args.grid_points, lo=math.log(.005), hi=math.log(5.0))
    prior = logprior(grid, math.log(PRIOR_MEDIAN), PRIOR_LOG_SD)
    p0 = posterior(grid, prior)
    print(f"V2 default: median={p0['mu_median']:.5f}, 90%=[{p0['mu_q05']:.5f}, {p0['mu_q95']:.5f}]")
    print(f"Common grid: {args.grid_points} points, mu in [0.005, 5.0]")
    if args.check_only:
        print('CACHE AND PRIOR CHECK PASSED; no API requests, simulations, or output files.')
        return
    from arm import plan_peak_speed
    # Fail before simulations if an output folder already exists.
    args.out.mkdir(parents=True, exist_ok=False)
    config = dict(status='v2_candidate_not_final', version='proposal_width_v2', pairs=pairs, seed=args.seed, target_x=.64,
                  x0=.55, peak_cap=.45, practice_peak=.45, jerk=100, fps=60,
                  frame_sd=.002, rest_sd=.0005, fk_sd=.0005,
                  generic_log_median=math.log(PRIOR_MEDIAN), generic_log_sd=PRIOR_LOG_SD,
                  grid_mu_min=.005, grid_mu_max=5.0, grid_points=args.grid_points,
                  llm='raw median-preserving lognormal approximation',
                  sensor='noisy simulator positions, ideal contact-event release time',
                  limits='planned q, qdot and feed-forward torque; not total actuator torque')
    files = ['run_arm_comparison_v2.py','demo_published_comparison.py','llm_prior.py',
             'belief.py','observe.py','arm.py','push.py','physics.py','scene.py',
             'drake_config.py','sim_config.yaml','ground_truth_published.csv',
             'friction_lookup.csv','llm_cache_published_pilot.json']
    config['sha256'] = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}
    (args.out/'settings.json').write_text(json.dumps(config, indent=2)+'\n')
    rows = []
    print('DRAKE ARM V2 CANDIDATE: x target=.64 m; peak cap=.45 m/s; no API calls', flush=True)
    fields = ['pair','method','slides','seed','mu_true','mu_median','mu_low','mu_high',
              'peak_speed','at_cap','x_rest','y_rest','error_x_mm','error_xy_mm',
              'q_limit_fraction','v_limit_fraction','ff_torque_limit_fraction','valid']
    with (args.out/'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for pair in pairs:
            mu = truth[pair]
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, int(pair[1:])]))
            generic = prior.copy()
            table = lookup_observation(pair, str(ROOT/'friction_lookup.csv'))
            initial = dict(generic=generic,
                           lookup=generic if table is None else generic+loglik_lookup(grid,*table),
                           llm_raw=raw_llm_logprior(grid,estimates[pair]))
            # Preflight the known-friction oracle against the same speed cap.
            oracle_u = plan_peak_speed(.64, mu, J=100, x0=.55, u_max=.45)
            observations = []
            saved_obs = []
            for n in range(3):
                for method in ['generic','lookup','llm_raw','oracle']:
                    post = None if method == 'oracle' else update_with_slides(grid, initial[method], observations, model='C')
                    u = oracle_u if post is None else plan_peak_speed(.64, post, J=100, x0=.55, u_max=.45)
                    result, _, limits, valid = execute(mu, u)
                    dx, dy = result['x_rest']-.64, result['y_rest']
                    row = dict(pair=pair, method=method, slides=n, seed=args.seed, mu_true=mu,
                               mu_median=mu if post is None else post['mu_median'],
                               mu_low=mu if post is None else post['mu_q05'],
                               mu_high=mu if post is None else post['mu_q95'],
                               peak_speed=u, at_cap=u >= .45-1e-4,
                               x_rest=result['x_rest'], y_rest=dy, error_x_mm=1000*dx,
                               error_xy_mm=1000*math.hypot(dx,dy), q_limit_fraction=limits['q'],
                               v_limit_fraction=limits['v'], ff_torque_limit_fraction=limits['tau'], valid=valid)
                    writer.writerow(row); f.flush(); rows.append(row)
                    print(f"{pair} {method:7s} n={n} u={u:.4f} miss={1000*dx:+.2f} mm valid={valid}", flush=True)
                    if not valid:
                        raise RuntimeError('Invalid scored rollout recorded; stopped. Inspect results.csv.')
                if n < 2:
                    result, pusher, limits, valid = execute(mu, .45)
                    if not valid:
                        raise RuntimeError(f'Invalid practice rollout: {limits}')
                    obs = make_observation(result, pusher, rng)
                    observations.append(obs)
                    saved_obs.append(dict(t=obs.t.tolist(), x=obs.x.tolist(), x_rest=obs.x_rest,
                                          x_fk=obs.x_fk, sigma_f=obs.sigma_f, sigma_rest=obs.sigma_rest,
                                          sigma_fk=obs.sigma_fk, planned_limits=limits))
                    (args.out/f'{pair}_observations.json').write_text(json.dumps(saved_obs,indent=2)+'\n')
                    print(f'  Shared practice {n+1}: {len(obs.t)} noisy frames', flush=True)
    print(f'Saved {len(rows)} scored shots to {args.out}/results.csv')
    print('Pilot settings remain provisional. No camera rendering or calibration was performed.')


if __name__ == '__main__':
    main()
