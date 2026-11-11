"""Offline pilot: real cached LLM responses, published friction, synthetic slides.

Run beside the existing project modules:
    python demo_published_comparison.py

No API calls. Uses llm_cache_published_pilot.json created by llm_prior.py.
This is not the calibrated LLM method in the repository specification, nor a
final evaluation. Lookup rows are used as-is, including their source caveats.
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np

from belief import theta_grid, logprior, loglik_lookup, lookup_observation, update_with_slides
from decide import Release, choose_speed, distance
from llm_prior import build_prompt, cached_response, parse_response
from observe import CAMERA_FPS, simulate_obs
from physics import release_speed, release_push_distance

ROOT = Path(__file__).resolve().parent
SURFACES = {"Y01": "An ABS plastic sheet", "Y02": "An acetal (Delrin) sheet",
            "Y03": "A plywood sheet"}
OBJECT = "A block with a stainless-steel contacting face"
Z90 = 1.6448536269514722


def raw_llm_logprior(grid, estimate):
    """Median-preserving lognormal approximation, not empirical calibration.

    A lognormal cannot generally match all three reported quantiles. Preserve
    median and match the reported log-interval WIDTH, averaging the two log-side
    scales. Grid truncation also changes quantiles. Print fitted intervals rather
    than labeling them as the LLM's original interval. No generic prior is added.
    """
    if not estimate['low'] < estimate['median'] < estimate['high']:
        raise ValueError("Pilot requires low < median < high")
    sd = math.log(estimate['high'] / estimate['low']) / (2 * Z90)
    return logprior(grid, math.log(estimate['median']), sd)


def run(cache_path, seed=7, verbose=True):
    with (ROOT / 'ground_truth_published.csv').open(newline='') as f:
        worlds = {r['pair_id']: float(r['mu_k_nominal']) for r in csv.DictReader(f)}
    # Load every response before doing work. Cache misses NEVER trigger API calls.
    estimates = {}
    for pair, surface in SURFACES.items():
        prompt = build_prompt({'description_object': OBJECT, 'description_surface': surface})
        raw = cached_response(prompt, cache_path)
        if raw is None:
            raise ValueError(f"Missing matching cache entry for {pair}. Check the cache path and model settings.")
        estimates[pair] = parse_response(raw)

    grid = theta_grid()
    rel, target, u_max = Release(J=100.0), 0.30, 2.0
    results = []
    if verbose:
        print('OFFLINE PILOT -- synthetic observations, analytic scores; no API calls or Drake')
        print('Generic ln(mu) prior: N(ln(0.30), 0.45^2), retained from demos, NOT final proposal settings')
        print('Raw LLM: median-preserving lognormal approximation; no calibration or generic-prior multiplication')
        print('Target=0.30 m from release; J=100; illustrative target peak cap=2 m/s')
        print('Shared practice: peak=1 m/s, x0=0.55 m; model C, 60 Hz')
        print(f'Noise SD: frame=2 mm, rest=0.5 mm, FK=0.5 mm; seed={seed}')
        print('Lookup entries used as-is; their source-verification caveats remain.')
    for index, pair in enumerate(SURFACES):
        true_mu = worlds[pair]  # World/scoring only; never passed to estimation.
        estimate = estimates[pair]
        generic = logprior(grid, math.log(0.3), 0.45)
        lookup = lookup_observation(pair, str(ROOT / 'friction_lookup.csv'))
        methods = {'generic': generic,
                   'lookup': generic.copy() if lookup is None else generic + loglik_lookup(grid, *lookup),
                   'llm_raw': raw_llm_logprior(grid, estimate)}
        rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
        observations = []
        for _ in range(2):
            vr = float(release_speed(1.0, true_mu, rel.J))
            xr = 0.55 + float(release_push_distance(1.0, true_mu, rel.J))
            obs = simulate_obs(xr, vr, true_mu, CAMERA_FPS, .002, .0005, .0005, rng)
            obs.truth.clear()
            observations.append(obs)
        # Oracle knows true friction; this exception is intentional and explicit.
        oracle_u = min(u_max, math.sqrt(2 * true_mu * rel.g * target)
                       + (true_mu * rel.g)**2 / (2 * rel.J))
        oracle_error = 100 * (float(distance(oracle_u, true_mu, rel)) - target)
        if verbose:
            print(f"\n{pair}: true mu={true_mu:.3f}; LLM median={estimate['median']:.3f}, "
                  f"original interval=[{estimate['low']:.3f}, {estimate['high']:.3f}]")
            print('method    slides  mu median   fitted/posterior 90%    peak m/s   error cm')
        for n in range(3):
            for name, logp0 in methods.items():
                post = update_with_slides(grid, logp0, observations[:n], model='C')
                choice = choose_speed(post, rel, target, u_max=u_max)
                error = 100 * (float(distance(choice['u'], true_mu, rel)) - target)
                results.append(dict(pair=pair, method=name, slides=n, posterior=post,
                                    speed=choice['u'], error_cm=error))
                if verbose:
                    print(f"{name:9s} {n:6d}  {post['mu_median']:9.4f}  "
                          f"[{post['mu_q05']:.4f}, {post['mu_q95']:.4f}]  "
                          f"{choice['u']:10.4f}  {error:+9.3f}")
            results.append(dict(pair=pair, method='oracle', slides=n, posterior=None,
                                speed=oracle_u, error_cm=oracle_error))
            if verbose:
                print(f"{'oracle':9s} {n:6d}  {true_mu:9.4f}  {'known friction':18s}  "
                      f"{oracle_u:10.4f}  {oracle_error:+9.3f}")
    if verbose:
        print('\nNegative error = undershoot; positive error = overshoot.')
        print('Three pairs and one seed are integration checks, not a general model ranking.')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT / 'llm_cache_published_pilot.json')
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    run(args.cache, args.seed)
