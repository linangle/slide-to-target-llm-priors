"""Compare generic and lookup beliefs on identical synthetic practice slides.

Run alongside belief.py, decide.py, observe.py and friction_lookup.csv:
    python evaluation/demo_compare_priors.py
    python evaluation/demo_compare_priors.py --pair M06

The pair ID selects a REAL lookup row, but true_mu is an ARBITRARY synthetic
world value, not a measurement of that pair. This is an integration check only.
No API calls, Drake simulation, or output files. Target distance is from release.
"""
import argparse
import math
from pathlib import Path
from paths import DATA

import numpy as np

from belief import (theta_grid, logprior, loglik_lookup, lookup_observation,
                    update_with_slides)
from decide import Release, choose_speed, distance
from observe import CAMERA_FPS, simulate_obs
from physics import release_speed, release_push_distance


def compare(pair="V03", true_mu=0.40, seed=7, verbose=True):
    if not 0.05 < true_mu < 1.0:
        raise ValueError("Choose synthetic true_mu strictly between 0.05 and 1.0.")
    grid = theta_grid()
    generic = logprior(grid, math.log(0.30), 0.45)
    table_path = DATA / "friction_lookup.csv"
    lookup = lookup_observation(pair, str(table_path))
    initial = {
        "generic": generic,
        "lookup": generic.copy() if lookup is None else generic + loglik_lookup(grid, *lookup),
    }
    rel = Release(J=100.0)
    target = 0.30
    rng = np.random.default_rng(seed)

    # Generate two observations ONCE. Both methods see the same action speeds,
    # frame noise, rest-pose noise and FK noise. No method adapts practice here.
    observations = []
    for _ in range(2):
        practice_peak = 1.0
        v_release = float(release_speed(practice_peak, true_mu, rel.J))
        if v_release <= 0:
            raise ValueError("Synthetic friction is too high for the practice action.")
        x_release = 0.55 + float(release_push_distance(practice_peak, true_mu, rel.J))
        obs = simulate_obs(x_release, v_release, true_mu, CAMERA_FPS,
                           0.002, 0.0005, 0.0005, rng)
        obs.truth.clear()
        observations.append(obs)

    if verbose:
        print("SYNTHETIC COMPARISON -- not measured-pair or robot results")
        print(f"Lookup row: {pair}; arbitrary synthetic truth: mu={true_mu:.3f}; seed={seed}")
        print(f"Lookup (mu, log-SD): {lookup}; None means generic-prior fallback")
        print("Generic prior: ln(mu) ~ N(ln(0.30), 0.45^2)")
        print("Shared practice: fixed 1.0 m/s peak, J=100, model C, 60 Hz")
        print("Noise SD: frames 2 mm, rest 0.5 mm, FK 0.5 mm")
        print("Target: 30 cm from release; 2 m/s cap is illustrative, not an arm limit")
        print("\nmethod   slides  mu median    90% interval     peak m/s  error cm  capped")

    results = []
    for n in range(3):
        for method, logp0 in initial.items():
            # Start from original method belief each time: no double-counting.
            post = update_with_slides(grid, logp0, observations[:n], model="C")
            choice = choose_speed(post, rel, target, u_max=2.0)
            # Truth is used only by the synthetic world/scorer, never inference.
            error_cm = 100 * (float(distance(choice["u"], true_mu, rel)) - target)
            results.append(dict(method=method, slides=n, posterior=post,
                                choice=choice, error_cm=error_cm))
            if verbose:
                print(f"{method:8s} {n:6d}  {post['mu_median']:9.4f}  "
                      f"[{post['mu_q05']:.4f}, {post['mu_q95']:.4f}]  "
                      f"{choice['u']:8.4f}  {error_cm:+8.3f}  {choice['clipped']}")
    if verbose:
        print("\nNegative error = undershoot. Errors use the analytic synthetic world.")
        print("No real LLM method yet. One synthetic case cannot rank methods.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", default="V03", help="Lookup row ID; not a source of truth")
    parser.add_argument("--true-mu", type=float, default=0.40, help="Synthetic world friction")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    compare(args.pair, args.true_mu, args.seed)
