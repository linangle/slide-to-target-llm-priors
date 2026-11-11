"""Synthetic single-pair integration demo; run from the project environment.

Uses existing inference, observation, and decision functions. No Drake simulation,
LLM calls, measured data, or files are produced. These are plumbing checks, not
project evaluation results. Distances are measured from release, not table origin.
"""
import math

import numpy as np

from belief import theta_grid, logprior, update_with_slides
from decide import Release, choose_speed, distance, practice_speed, slide_room
from observe import CAMERA_FPS, simulate_obs
from physics import release_push_distance, release_speed


def run_demo(seed=7):
    # Synthetic world: truth is used ONLY to generate measurements and score actions.
    true_mu = 0.40
    x_start = 0.55
    target_distance = 0.30
    rel = Release(J=100.0)
    rng = np.random.default_rng(seed)

    # Illustrative generic prior, deliberately different from the synthetic truth.
    grid = theta_grid()
    initial_logprior = logprior(grid, math.log(0.30), 0.45)
    observations = []
    results = []

    print("SYNTHETIC DEMO -- no robot simulation or LLM calls")
    print(f"True mu={true_mu:.2f}; target={target_distance:.2f} m from release; seed={seed}")
    print("Prior: ln(mu) ~ N(ln(0.30), 0.45^2); model C; camera 60 Hz")
    print("Noise SD: frames 2 mm, rest 0.5 mm, FK release point 0.5 mm")
    print("Target-shot speed cap: 2 m/s (illustrative, not a validated arm limit)")
    print("\nslides  mu median       90% interval      peak m/s   distance m   error cm")

    for n in range(3):
        # Rebuild from the ORIGINAL prior plus each observation exactly once.
        # Never pass true_mu (or obs.truth) to inference or action selection.
        post = update_with_slides(grid, initial_logprior, observations, model="C")
        choice = choose_speed(post, rel, target_distance, u_max=2.0)
        actual_distance = float(distance(choice["u"], true_mu, rel))
        error = actual_distance - target_distance
        results.append(dict(slides=n, posterior=post, choice=choice,
                            distance=actual_distance, error=error))
        print(f"{n:6d}  {post['mu_median']:9.4f}  "
              f"[{post['mu_q05']:.4f}, {post['mu_q95']:.4f}]  "
              f"{choice['u']:10.4f}  {actual_distance:11.4f}  {100*error:+9.3f}")

        if n < 2:
            # Practice is a separate action; the scored target shot above does
            # not generate observations or update the belief.
            u_practice = practice_speed(1.2, slide_room(x_start), post, rel)
            v_release = float(release_speed(u_practice, true_mu, rel.J))
            x_release = x_start + float(release_push_distance(u_practice, true_mu, rel.J))
            obs = simulate_obs(x_release, v_release, true_mu, CAMERA_FPS,
                               0.002, 0.0005, 0.0005, rng)
            obs.truth.clear()  # Inference receives measurements, not truth metadata.
            observations.append(obs)
            print(f"        Practice {n+1}: peak={u_practice:.4f} m/s; "
                  f"{len(obs.t)} noisy position frames")

    print("\nNegative error = undershoot; positive error = overshoot.")
    print("Distance/error use the analytic model at the synthetic true friction.")
    print("This one seed does not establish performance or guarantee improvement.")
    return results


if __name__ == "__main__":
    run_demo()
