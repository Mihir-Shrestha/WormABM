import argparse
import sys

import h5py
import numpy as np


def verify_worm_level(worm_path, eps=1e-12):
    with h5py.File(worm_path, "r") as f:
        required = ["worm_i", "on_patch", "cells_eaten_step", "cells_eaten_total"]
        missing = [k for k in required if k not in f]
        if missing:
            raise KeyError(f"Missing required worm datasets: {missing}")

        worm_i = np.asarray(f["worm_i"][:], dtype=int)
        on_patch = np.asarray(f["on_patch"][:]).astype(bool)
        eaten_step = np.asarray(f["cells_eaten_step"][:], dtype=float)
        eaten_total = np.asarray(f["cells_eaten_total"][:], dtype=float)

    offpatch_nonzero = int(np.sum((~on_patch) & (eaten_step > eps)))
    onpatch_nonpos = int(np.sum(on_patch & (eaten_step <= eps)))

    worms = np.unique(worm_i)
    nonmono_total_worms = 0
    sum_mismatch_worms = 0

    for w in worms:
        m = worm_i == w
        step_w = eaten_step[m]
        total_w = eaten_total[m]

        if np.any(np.diff(total_w) < -eps):
            nonmono_total_worms += 1

        if not np.isclose(total_w[-1], np.sum(step_w), atol=1e-9, rtol=1e-9):
            sum_mismatch_worms += 1

    result = {
        "rows": int(len(worm_i)),
        "worms": int(len(worms)),
        "on_patch_fraction": float(np.mean(on_patch)),
        "eaten_nonzero_fraction": float(np.mean(eaten_step > eps)),
        "violations_offpatch_nonzero": offpatch_nonzero,
        "violations_onpatch_nonpositive": onpatch_nonpos,
        "worms_nonmonotonic_total": nonmono_total_worms,
        "worms_total_step_mismatch": sum_mismatch_worms,
    }
    return result


def verify_global_consistency(worm_path, env_path, eps=1e-12):
    with h5py.File(worm_path, "r") as wf:
        worm_i = np.asarray(wf["worm_i"][:], dtype=int)
        eaten_step = np.asarray(wf["cells_eaten_step"][:], dtype=float)

    with h5py.File(env_path, "r") as ef:
        required = ["dB_feed_requested", "dB_feed"]
        missing = [k for k in required if k not in ef]
        if missing:
            raise KeyError(f"Missing required environment datasets: {missing}")

        dB_feed_requested = np.asarray(ef["dB_feed_requested"][:], dtype=float)
        dB_feed = np.asarray(ef["dB_feed"][:], dtype=float)

    n_worms = len(np.unique(worm_i))
    if n_worms == 0:
        raise ValueError("No worms found in worm history.")

    if len(eaten_step) % n_worms != 0:
        raise ValueError(
            "Worm history rows are not divisible by number of worms; cannot reshape by timestep."
        )

    n_steps = len(eaten_step) // n_worms
    step_sum_from_worms = eaten_step.reshape(n_steps, n_worms).sum(axis=1)

    if len(dB_feed_requested) != n_steps or len(dB_feed) != n_steps:
        raise ValueError(
            "Environment and worm step counts do not match. "
            f"worm_steps={n_steps}, env_requested={len(dB_feed_requested)}, env_feed={len(dB_feed)}"
        )

    mismatch_requested = int(np.sum(np.abs(step_sum_from_worms - dB_feed_requested) > 1e-9))
    invalid_clamp = int(np.sum(dB_feed - dB_feed_requested > eps))

    result = {
        "timesteps": int(n_steps),
        "requested_mismatch_steps": mismatch_requested,
        "clamp_violation_steps": invalid_clamp,
        "requested_sum_worms": float(np.sum(step_sum_from_worms)),
        "requested_sum_environment": float(np.sum(dB_feed_requested)),
        "applied_sum_environment": float(np.sum(dB_feed)),
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Verify worm feeding logic: worms eat only on patch and environment depletion "
            "matches per-worm requested consumption."
        )
    )
    parser.add_argument("worm_hist", help="Path to worm_hist.h5")
    parser.add_argument("--env-hist", default=None, help="Optional path to environment_hist.h5")
    args = parser.parse_args()

    ok = True

    print("WORM-LEVEL CHECKS")
    worm_result = verify_worm_level(args.worm_hist)
    for k, v in worm_result.items():
        print(f"  {k}: {v}")

    if worm_result["violations_offpatch_nonzero"] > 0:
        ok = False
    if worm_result["worms_nonmonotonic_total"] > 0:
        ok = False
    if worm_result["worms_total_step_mismatch"] > 0:
        ok = False

    if args.env_hist is not None:
        print("\nGLOBAL CONSISTENCY CHECKS")
        global_result = verify_global_consistency(args.worm_hist, args.env_hist)
        for k, v in global_result.items():
            print(f"  {k}: {v}")

        if global_result["requested_mismatch_steps"] > 0:
            ok = False
        if global_result["clamp_violation_steps"] > 0:
            ok = False

    if ok:
        print("\nPASS: Feeding logic checks succeeded.")
        sys.exit(0)

    print("\nFAIL: One or more feeding logic checks failed.")
    sys.exit(1)


if __name__ == "__main__":
    main()
