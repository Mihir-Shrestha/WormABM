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
        t_hist = np.asarray(f["t"][:], dtype=int) if "t" in f else None

    offpatch_nonzero = int(np.sum((~on_patch) & (eaten_step > eps)))
    onpatch_nonpos = int(np.sum(on_patch & (eaten_step <= eps)))

    worms = np.unique(worm_i)
    nonmono_total_worms = 0
    sum_mismatch_worms = 0
    sum_check_worms = 0
    sum_check_skipped_worms = 0

    for w in worms:
        m = worm_i == w
        step_w = eaten_step[m]
        total_w = eaten_total[m]

        if np.any(np.diff(total_w) < -eps):
            nonmono_total_worms += 1

        # The exact identity total[-1] == sum(step) is valid only when every
        # simulation step is recorded. With sparse logging, step_w is sampled.
        if t_hist is None:
            should_check_sum = True
        else:
            t_w = t_hist[m]
            should_check_sum = bool(len(t_w) <= 1 or np.all(np.diff(t_w) == 1))

        if should_check_sum:
            sum_check_worms += 1
            if not np.isclose(total_w[-1], np.sum(step_w), atol=1e-9, rtol=1e-9):
                sum_mismatch_worms += 1
        else:
            sum_check_skipped_worms += 1

    result = {
        "rows": int(len(worm_i)),
        "worms": int(len(worms)),
        "on_patch_fraction": float(np.mean(on_patch)),
        "eaten_nonzero_fraction": float(np.mean(eaten_step > eps)),
        "violations_offpatch_nonzero": offpatch_nonzero,
        "violations_onpatch_nonpositive": onpatch_nonpos,
        "worms_nonmonotonic_total": nonmono_total_worms,
        "worms_total_step_mismatch": sum_mismatch_worms,
        "worms_total_step_mismatch_checked": sum_check_worms,
        "worms_total_step_mismatch_skipped": sum_check_skipped_worms,
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
        F = np.asarray(ef["F"][:] if "F" in ef else np.ones_like(dB_feed_requested), dtype=float)

    n_worms = len(np.unique(worm_i))
    if n_worms == 0:
        raise ValueError("No worms found in worm history.")

    if len(eaten_step) % n_worms != 0:
        raise ValueError(
            "Worm history rows are not divisible by number of worms; cannot reshape by timestep."
        )

    n_steps = len(eaten_step) // n_worms
    step_sum_from_worms = eaten_step.reshape(n_steps, n_worms).sum(axis=1)

    n_common = min(n_steps, len(dB_feed_requested), len(dB_feed), len(F))
    if n_common == 0:
        raise ValueError("No overlapping timesteps between worm and environment history.")

    worm_requested_raw = step_sum_from_worms[:n_common]
    worm_requested_scaled = worm_requested_raw * F[:n_common]
    env_requested = dB_feed_requested[:n_common]
    env_applied = dB_feed[:n_common]

    mismatch_requested = int(np.sum(np.abs(worm_requested_scaled - env_requested) > 1e-9))
    invalid_clamp = int(np.sum(env_applied - env_requested > eps))
    negative_applied = int(np.sum(env_applied < -eps))

    # Legacy comparison is still helpful to spot accidental F=1 assumptions.
    mismatch_requested_legacy = int(np.sum(np.abs(worm_requested_raw - env_requested) > 1e-9))

    result = {
        "timesteps_compared": int(n_common),
        "timesteps_worm": int(n_steps),
        "timesteps_env_requested": int(len(dB_feed_requested)),
        "timesteps_env_feed": int(len(dB_feed)),
        "requested_mismatch_steps": mismatch_requested,
        "requested_mismatch_steps_legacy_raw": mismatch_requested_legacy,
        "clamp_violation_steps": invalid_clamp,
        "negative_applied_steps": negative_applied,
        "requested_sum_worms_raw": float(np.sum(worm_requested_raw)),
        "requested_sum_worms_scaled_by_F": float(np.sum(worm_requested_scaled)),
        "requested_sum_environment": float(np.sum(env_requested)),
        "applied_sum_environment": float(np.sum(env_applied)),
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
        if global_result["negative_applied_steps"] > 0:
            ok = False

    if ok:
        print("\nPASS: Feeding logic checks succeeded.")
        sys.exit(0)

    print("\nFAIL: One or more feeding logic checks failed.")
    sys.exit(1)


if __name__ == "__main__":
    main()
