import sys
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_feeding_diagnostics.py path/to/environment_hist.h5 [worm_hist.h5]")
        sys.exit(1)

    env_path = sys.argv[1]
    dt = 0.01

    if len(sys.argv) >= 3:
        worm_path = sys.argv[2]
    else:
        worm_path = str(Path(env_path).with_name("worm_hist.h5"))

    with h5py.File(env_path, "r") as f:
        B_before = f["B_before"][:]
        B_after = f["B_after"][:]
        dB_feed = f["dB_feed"][:]
        dB_feed_requested = f["dB_feed_requested"][:] if "dB_feed_requested" in f else dB_feed.copy()

    worm_stats_available = False
    worm_on_frac = None
    worm_step_sum = None
    if Path(worm_path).exists():
        with h5py.File(worm_path, "r") as wf:
            has_required = all(k in wf for k in ["worm_i", "on_patch", "cells_eaten_step"])
            if has_required:
                worm_i = np.asarray(wf["worm_i"][:], dtype=int)
                on_patch = np.asarray(wf["on_patch"][:]).astype(bool)
                cells_eaten_step = np.asarray(wf["cells_eaten_step"][:], dtype=float)

                n_worms = len(np.unique(worm_i))
                if n_worms > 0 and len(cells_eaten_step) % n_worms == 0:
                    n_steps = len(cells_eaten_step) // n_worms
                    worm_on_frac = on_patch.reshape(n_steps, n_worms).mean(axis=1)
                    worm_step_sum = cells_eaten_step.reshape(n_steps, n_worms).sum(axis=1)
                    worm_stats_available = True

    # --- Resample to one point per minute ---
    steps_per_minute = int(round(1.0 / dt))  # = 100 for dt = 0.01
    idx = np.arange(0, len(B_before), steps_per_minute)

    t_min = idx * dt  # time in minutes at those sample points
    B_before_min = B_before[idx]
    B_after_min = B_after[idx]
    dB_feed_min = dB_feed[idx]
    dB_feed_requested_min = dB_feed_requested[idx]

    # Cumulative bacteria removed (computed on full resolution first)
    cum_removed = np.cumsum(dB_feed)
    cum_removed_min = cum_removed[idx]

    if worm_stats_available:
        # Align lengths defensively in case histories were truncated differently.
        n_common = min(len(worm_on_frac), len(dB_feed), len(dB_feed_requested))
        worm_on_frac = worm_on_frac[:n_common]
        worm_step_sum = worm_step_sum[:n_common]
        dB_feed = dB_feed[:n_common]
        dB_feed_requested = dB_feed_requested[:n_common]

        idx_w = np.arange(0, n_common, steps_per_minute)
        t_w_min = idx_w * dt
        worm_on_frac_min = worm_on_frac[idx_w]
        worm_step_sum_min = worm_step_sum[idx_w]
        dB_feed_min = dB_feed[idx_w]
        dB_feed_requested_min = dB_feed_requested[idx_w]

        mismatch = np.abs(worm_step_sum - dB_feed_requested)
        print(f"[diag] worm vs requested max abs mismatch per step: {mismatch.max():.3e}")
        print(f"[diag] total requested from worms: {worm_step_sum.sum():.3f}")
        print(f"[diag] total requested in env:   {dB_feed_requested.sum():.3f}")
        print(f"[diag] total applied in env:     {dB_feed.sum():.3f}")
    else:
        print("[diag] worm verification curves unavailable (missing worm_hist datasets or shape mismatch).")
        print(f"[diag] looked for worm history at: {worm_path}")

    # Use the same time base for the environment-only panels.
    t_env_min = idx * dt

    plt.figure(figsize=(11, 11))

    # ---- Top: total bacteria before vs after feeding (per minute) ----
    plt.subplot(4, 1, 1)
    plt.plot(t_env_min, B_before_min, label="B_before (per minute)")
    plt.plot(t_env_min, B_after_min, label="B_after (per minute)")
    plt.ylabel("Total bacteria B")
    plt.title("Total bacteria before vs after feeding (sampled each minute)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # ---- Bottom: cumulative bacteria removed over time ----
    plt.subplot(4, 1, 2)
    plt.plot(t_env_min, cum_removed_min, label="Cumulative bacteria removed")
    plt.ylabel("Cumulative ΔB")
    plt.title("Cumulative feeding over time")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # ---- Added: fraction of worms on patch ----
    plt.subplot(4, 1, 3)
    if worm_stats_available:
        plt.plot(t_w_min, worm_on_frac_min, label="Fraction worms on patch")
        plt.ylim(-0.02, 1.02)
    else:
        plt.text(0.5, 0.5, "worm verification data unavailable", ha="center", va="center")
    plt.ylabel("On-patch fraction")
    plt.title("Worm occupancy of patch")
    plt.legend(loc="best") if worm_stats_available else None
    plt.grid(True, alpha=0.3)

    # ---- Added: per-step consistency overlay ----
    plt.subplot(4, 1, 4)
    if worm_stats_available:
        plt.plot(t_w_min, worm_step_sum_min, label="Σ worms cells_eaten_step", linewidth=1.7)
        plt.plot(t_w_min, dB_feed_requested_min, "--", label="Environment dB_feed_requested", linewidth=1.3)
        plt.plot(t_w_min, dB_feed_min, ":", label="Environment dB_feed (applied)", linewidth=1.8)
    else:
        plt.plot(t_env_min, dB_feed_requested_min, "--", label="Environment dB_feed_requested", linewidth=1.3)
        plt.plot(t_env_min, dB_feed_min, ":", label="Environment dB_feed (applied)", linewidth=1.8)
    plt.xlabel("time (minutes)")
    plt.ylabel("Cells / step")
    plt.title("Per-step feeding consistency")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()