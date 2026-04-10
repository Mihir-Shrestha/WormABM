import sys
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt


def _read_first_available(h5_obj, keys, default=None):
    """Return the first available dataset among keys."""
    for key in keys:
        if key in h5_obj:
            return np.asarray(h5_obj[key][:])
    return default


def _aggregate_per_minute(values, dt, agg="sum"):
    """Aggregate per-step values into one-minute block sums or means."""
    steps_per_minute = max(int(round(1.0 / dt)), 1)
    n_full = (len(values) // steps_per_minute) * steps_per_minute
    if n_full == 0:
        return np.array([]), np.array([])

    reshaped = values[:n_full].reshape(-1, steps_per_minute)
    if agg == "mean":
        block = reshaped.mean(axis=1)
    else:
        block = reshaped.sum(axis=1)
    t_min = np.arange(len(block), dtype=float)
    return t_min, block


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
        dB_feed = np.asarray(f["dB_feed"][:], dtype=float)
        dB_feed_requested = np.asarray(
            f["dB_feed_requested"][:] if "dB_feed_requested" in f else dB_feed.copy(),
            dtype=float,
        )
        B_deposited = np.asarray(
            f["B_deposited"][:] if "B_deposited" in f else np.zeros_like(dB_feed),
            dtype=float,
        )

    worm_stats_available = False
    dep_stats_available = False

    worm_on_frac = None
    surface_drop_step = None
    gut_drop_step = None

    if Path(worm_path).exists():
        with h5py.File(worm_path, "r") as wf:
            has_required = all(k in wf for k in ["worm_i", "on_patch", "cells_eaten_step"])
            if has_required:
                worm_i = np.asarray(wf["worm_i"][:], dtype=int)
                on_patch = np.asarray(wf["on_patch"][:]).astype(bool)
                cells_eaten_step = np.asarray(wf["cells_eaten_step"][:], dtype=float)

                # Backward compatibility: older runs may still use poop_* names.
                surface_drop_raw = _read_first_available(
                    wf,
                    ["surface_drop_step"],
                    default=np.zeros_like(cells_eaten_step),
                )
                gut_drop_raw = _read_first_available(
                    wf,
                    ["gut_drop_step", "poop_drop_step"],
                    default=np.zeros_like(cells_eaten_step),
                )

                n_worms = len(np.unique(worm_i))
                if n_worms > 0 and len(cells_eaten_step) % n_worms == 0:
                    n_steps = len(cells_eaten_step) // n_worms
                    worm_on_frac = on_patch.reshape(n_steps, n_worms).mean(axis=1)
                    worm_step_sum = cells_eaten_step.reshape(n_steps, n_worms).sum(axis=1)
                    surface_drop_step = np.asarray(surface_drop_raw, dtype=float).reshape(n_steps, n_worms).sum(axis=1)
                    gut_drop_step = np.asarray(gut_drop_raw, dtype=float).reshape(n_steps, n_worms).sum(axis=1)
                    worm_stats_available = True
                    dep_stats_available = True

    if worm_stats_available:
        n_common = min(len(worm_on_frac), len(dB_feed), len(dB_feed_requested))
        worm_on_frac = worm_on_frac[:n_common]
        dB_feed = dB_feed[:n_common]
        dB_feed_requested = dB_feed_requested[:n_common]
        B_deposited = B_deposited[:n_common]

        if dep_stats_available:
            surface_drop_step = surface_drop_step[:n_common]
            gut_drop_step = gut_drop_step[:n_common]

        mismatch = np.abs(worm_step_sum[:n_common] - dB_feed_requested)
        print(f"[diag] worm vs requested max abs mismatch per step: {mismatch.max():.3e}")
        print(f"[diag] total requested from worms: {worm_step_sum[:n_common].sum():.2f}")
        print(f"[diag] total requested in env:   {dB_feed_requested.sum():.2f}")
        print(f"[diag] total applied in env:     {dB_feed.sum():.2f}")

        if dep_stats_available:
            total_surface = surface_drop_step.sum()
            total_gut = gut_drop_step.sum()
            print(f"[diag] total deposited by surface shedding: {total_surface:.2f}")
            print(f"[diag] total deposited by gut-drop:         {total_gut:.2f}")
            print(f"[diag] total deposited (worm history):      {total_surface + total_gut:.2f}")
    else:
        print("[diag] worm verification curves unavailable (missing worm_hist datasets or shape mismatch).")
        print(f"[diag] looked for worm history at: {worm_path}")
        n_common = len(dB_feed)
        dB_feed = dB_feed[:n_common]
        dB_feed_requested = dB_feed_requested[:n_common]
        B_deposited = B_deposited[:n_common]

    t_step_min = np.arange(n_common, dtype=float) * dt

    cum_removed = np.cumsum(dB_feed)

    if dep_stats_available:
        cum_surface = np.cumsum(surface_drop_step)
        cum_gut = np.cumsum(gut_drop_step)
        cum_drop_total = cum_surface + cum_gut
    else:
        cum_surface = np.zeros_like(cum_removed)
        cum_gut = np.zeros_like(cum_removed)
        cum_drop_total = np.zeros_like(cum_removed)

    total_drop_step = (surface_drop_step + gut_drop_step) if dep_stats_available else np.zeros_like(dB_feed)

    t_rate_min, feed_rate_min = _aggregate_per_minute(dB_feed, dt, agg="sum")
    _, total_drop_rate_min = _aggregate_per_minute(total_drop_step, dt, agg="sum")
    _, on_patch_rate_min = _aggregate_per_minute(
        worm_on_frac if worm_stats_available else np.zeros_like(dB_feed),
        dt,
        agg="mean",
    )

    plt.figure(figsize=(12, 10))

    # 1) Cumulative feeding and deposition
    plt.subplot(3, 1, 1)
    plt.plot(t_step_min, cum_removed, label="Cumulative fed (applied)", linewidth=1.8)
    if dep_stats_available:
        plt.plot(t_step_min, cum_surface, label="Cumulative shed deposit", linewidth=1.4)
        plt.plot(t_step_min, cum_gut, label="Cumulative gut-drop deposit", linewidth=1.4)
        plt.plot(t_step_min, cum_drop_total, "--", label="Cumulative total deposit", linewidth=1.8)
    plt.ylabel("Cumulative cells")
    plt.title("Cumulative feeding and deposition over time")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)

    # 2) Worm occupancy on patch
    plt.subplot(3, 1, 2)
    if len(on_patch_rate_min) > 0:
        plt.plot(t_rate_min, on_patch_rate_min, label="Fraction worms on patch")
        plt.ylim(-0.02, 1.02)
        plt.legend(loc="best")
    else:
        plt.text(0.5, 0.5, "worm occupancy unavailable", ha="center", va="center")
    plt.ylabel("On-patch fraction")
    plt.title("Worm occupancy of patch")
    plt.grid(True, alpha=0.3)

    # 3) Per-minute feeding and deposition rates
    plt.subplot(3, 1, 3)
    if len(t_rate_min) > 0:
        plt.plot(t_rate_min, feed_rate_min, label="Feeding rate (cells/min)", linewidth=1.6)
        plt.plot(t_rate_min, total_drop_rate_min, "--", label="Total deposition rate", linewidth=1.8)
        plt.legend(loc="best")
    else:
        plt.text(0.5, 0.5, "rate data unavailable", ha="center", va="center")
    plt.xlabel("time (minutes)")
    plt.ylabel("Cells / minute")
    plt.title("Per-minute feeding and deposition rates")
    plt.grid(True, alpha=0.3)

    if dep_stats_available:
        difference = cum_removed - cum_drop_total
        print(f"[diag] final cumulative fed:             {cum_removed[-1]:.2f}")
        print(f"[diag] final cumulative deposited total: {cum_drop_total[-1]:.2f}")
        print(f"[diag] final fed - deposited:            {difference[-1]:.2f}")
    else:
        print("[diag] deposition pathway curves unavailable (missing drop-step datasets).")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
