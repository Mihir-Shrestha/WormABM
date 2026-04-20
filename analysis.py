"""
analysis.py  —  Standalone analysis plots using saved HDF5 data.

Produces:
  1. single_worm_N_trail.png         — full rainbow trail of one worm over 4 days (no histogram)
  2. multi_worm_Xh_analysis.png      — three panels: trajectory map + end-to-end histogram + MSD

Usage:
    python analysis.py --exp_path experiments/N10_dx0.01_dt1_k10.0
    python analysis.py --exp_path experiments/N10_dx0.01_dt1_k10.0 --single_worm 0
    python analysis.py --exp_path experiments/N10_dx0.01_dt1_k10.0 --hours 4
"""

import argparse
import os
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from glob import glob


# ===========================================================================
#   Constants — fixed arena limits so all single-worm plots are same size
# ===========================================================================

ARENA_MIN_MM = -15.0    # -1.5 cm in mm
ARENA_MAX_MM =  15.0    #  1.5 cm in mm


# ===========================================================================
#   I/O helpers
# ===========================================================================

def read_config(exp_dir):
    """Read .cfg file from experiment directory into a dict."""
    cfg_paths = glob(f"{exp_dir}/*.cfg")
    if not cfg_paths:
        raise FileNotFoundError(f"No .cfg file found in {exp_dir}")
    cfg = {}
    with open(cfg_paths[0], "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("--"):
                parts = line.split()
                if len(parts) == 2:
                    key = parts[0].lstrip("-")
                    try:
                        cfg[key] = float(parts[1])
                    except ValueError:
                        cfg[key] = parts[1]
    return cfg


def load_worm_data(exp_dir):
    """
    Load worm history from HDF5.
    Returns dict: worm_num -> {"x": array, "y": array, "theta": array, "t": array}
    """
    worm_path = os.path.join(exp_dir, "worm_hist.h5")
    with h5py.File(worm_path, "r") as f:
        t      = np.array(f["t"])
        worm_i = np.array(f["worm_i"])
        x      = np.array(f["x"])
        y      = np.array(f["y"])
        theta  = np.array(f["theta"])

    worms = {}
    for num in np.unique(worm_i):
        idx = worm_i == num
        worms[int(num)] = {
            "x"     : x[idx],
            "y"     : y[idx],
            "t"     : t[idx],
            "theta" : theta[idx],
        }
    return worms


def load_bacteria_frame(exp_dir, frame_idx=-1, target_step=None):
    """
    Load one bacteria grid frame from HDF5.

    If target_step is provided and the environment file stores a time/step vector,
    pick the closest available recorded frame at or after that step.
    """
    env_path = os.path.join(exp_dir, "environment_hist.h5")
    with h5py.File(env_path, "r") as f:
        bacteria_ds = f["bacteria"]
        n_frames = bacteria_ds.shape[0]
        if n_frames == 0:
            raise ValueError("No bacteria frames found in environment_hist.h5")

        if target_step is not None and "t" in f:
            env_t = np.asarray(f["t"][:], dtype=float)
            idx = int(np.searchsorted(env_t, float(target_step), side="left"))
            idx = min(max(idx, 0), n_frames - 1)
        else:
            if frame_idx < 0:
                idx = n_frames + frame_idx
            else:
                idx = frame_idx
            idx = min(max(int(idx), 0), n_frames - 1)

        bacteria = np.array(bacteria_ds[idx])
    return bacteria


def _connected_component(mask, start_row, start_col):
    """Return boolean mask of the 4-neighbour component containing the start cell."""
    h, w = mask.shape
    comp = np.zeros_like(mask, dtype=bool)
    if not (0 <= start_row < h and 0 <= start_col < w):
        return comp
    if not mask[start_row, start_col]:
        return comp

    stack = [(start_row, start_col)]
    comp[start_row, start_col] = True
    while stack:
        r, c = stack.pop()
        if r > 0 and mask[r - 1, c] and not comp[r - 1, c]:
            comp[r - 1, c] = True
            stack.append((r - 1, c))
        if r + 1 < h and mask[r + 1, c] and not comp[r + 1, c]:
            comp[r + 1, c] = True
            stack.append((r + 1, c))
        if c > 0 and mask[r, c - 1] and not comp[r, c - 1]:
            comp[r, c - 1] = True
            stack.append((r, c - 1))
        if c + 1 < w and mask[r, c + 1] and not comp[r, c + 1]:
            comp[r, c + 1] = True
            stack.append((r, c + 1))

    return comp


def get_final_patch_radius(exp_dir):
    """
    Estimate final main-patch radius in cm from the last bacteria frame.

    Uses a center-connected component above a threshold tied to rho_0,
    so sparse deposited micro-patches far away do not inflate the radius.
    """
    bacteria = load_bacteria_frame(exp_dir, frame_idx=-1)
    cfg    = read_config(exp_dir)
    dx     = float(cfg.get("dx", 0.01))
    rho_0  = float(cfg.get("rho_0", 1.27e6))
    threshold = max(1e-12, 0.05 * rho_0)

    mask = bacteria >= threshold
    if not np.any(mask):
        mask = bacteria > 0.0
    if not np.any(mask):
        return 0.5

    h, w = mask.shape
    c_row = int(round((h - 1) / 2.0))
    c_col = int(round((w - 1) / 2.0))

    if mask[c_row, c_col]:
        core = _connected_component(mask, c_row, c_col)
    else:
        coords = np.argwhere(mask)
        d2 = (coords[:, 0] - c_row) ** 2 + (coords[:, 1] - c_col) ** 2
        nearest = coords[int(np.argmin(d2))]
        core = _connected_component(mask, int(nearest[0]), int(nearest[1]))

    points = np.argwhere(core)
    if len(points) == 0:
        return 0.5

    dists = np.sqrt(((points[:, 0] - c_row) * dx) ** 2 + ((points[:, 1] - c_col) * dx) ** 2)
    return float(np.max(dists))


def make_out_dir(exp_dir):
    """
    Create and return output directory:
        WormABM/analysis/<exp_folder_name>/
    """
    exp_name      = os.path.basename(os.path.normpath(exp_dir))
    worm_abm_root = os.path.dirname(os.path.abspath(__file__))
    out_dir       = os.path.join(worm_abm_root, "analysis", exp_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# ===========================================================================
#   MSD helper
# ===========================================================================

def compute_msd(x_cm, y_cm, t_min, max_lag_fraction=0.5):
    """
    Compute the time-averaged MSD for a single worm trajectory.

        MSD(τ) = < |r(t+τ) - r(t)|² >_t

    Uses every starting point t to average over, up to
    max_lag_fraction * total_steps as the maximum lag.

    Parameters
    ----------
    x_cm, y_cm      : position arrays in cm
    t_min           : time array in minutes
    max_lag_fraction: fraction of total length used as max lag (default 0.5)

    Returns
    -------
    lag_times_min : 1-D array of lag times in minutes
    msd_mm2       : 1-D array of MSD values in mm²
    """
    n          = len(x_cm)
    max_lag    = max(1, int(n * max_lag_fraction))
    dt_min     = float(t_min[1] - t_min[0]) if len(t_min) > 1 else 1.0

    x_mm = x_cm * 10   # cm -> mm
    y_mm = y_cm * 10

    lag_times = np.arange(1, max_lag + 1) * dt_min   # minutes
    msd       = np.zeros(max_lag)

    for lag in range(1, max_lag + 1):
        dx   = x_mm[lag:] - x_mm[:-lag]
        dy   = y_mm[lag:] - y_mm[:-lag]
        msd[lag - 1] = np.mean(dx**2 + dy**2)

    return lag_times, msd


# ===========================================================================
#   Plot 1 — Single worm, full 4-day trail only  (no histogram)
# ===========================================================================

def plot_single_worm_trail(exp_dir, worm_num=0, out_dir=None):
    """
    Full trajectory of one worm over 4 days:
      - White background, fixed arena axes (same size for all worms)
      - Trail coloured by time (blue=start -> red=end, rainbow cmap)
      - Dashed circle = final bacteria patch boundary
      - Colorbar and legend placed OUTSIDE the plot on the right
    """
    worms   = load_worm_data(exp_dir)
    out_dir = out_dir or make_out_dir(exp_dir)

    if worm_num not in worms:
        raise ValueError(f"Worm {worm_num} not found. Available: {list(worms.keys())}")

    cfg  = read_config(exp_dir)
    dt_min = float(cfg.get("dt", 1.0))

    worm = worms[worm_num]
    x_mm = worm["x"] * 10
    y_mm = worm["y"] * 10
    t_min = worm["t"] * dt_min
    days = t_min.max() / 60 / 24

    norm   = plt.Normalize(t_min.min(), t_min.max())
    cmap   = cm.rainbow
    colors = cmap(norm(t_min))

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.subplots_adjust(right=0.72)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    for i in range(len(x_mm) - 1):
        ax.plot(x_mm[i:i+2], y_mm[i:i+2],
                color=colors[i], linewidth=0.6, alpha=0.85)

    ax.scatter(x_mm[0],  y_mm[0],  color='blue', s=50, zorder=6,
               edgecolors='black', linewidths=0.5)
    ax.scatter(x_mm[-1], y_mm[-1], color='red',  s=50, marker='*',
               zorder=6, edgecolors='black', linewidths=0.5)

    R_mm    = get_final_patch_radius(exp_dir) * 10
    theta_c = np.linspace(0, 2 * np.pi, 300)
    ax.plot(R_mm * np.cos(theta_c), R_mm * np.sin(theta_c),
            'k--', linewidth=1.2, alpha=0.7)

    ax.set_xlim(ARENA_MIN_MM, ARENA_MAX_MM)
    ax.set_ylim(ARENA_MIN_MM, ARENA_MAX_MM)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)", fontsize=12)
    ax.set_ylabel("y (mm)", fontsize=12)
    ax.set_title(f"Worm {worm_num} — {days:.1f} day full trail", fontsize=13)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.74, 0.15, 0.03, 0.55])
    cbar    = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Time (min)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
               markersize=8, label='Start', markeredgecolor='black'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
               markersize=10, label='End', markeredgecolor='black'),
        Line2D([0], [0], color='black', linewidth=1.2, linestyle='--',
               label='Patch edge'),
    ]
    fig.legend(handles=legend_handles, fontsize=9,
               loc='center right', bbox_to_anchor=(1.0, 0.82),
               framealpha=0.8)

    out_path = os.path.join(out_dir, f"single_worm_{worm_num}_trail.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ===========================================================================
#   Plot 2 — Three-panel: trajectory map + end-to-end histogram + MSD
# ===========================================================================

def plot_multi_worm_trajectories(exp_dir, hours=4, out_dir=None):
    """
    Three-panel figure:
      Left   — all worm trails on white background with green bacteria patch
      Centre — end-to-end distance histogram for all worms combined
      Right  — MSD log-log plot per worm + ensemble mean, with slope=1 and slope=2 guides
    """
    cfg     = read_config(exp_dir)
    worms   = load_worm_data(exp_dir)
    out_dir = out_dir or make_out_dir(exp_dir)

    dt_min  = float(cfg.get("dt",    1.0))
    target_step = int((hours * 60) / max(dt_min, 1e-12))
    x_min   = float(cfg.get("x_min", -1.5))
    x_max   = float(cfg.get("x_max",  1.5))
    K_rho   = float(cfg.get("K_rho",  4.58e8))
    rho_0   = float(cfg.get("rho_0",  1.27e8))

    bacteria    = load_bacteria_frame(exp_dir, target_step=target_step)
    num_worms   = len(worms)
    worm_colors = cm.tab10(np.linspace(0, 1, max(num_worms, 1)))

    fig, (ax_traj, ax_hist, ax_msd) = plt.subplots(
        1, 3, figsize=(24, 8),
        gridspec_kw={"width_ratios": [1, 1, 1]}
    )
    fig.subplots_adjust(bottom=0.22, wspace=0.35)
    fig.patch.set_facecolor('white')

    # ── LEFT: trajectory map ───────────────────────────────────────────────
    ax_traj.set_facecolor('white')

    bacteria_masked = np.ma.masked_where(bacteria <= 0, bacteria)
    cmap_bac = plt.cm.Greens.copy()
    cmap_bac.set_bad(color='white')

    ax_traj.imshow(bacteria_masked, cmap=cmap_bac,
                   vmin=rho_0, vmax=K_rho,
                   origin='upper', alpha=0.7,
                   extent=[x_min, x_max, x_min, x_max],
                   aspect='equal')

    all_end_to_end  = []
    all_msd_arrays  = []       # list of (lag_times, msd) per worm
    worm_legend_handles = []

    for worm_num, worm in worms.items():
        t_w_min = worm["t"] * dt_min
        keep = t_w_min <= (hours * 60.0)
        if not np.any(keep):
            continue

        x_cm = worm["x"][keep]
        y_cm = worm["y"][keep]
        t_w  = t_w_min[keep]
        if len(x_cm) == 0:
            continue
        color = worm_colors[int(worm_num) % len(worm_colors)]

        # --- trajectory ---
        ax_traj.plot(x_cm, y_cm, color=color, linewidth=0.9,
                     alpha=0.85, clip_on=False)
        ax_traj.scatter(x_cm[0],  y_cm[0],  color=color, s=35,
                        edgecolors='black', linewidths=0.5,
                        zorder=6, clip_on=False)
        ax_traj.scatter(x_cm[-1], y_cm[-1], color=color, s=70,
                        marker='*', edgecolors='black', linewidths=0.4,
                        zorder=7, clip_on=False)

        worm_legend_handles.append(
            Line2D([0], [0], color=color, linewidth=1.5,
                   label=f"Worm {worm_num}")
        )

        # --- end-to-end ---
        x_mm = x_cm * 10
        y_mm = y_cm * 10
        end_to_end = np.sqrt((x_mm - x_mm[0])**2 + (y_mm - y_mm[0])**2)
        all_end_to_end.extend(end_to_end.tolist())

        # --- MSD per worm ---
        if len(x_cm) > 2:
            lag_times, msd = compute_msd(x_cm, y_cm, t_w, max_lag_fraction=0.5)
            all_msd_arrays.append((lag_times, msd, color, worm_num))

    # Marker legend entries
    marker_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=7, label='Start', linewidth=0,
               markeredgecolor='black'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gray',
               markersize=10, label='End', linewidth=0,
               markeredgecolor='black'),
    ]

    ax_traj.legend(
        handles=worm_legend_handles + marker_handles,
        fontsize=7, loc='upper center',
        bbox_to_anchor=(0.5, -0.12),
        ncol=6, framealpha=0.85, borderaxespad=0.
    )

    ax_traj.set_xlim(x_min, x_max)
    ax_traj.set_ylim(x_min, x_max)
    ax_traj.set_xlabel("x (cm)", fontsize=12)
    ax_traj.set_ylabel("y (cm)", fontsize=12)
    ax_traj.set_title(f"{num_worms} worms — {hours}h trajectories", fontsize=13)

    # ── CENTRE: end-to-end histogram ──────────────────────────────────────
    ax_hist.set_facecolor('white')
    all_end_to_end = np.array(all_end_to_end)

    ax_hist.hist(all_end_to_end, bins=60, color='steelblue',
                 edgecolor='steelblue', density=True, alpha=1.0)
    ax_hist.set_xlabel("End-to-end distance (mm)", fontsize=12)
    ax_hist.set_ylabel("Probability density",      fontsize=12)
    ax_hist.set_title(f"End-to-end distance — all {num_worms} worms\n"
                      f"({hours}h window)", fontsize=13)
    ax_hist.set_xlim(left=0)

    hist_patch = mpatches.Patch(
        color='steelblue',
        label=f"All worms combined\n(n={len(all_end_to_end)} measurements)"
    )
    ax_hist.legend(
        handles=[hist_patch], fontsize=9,
        loc='upper center', bbox_to_anchor=(0.5, -0.12),
        framealpha=0.85, borderaxespad=0.
    )

    # ── RIGHT: MSD log-log ────────────────────────────────────────────────
    ax_msd.set_facecolor('white')

    # Plot each worm's MSD in its colour (thin, alpha)
    for lag_times, msd, color, worm_num in all_msd_arrays:
        ax_msd.loglog(lag_times, msd, color=color,
                      linewidth=0.8, alpha=0.5)

    # Ensemble mean MSD across all worms
    # Interpolate all to the common shortest lag_time array
    if all_msd_arrays:
        min_len    = min(len(lt) for lt, _, _, _ in all_msd_arrays)
        ref_lags   = all_msd_arrays[0][0][:min_len]
        msd_stack  = np.array([msd[:min_len] for _, msd, _, _ in all_msd_arrays])
        mean_msd   = np.mean(msd_stack, axis=0)

        ax_msd.loglog(ref_lags, mean_msd,
                      color='black', linewidth=2.5,
                      label='Ensemble mean', zorder=5)

        anchor_idx  = max(1, min_len // 10)
        anchor_lag  = ref_lags[anchor_idx]
        anchor_msd  = mean_msd[anchor_idx]

        lags_ref    = np.array([ref_lags[0], ref_lags[-1]])
        slope1 = anchor_msd * (lags_ref / anchor_lag) ** 1
        slope2 = anchor_msd * (lags_ref / anchor_lag) ** 2

        ax_msd.loglog(lags_ref, slope1, 'k--', linewidth=1.2,
                      label='slope = 1 (diffusive)')
        ax_msd.loglog(lags_ref, slope2, 'k:',  linewidth=1.2,
                      label='slope = 2 (ballistic)')
    else:
        ax_msd.text(0.5, 0.5, "MSD unavailable for selected window",
                    transform=ax_msd.transAxes, ha='center', va='center', fontsize=10)

    ax_msd.set_xlabel("Time lag (min)",          fontsize=12)
    ax_msd.set_ylabel("MSD (mm²)",               fontsize=12)
    ax_msd.set_title(f"Mean Squared Displacement\n({hours}h window)", fontsize=13)

    # Per-worm colour entries for MSD legend
    msd_worm_handles = [
        Line2D([0], [0], color=worm_colors[int(wn) % len(worm_colors)],
               linewidth=1.0, alpha=0.6, label=f"Worm {wn}")
        for _, _, _, wn in all_msd_arrays
    ]
    if all_msd_arrays:
        msd_ref_handles = [
            Line2D([0], [0], color='black', linewidth=2.5,
                   label='Ensemble mean'),
            Line2D([0], [0], color='black', linewidth=1.2,
                   linestyle='--', label='slope = 1 (diffusive)'),
            Line2D([0], [0], color='black', linewidth=1.2,
                   linestyle=':',  label='slope = 2 (ballistic)'),
        ]
    else:
        msd_ref_handles = []

    ax_msd.legend(
        handles=msd_worm_handles + msd_ref_handles,
        fontsize=7, loc='upper center',
        bbox_to_anchor=(0.5, -0.12),
        ncol=4, framealpha=0.85, borderaxespad=0.
    )

    out_path = os.path.join(out_dir, f"multi_worm_{int(hours)}h_analysis.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ===========================================================================
#   CLI
# ===========================================================================

def setup_opts():
    parser = argparse.ArgumentParser(description="Worm trajectory analysis plots")
    parser.add_argument("--exp_path",    type=str,   required=True,
                        help="Path to experiment directory")
    parser.add_argument("--single_worm", type=int,   default=None,
                        help="Worm number for single-worm trail plot "
                             "(omit to plot all worms)")
    parser.add_argument("--hours",       type=float, default=4.0,
                        help="Hours window for multi-worm analysis plot "
                             "(default: 4)")
    return parser.parse_args()


if __name__ == "__main__":
    opts    = setup_opts()
    exp_dir = opts.exp_path
    out_dir = make_out_dir(exp_dir)

    print(f"\n{'='*60}")
    print(f"Analysis : {exp_dir}")
    print(f"Output   : {out_dir}")
    print(f"{'='*60}\n")

    # Three-panel multi-worm plot (always produced)
    print(f"Plotting {opts.hours}h multi-worm analysis (trajectory + end-to-end + MSD)...")
    plot_multi_worm_trajectories(exp_dir, hours=opts.hours, out_dir=out_dir)

    # Single worm full trail
    worms = load_worm_data(exp_dir)
    if opts.single_worm is not None:
        print(f"Plotting single worm {opts.single_worm} full trail...")
        plot_single_worm_trail(exp_dir, worm_num=opts.single_worm, out_dir=out_dir)
    else:
        for worm_num in worms.keys():
            print(f"Plotting single worm {worm_num} full trail...")
            plot_single_worm_trail(exp_dir, worm_num=worm_num, out_dir=out_dir)

    print(f"\nDone! Plots saved to: {out_dir}\n")