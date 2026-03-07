"""
analysis.py  —  Standalone analysis plots using saved HDF5 data.

Produces:
  1. single_worm_N_trail.png         — full rainbow trail of one worm over 4 days (no histogram)
  2. multi_worm_Xh_analysis.png      — side-by-side: trajectory map + end-to-end histogram

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


def load_bacteria_frame(exp_dir, frame_idx=-1):
    """Load a single bacteria grid frame from HDF5."""
    env_path = os.path.join(exp_dir, "environment_hist.h5")
    with h5py.File(env_path, "r") as f:
        bacteria = np.array(f["bacteria"][frame_idx])
    return bacteria


def get_final_patch_radius(exp_dir):
    """Compute final patch radius in cm from last bacteria frame."""
    bacteria = load_bacteria_frame(exp_dir, frame_idx=-1)
    nonzero  = np.argwhere(bacteria > 0)
    if len(nonzero) == 0:
        return 0.5
    cfg    = read_config(exp_dir)
    dx     = float(cfg.get("dx", 0.01))
    centre = bacteria.shape[0] / 2
    dists  = np.sqrt(((nonzero[:, 0] - centre) * dx)**2 +
                     ((nonzero[:, 1] - centre) * dx)**2)
    return float(np.max(dists))


def make_out_dir(exp_dir):
    """
    Create and return output directory:
        WormABM/analysis/<exp_folder_name>/
    """
    exp_name = os.path.basename(os.path.normpath(exp_dir))
    # Walk up to WormABM root (parent of experiments/)
    worm_abm_root = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(worm_abm_root, "analysis", exp_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# ===========================================================================
#   Plot 1 — Single worm, full 4-day trail only  (no histogram)
#   Fixed arena size so all worm plots are identical dimensions
#   Legend placed outside the plot on the right
# ===========================================================================

def plot_single_worm_trail(exp_dir, worm_num=0, out_dir=None):
    """
    Full trajectory of one worm over 4 days:
      - White background, fixed arena axes [-1.5, 1.5] cm (same size for all worms)
      - Trail coloured by time (blue=start -> red=end, rainbow cmap)
      - Dashed circle = final bacteria patch boundary
      - Colorbar and legend placed OUTSIDE the plot on the right
    """
    worms   = load_worm_data(exp_dir)
    out_dir = out_dir or make_out_dir(exp_dir)

    if worm_num not in worms:
        raise ValueError(f"Worm {worm_num} not found. Available: {list(worms.keys())}")

    worm = worms[worm_num]
    x_mm = worm["x"] * 10      # cm -> mm
    y_mm = worm["y"] * 10
    t    = worm["t"]            # minutes
    days = t.max() / 60 / 24

    norm   = plt.Normalize(t.min(), t.max())
    cmap   = cm.rainbow
    colors = cmap(norm(t))

    # Extra right margin for legend + colorbar outside the axes
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.subplots_adjust(right=0.72)     # leave 28% on right for legend/colorbar
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Trail coloured by time
    for i in range(len(x_mm) - 1):
        ax.plot(x_mm[i:i+2], y_mm[i:i+2],
                color=colors[i], linewidth=0.6, alpha=0.85)

    # Start and end markers
    ax.scatter(x_mm[0],  y_mm[0],  color='blue', s=50, zorder=6,
               edgecolors='black', linewidths=0.5)
    ax.scatter(x_mm[-1], y_mm[-1], color='red',  s=50, marker='*',
               zorder=6, edgecolors='black', linewidths=0.5)

    # Final patch boundary circle
    R_mm    = get_final_patch_radius(exp_dir) * 10
    theta_c = np.linspace(0, 2 * np.pi, 300)
    ax.plot(R_mm * np.cos(theta_c), R_mm * np.sin(theta_c),
            'k--', linewidth=1.2, alpha=0.7)

    # Fixed arena limits — same for every worm
    ax.set_xlim(ARENA_MIN_MM, ARENA_MAX_MM)
    ax.set_ylim(ARENA_MIN_MM, ARENA_MAX_MM)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)", fontsize=12)
    ax.set_ylabel("y (mm)", fontsize=12)
    ax.set_title(f"Worm {worm_num} — {days:.1f} day full trail", fontsize=13)

    # Colorbar outside axes (right side)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.74, 0.15, 0.03, 0.55])   # [left, bottom, width, height]
    cbar    = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Time (min)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # Legend outside axes below colorbar
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
               markersize=8, label='Start', markeredgecolor='black'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
               markersize=10, label='End', markeredgecolor='black'),
        Line2D([0], [0], color='black', linewidth=1.2, linestyle='--',
               label='Patch edge'),
    ]
    fig.legend(handles=legend_handles, fontsize=9,
               loc='center right',
               bbox_to_anchor=(1.0, 0.82),
               framealpha=0.8)

    out_path = os.path.join(out_dir, f"single_worm_{worm_num}_trail.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ===========================================================================
#   Plot 2 — Multi-worm analysis: trajectory map  +  end-to-end histogram
#   Legend outside trajectory axes, no trail clipping
# ===========================================================================

def plot_multi_worm_trajectories(exp_dir, hours=4, out_dir=None):
    """
    Side-by-side figure:
      Left  — all worm trails, white background, green bacteria patch
              legend placed OUTSIDE below the axes
      Right — end-to-end distance histogram for all worms combined
    """
    cfg     = read_config(exp_dir)
    worms   = load_worm_data(exp_dir)
    out_dir = out_dir or make_out_dir(exp_dir)

    dt_min  = float(cfg.get("dt",    1.0))
    steps   = int((hours * 60) / dt_min)
    x_min   = float(cfg.get("x_min", -1.5))
    x_max   = float(cfg.get("x_max",  1.5))
    K_rho   = float(cfg.get("K_rho",  4.58e8))
    rho_0   = float(cfg.get("rho_0",  1.27e8))

    bacteria    = load_bacteria_frame(exp_dir, frame_idx=min(steps - 1, -1))
    num_worms   = len(worms)
    worm_colors = cm.tab10(np.linspace(0, 1, max(num_worms, 1)))

    # Extra bottom margin for legend below trajectory axes
    fig, (ax_traj, ax_hist) = plt.subplots(
        1, 2, figsize=(17, 8),
        gridspec_kw={"width_ratios": [1, 1]}
    )
    fig.subplots_adjust(bottom=0.22, wspace=0.3)
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

    all_end_to_end = []
    worm_legend_handles = []

    for worm_num, worm in worms.items():
        x_cm = worm["x"][:steps]
        y_cm = worm["y"][:steps]
        color = worm_colors[int(worm_num) % len(worm_colors)]

        ax_traj.plot(x_cm, y_cm, color=color, linewidth=0.9, alpha=0.85,
                     clip_on=False)     # clip_on=False prevents edge cutting

        # Start: circle
        ax_traj.scatter(x_cm[0], y_cm[0], color=color, s=35,
                        edgecolors='black', linewidths=0.5, zorder=6,
                        clip_on=False)
        # End: star
        ax_traj.scatter(x_cm[-1], y_cm[-1], color=color, s=70,
                        marker='*', edgecolors='black', linewidths=0.4,
                        zorder=7, clip_on=False)

        worm_legend_handles.append(
            Line2D([0], [0], color=color, linewidth=1.5, label=f"Worm {worm_num}")
        )

        x_mm = x_cm * 10
        y_mm = y_cm * 10
        end_to_end = np.sqrt((x_mm - x_mm[0])**2 + (y_mm - y_mm[0])**2)
        all_end_to_end.extend(end_to_end.tolist())

    # Marker legend entries
    marker_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=7, label='Start', linewidth=0,
               markeredgecolor='black'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gray',
               markersize=10, label='End', linewidth=0,
               markeredgecolor='black'),
    ]

    # Legend BELOW the trajectory axes
    ax_traj.legend(
        handles=worm_legend_handles + marker_handles,
        fontsize=7,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.12),    # below axes
        ncol=6,
        framealpha=0.85,
        borderaxespad=0.
    )

    # Fixed arena limits — no clipping
    ax_traj.set_xlim(x_min, x_max)
    ax_traj.set_ylim(x_min, x_max)
    ax_traj.set_xlabel("x (cm)", fontsize=12)
    ax_traj.set_ylabel("y (cm)", fontsize=12)
    ax_traj.set_title(f"{num_worms} worms — {hours}h trajectories", fontsize=13)

    # ── RIGHT: end-to-end histogram ───────────────────────────────────────
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
    # Histogram legend outside below — matching trajectory axes
    ax_hist.legend(
        handles=[hist_patch],
        fontsize=9,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.12),
        framealpha=0.85,
        borderaxespad=0.
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

    # Multi-worm side-by-side (always produced)
    print(f"Plotting {opts.hours}h multi-worm analysis...")
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