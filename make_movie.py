import os
import time
import cv2
import sys
import glob2
import h5py
import shutil
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

def read_config(base_exp_dir):
    """Read configuration options from .cfg file in experiment directory"""
    cfg_paths = glob2.glob(f"{base_exp_dir}/*.cfg")

    if not cfg_paths:
        print(f"Error: No .cfg file found in {base_exp_dir}")
        print(f"Contents: {os.listdir(base_exp_dir)}")
        raise FileNotFoundError(f"No config file in {base_exp_dir}")
    
    cfg_path = cfg_paths[0]

    with open(cfg_path, "r") as infile:
        lines = [line.split() for line in infile]
        cfg_opts = {}
        for key, val in lines:
            key = key.replace('--', '')

            try:
                val = float(val)
            except:
                try:
                    val = int(val)
                except:
                    if val.startswith("T"):
                        val = True
                    elif val.startswith("F"):
                        val = False
                    pass
            cfg_opts[key] = val
    return cfg_opts

def imgs2vid(imgs, outpath, fps=15):
    """Convert list of images to video using OpenCV"""
    height, width, layers = imgs[0].shape
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    video = cv2.VideoWriter(outpath, fourcc, fps, (width, height), True)

    for img_i, img in enumerate(imgs):
        video.write(img)

    cv2.destroyAllWindows()
    video.release()

def process_data(env_path, worm_path):
    """Process worm and bacteria data from HDF5 files"""
    # Load bacteria time series (3D array)
    bacteria_history = []
    with h5py.File(env_path, 'r') as infile:
        if 'bacteria' in infile:
            bacteria_history = np.array(infile['bacteria'])

    # Get worm measurements
    worm_data = {}
    with h5py.File(worm_path, 'r') as infile:
        for key, val in infile.items():
            worm_data[key] = np.array(val)
    worm_nums = np.unique(worm_data['worm_i'])
    worms = {}
    for worm_num in worm_nums:
        idxs = np.where(worm_data['worm_i']==worm_num)
        worms[worm_num] = {
            "x" : worm_data['x'][idxs],
            "y" : worm_data['y'][idxs],
        }

    return worms, bacteria_history

def plot_frame(frame_i, worms, bacteria_history, source_template, legend_colors, texts, script_config, convert_xy_to_index, total_frames, show_gradient):  
    """Plot a single frame of the simulation including worms and bacteria concentration"""

    # Fixed figure size — must be identical every frame so cv2 can stack them
    fig = plt.figure(figsize=(8, 7), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')

    if len(bacteria_history) > frame_i:
        bacteria_grid = bacteria_history[frame_i].copy()

        if show_gradient:
            # Compute gradient magnitude of the normalised density field
            K_rho      = script_config.get('K_rho', 4.58e8)
            b_norm     = bacteria_grid / K_rho                      # normalise to [0, 1]
            dx = script_config.get('dx', 0.01)

            grad_y, grad_x = np.gradient(b_norm, dx)               # both in cm^-1
            grad_mag   = np.sqrt(grad_x**2 + grad_y**2)            # magnitude |∇b_norm|

            # Mask zeros so background stays black
            grad_masked = np.ma.masked_where(grad_mag == 0, grad_mag)
            # Mask near-zero values (inside and outside patch)
            # Use a small threshold instead of == 0 to catch floating point near-zeros
            # grad_masked = np.ma.masked_where(grad_mag < 1e-6, grad_mag)

            cmap = plt.cm.Greens.copy()                                
            cmap.set_bad(color='black')

            im = plt.imshow(grad_masked, cmap=cmap,
                            origin='upper', alpha=0.9,
                            interpolation='nearest')

            clb = plt.colorbar(im, shrink=0.8, format='%.2e')
            clb.ax.set_title('|∇b|\n(cm⁻¹)', fontsize=7)
            clb.ax.set_facecolor('black')
            clb.outline.set_edgecolor('white')
            clb.ax.tick_params(color='white', labelcolor='white')
            clb.ax.title.set_color('white')

        else:
            # --- default: show density ---
            min_factor = script_config.get('movie_min_density_factor', 0.1)
            min_b = script_config.get('rho_0', 1.27e8) * min_factor
            max_b = script_config.get('K_rho', 4.58e8)
            bacteria_masked = np.ma.masked_where(bacteria_grid == 0, bacteria_grid)

            cmap = plt.cm.Greens.copy()
            cmap.set_bad(color='black')

            im = plt.imshow(bacteria_masked, cmap=cmap, vmin=min_b, vmax=max_b,
                            origin='upper', alpha=0.8, interpolation='bilinear')

            clb = plt.colorbar(im, shrink=0.8, format='%.2e')
            clb.ax.set_title('Density\n(cells/cm²)', fontsize=7)
            clb.ax.set_facecolor('black')
            clb.outline.set_edgecolor('white')
            clb.ax.tick_params(color='white', labelcolor='white')
            clb.ax.title.set_color('white')

            # Draw deposits as small green circles from local peaks.
            if source_template is not None and source_template.shape == bacteria_grid.shape:
                deposit_grid = bacteria_grid - source_template
                deposit_grid = np.where(deposit_grid > 0.0, deposit_grid, 0.0)
                if np.any(deposit_grid > 0.0):
                    draw_threshold = float(script_config.get('deposit_display_threshold', 10.0))
                    deposit_grid = np.where(deposit_grid >= draw_threshold, deposit_grid, 0.0)
                    if np.any(deposit_grid > 0.0):
                        kernel = np.ones((3, 3), dtype=np.uint8)
                        dilated = cv2.dilate(deposit_grid.astype(np.float32), kernel, iterations=1)
                        peak_mask = (deposit_grid > 0.0) & (deposit_grid >= (dilated - 1e-12))
                        rows, cols = np.where(peak_mask)
                        if rows.size > 0:
                            plt.scatter(
                                cols,
                                rows,
                                marker='o',
                                s=10,
                                c='#00ff66',
                                alpha=0.95,
                                edgecolors='none',
                                zorder=6,
                            )

    # --- worms and trails ---
    num_worms  = len(worms)
    worm_colors = plt.cm.tab10(np.linspace(0, 1, max(num_worms, 1)))
    grid_size = bacteria_history[0].shape[0] if len(bacteria_history) > 0 else 301

    # How many past frames to show in trail (e.g. last 100 frames)
    TRAIL_LENGTH = 100

    for worm_key, worm_vals in worms.items():
        if frame_i >= len(worm_vals['x']):
            continue

        color = worm_colors[int(worm_key) % len(worm_colors)]

        # --- trail: past positions as faint dots ---
        trail_start = max(0, frame_i - TRAIL_LENGTH)
        trail_x = worm_vals['x'][trail_start:frame_i]
        trail_y = worm_vals['y'][trail_start:frame_i]

        if len(trail_x) > 0:
            # Fade trail opacity from 0 (oldest) to 0.4 (most recent)
            n_trail = len(trail_x)
            for t_i in range(n_trail):
                alpha_trail = 0.05 + 0.35 * (t_i / max(n_trail - 1, 1))
                plt.scatter(
                    convert_xy_to_index(trail_x[t_i]),
                    convert_xy_to_index(trail_y[t_i]),
                    color=color, s=5,                  # small trail dots
                    alpha=alpha_trail,
                    edgecolors='none', zorder=5
                )

        # --- current worm position ---
        plt.scatter(
            convert_xy_to_index(worm_vals['x'][frame_i]),
            convert_xy_to_index(worm_vals['y'][frame_i]),
            color=color, s=30,                         # reduced from 100
            edgecolors='white', linewidths=0.5,
            zorder=10
        )

    # --- formatting ---
    GRID_SIZE = grid_size
    plt.xlim(0, GRID_SIZE)
    plt.ylim(0, GRID_SIZE)
    plt.xlabel("X (cm)", color='white')
    plt.ylabel("Y (cm)", color='white')

    # Tick labels in cm instead of grid indices
    n_ticks  = 7
    x_min    = script_config['x_min']
    x_max    = script_config['x_max']
    tick_idx = np.linspace(0, GRID_SIZE, n_ticks)
    tick_lbl = [f"{v:.1f}" for v in np.linspace(x_min, x_max, n_ticks)]
    plt.xticks(tick_idx, tick_lbl, fontsize=7, color='white')
    plt.yticks(tick_idx, tick_lbl, fontsize=7, color='white')
    plt.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    N      = script_config['num_worms']
    dx_val = f"{script_config.get('dx', 0):.3f}".rstrip('0').rstrip('.')
    dt_val = f"{script_config.get('dt', 0):.10f}".rstrip('0').rstrip('.')
    vmax_val = f"{script_config.get('v_max', 0):.2f}".rstrip('0').rstrip('.')
    vmin_val = f"{script_config.get('v_min', 0):.2f}".rstrip('0').rstrip('.')
    title  = (f"Worms: {int(N)} -- dx: {dx_val} -- dt: {dt_val} "
              f"-- k: {script_config.get('boundary_k')} -- R: {script_config.get('R')} -- vmax: {vmax_val} -- vmin: {vmin_val}")
    plt.title(f"{title}\nt: {frame_i}/{total_frames}", color='white')

    filename = f'{MOVIE_FRAME_PATH}/t{frame_i:05d}.png'
    plt.savefig(filename, dpi=150, facecolor=fig.get_facecolor())
    plt.close()

def setup_opts():
    """Setup command line options for the script"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, default='default_path_value', help='Path to experiment folder')
    parser.add_argument('-r', '--fps', type=int, default=5, help='FPS for output movie')
    parser.add_argument('-s', '--stepsize', type=int, default=1, help='Step size for plotting data')
    parser.add_argument('-a', '--all', action='store_true', help='Plot all experiments in the experiments/ folder')
    parser.add_argument('-sg', '--show_gradient', action='store_true', help="Visualize gradient magnitude(|∇b|) instead of density")
    return parser.parse_args()

def main(exp_path, fps, stepsize, show_gradient):
    """Main function to generate movie frames and compile them into a video"""
    # Obtain parameters from config
    script_config = read_config(exp_path)
    X_MIN = script_config['x_min']
    X_MAX = script_config['x_max']
    DX = script_config['dx']
    GRID_SIZE = np.arange(X_MIN, X_MAX+DX, DX).shape[0]
    convert_xy_to_index = lambda xy: ((xy - X_MIN) / (X_MAX - X_MIN)) * GRID_SIZE

    # Get data path
    worm_path = os.path.join(exp_path, "worm_hist.h5")
    env_path = os.path.join(exp_path, "environment_hist.h5")

    # Obtain & process data
    worms, bacteria_sources = process_data(env_path, worm_path)
    source_template = None
    if len(bacteria_sources) > 0:
        # Use frame 0 as baseline source map to avoid tiny subtraction artifacts near sources.
        source_template = bacteria_sources[0].copy()

    # Setup for plotting
    total_frames = len(list(worms.values())[0]['x'])

    # Calculate total number of frames (stepsize) to be used in each movie
    calculated_stepsize = max(1, total_frames // stepsize)

    texts = ['Worm', 'Bacteria']
    legend_colors = ['Gray', 'Green']

    for frame_i in range(0, total_frames, calculated_stepsize):
            sys.stdout.write(f"\rMaking frame {frame_i}/{total_frames-1}")
            sys.stdout.flush()

            plot_frame(frame_i, worms, bacteria_sources, source_template, legend_colors, texts, script_config, convert_xy_to_index, total_frames, show_gradient)
    
    # Last frame
    frame_i = total_frames - 1
    sys.stdout.write(f"\rMaking frame {frame_i}/{total_frames-1}")
    sys.stdout.flush()
    plot_frame(frame_i, worms, bacteria_sources, source_template, legend_colors, texts, script_config, convert_xy_to_index, total_frames, show_gradient)

    # Stitching frames together to create video
    all_img_paths = np.sort(glob2.glob(f"{MOVIE_FRAME_PATH}/*.png"))
    all_imgs = np.array([cv2.imread(img) for img in all_img_paths])
    trial_name = os.path.basename(exp_path)  # Extract just 'N1_seed42'
    savepath = os.path.join("movies", f"{trial_name}.mp4")
    imgs2vid(all_imgs, savepath, fps)

if __name__ == '__main__':
    opts = setup_opts()

    FPS = opts.fps
    INTERVAL = opts.stepsize
    SHOW_GRADIENT = opts.show_gradient

    print("\n---------- Visualizing worm model data ----------")
    
    # Create movies folder
    movies_dir = "movies"
    os.makedirs(movies_dir, exist_ok=True)

    # Check if neither -p nor --all is specified
    if not opts.all and opts.path == 'default_path_value':
        print("Error: You must specify either -p/--path <experiment_folder> or use --all")
        print("Example: python make_movie.py -p N1_seed42_dx0.01_dt0.0002_D0.1")
        print("Or:      python make_movie.py --all")
        sys.exit(1)

    if opts.all:
        # Process all experiments in the experiments/ folder
        experiments_dir = "experiments"
        
        if not os.path.exists(experiments_dir):
            print(f"Error: {experiments_dir} directory not found!")
            sys.exit(1)

        exp_folders = [f for f in os.listdir(experiments_dir) 
                if os.path.isdir(os.path.join(experiments_dir, f)) 
                and not f.startswith('.')]
        
        if not exp_folders:
            print(f"Error: No experiment folders found in {experiments_dir}!")
            sys.exit(1)
        
        exp_folders.sort()  # Sort folders alphabetically

        print(f"\nFound {len(exp_folders)} experiments:")
        for folder in exp_folders:
            print(f"  - {folder}")
        print()

        for i, exp_folder in enumerate(exp_folders, 1):
            print(f"{'='*60}\n")
            print("="*60)
            print(f"Processing {i}/{len(exp_folders)}: {exp_folder}")
            
            BASE_EXPERIMENT_DIR = os.path.join(experiments_dir, exp_folder)
            MOVIE_FRAME_PATH = os.path.join(BASE_EXPERIMENT_DIR, "movie_frames")
            
            # Clean and create frame directory
            if os.path.exists(MOVIE_FRAME_PATH):
                shutil.rmtree(MOVIE_FRAME_PATH)
            os.makedirs(MOVIE_FRAME_PATH, exist_ok=True)
            
            try:
                main(BASE_EXPERIMENT_DIR, FPS, INTERVAL, SHOW_GRADIENT)
                print(f"\n✓ Completed: {exp_folder}")
            except Exception as e:
                print(f"\n✗ Error processing {exp_folder}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print("="*60 + "\n")
        print(f"Finished! Processed {len(exp_folders)} experiments")
        print("\n" + "="*60)
    
    else:
        # Process single experiment specified by --path
        TRIAL_PATH = opts.path
        BASE_EXPERIMENT_DIR = f"experiments/{TRIAL_PATH}"
        MOVIE_FRAME_PATH = f"{BASE_EXPERIMENT_DIR}/movie_frames"

        if not os.path.exists(BASE_EXPERIMENT_DIR):
            print(f"Error: {BASE_EXPERIMENT_DIR} not found!")
            sys.exit(1)
        
        # Clean and create frame directory
        if os.path.exists(MOVIE_FRAME_PATH):
            shutil.rmtree(MOVIE_FRAME_PATH)
        os.makedirs(MOVIE_FRAME_PATH, exist_ok=True)
        
        print(f"Processing: {BASE_EXPERIMENT_DIR}\n")
        
        main(BASE_EXPERIMENT_DIR, FPS, INTERVAL, SHOW_GRADIENT)
    
    print("\nDone!\n")