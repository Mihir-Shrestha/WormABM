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
        worm_x = worm_data['x'][idxs]
        worm_y = worm_data['y'][idxs]
        worms[worm_num] = {"x" : worm_x, "y" : worm_y,}

    return worms, bacteria_history

def plot_frame(frame_i, worms, bacteria_history, legend_colors, texts, script_config, convert_xy_to_index, total_frames):  
    """Plot a single frame of the simulation including worms and bacteria concentration"""
    # Plot bacteria concentration map with gradient
    if len(bacteria_history) > frame_i:
        bacteria_grid = bacteria_history[frame_i]
        
        # Get min/max for color scaling
        min_b = np.min(bacteria_history)
        # max_b = np.max(bacteria_history) * 0.85
        max_b = 1
        
        # Display bacteria as heatmap 
        im = plt.imshow(bacteria_grid, cmap='Greens', vmin=min_b, vmax=max_b, 
                        origin='upper', alpha=0.8, interpolation='bilinear')
        
        # Add colorbar to show concentration scale
        clb = plt.colorbar(im, shrink=0.8, format='%.2f')
        clb.ax.set_title('Bacteria\nConc.')

    # Process worm data
    for worm_key, worm_vals in worms.items():
        x = worm_vals['x'][frame_i]
        y = worm_vals['y'][frame_i]
        color = 'Gray'
        plt.scatter(convert_xy_to_index(x), convert_xy_to_index(y),
                    color=color, s=100, edgecolors='black', zorder = 10)

    # Plot formatting
    patches = [ plt.plot([],[], marker="o", ms=10 if color_i==0 else 6, ls="", color=legend_colors[color_i],
                markeredgecolor="black", label="{:s}".format(texts[color_i]) )[0]  for color_i in range(len(texts)) ]
    plt.legend(handles=patches, bbox_to_anchor=(0.5, -0.15),
               loc='center', ncol=4, numpoints=1, labelspacing=0.3,
               fontsize='small', fancybox="True",
               handletextpad=0, columnspacing=0)
    plt.xlabel("X")
    plt.ylabel("Y")    
    
    # Set axes to INDEX SPACE
    GRID_SIZE = bacteria_grid.shape[0]
    plt.xlim(0, GRID_SIZE)
    plt.ylim(0, GRID_SIZE)

    # Title
    N = script_config['num_worms']
    seed = int(script_config['random_seed'])
    dx = script_config.get('dx', 'N/A')
    dt = script_config.get('dt', 'N/A')
    dx_value = f"{script_config.get('dx', 0):.3f}".rstrip('0').rstrip('.')
    dt_value = f"{script_config.get('dt', 0):.10f}".rstrip('0').rstrip('.')
    D = script_config.get('diffusion_coefficient', 'N/A')
    title = f"Worms: {int(N)} -- dx: {dx_value} -- dt: {dt_value} -- D: {D}"
    plt.title(f"{title} \n t: {frame_i}/{total_frames}")

    # Save frames
    file_path = f't{frame_i:04d}.png'
    filename = f'{MOVIE_FRAME_PATH}/{file_path}'
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()

def setup_opts():
    """Setup command line options for the script"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, default='default_path_value', help='Path to experiment folder')
    parser.add_argument('-r', '--fps', type=int, default=5, help='FPS for output movie')
    parser.add_argument('-s', '--stepsize', type=int, default=1, help='Step size for plotting data')
    parser.add_argument('-a', '--all', action='store_true', help='Plot all experiments in the experiments/ folder')
    return parser.parse_args()

def main(exp_path, fps, stepsize):
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

    # Setup for plotting
    total_frames = len(list(worms.values())[0]['x'])

    # Calculate total number of frames (stepsize) to be used in each movie
    calculated_stepsize = max(1, total_frames // stepsize)

    texts = ['Worm', 'Bacteria']
    legend_colors = ['Gray', 'Green']

    for frame_i in range(0, total_frames, calculated_stepsize):
            sys.stdout.write(f"\rMaking frame {frame_i}/{total_frames-1}")
            sys.stdout.flush()

            plot_frame(frame_i, worms, bacteria_sources, legend_colors, texts, script_config, convert_xy_to_index, total_frames)
    
    # Last frame
    frame_i = total_frames - 1
    sys.stdout.write(f"\rMaking frame {frame_i}/{total_frames-1}")
    sys.stdout.flush()
    plot_frame(frame_i, worms, bacteria_sources, legend_colors, texts, script_config, convert_xy_to_index, total_frames)

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
                main(BASE_EXPERIMENT_DIR, FPS, INTERVAL)
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
        
        main(BASE_EXPERIMENT_DIR, FPS, INTERVAL)
    
    print("\nDone!\n")