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
    height, width, layers = imgs[0].shape
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    video = cv2.VideoWriter(outpath, fourcc, fps, (width, height), True)

    for img_i, img in enumerate(imgs):
        video.write(img)

    cv2.destroyAllWindows()
    video.release()

def process_data(worm_path):
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

    return worms

def plot_frame(frame_i, worms, legend_colors, texts, script_config, convert_xy_to_index, total_frames):

    # Process worm data
    for worm_key, worm_vals in worms.items():
        x = worm_vals['x'][frame_i]
        y = worm_vals['y'][frame_i]
        color = 'Gray'
        plt.scatter(convert_xy_to_index(x), convert_xy_to_index(y),
                    color=color, s=100, edgecolors='black')

    # Plot formatting
    patches = [ plt.plot([],[], marker="o", ms=10 if color_i==0 else 6, ls="", color=legend_colors[color_i],
                markeredgecolor="black", label="{:s}".format(texts[color_i]) )[0]  for color_i in range(len(texts)) ]
    plt.legend(handles=patches, bbox_to_anchor=(0.5, -0.15),
               loc='center', ncol=4, numpoints=1, labelspacing=0.3,
               fontsize='small', fancybox="True",
               handletextpad=0, columnspacing=0)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(0, 300)
    plt.ylim(0, 300)

    # Title
    N = script_config['num_worms']
    seed = int(script_config['random_seed'])
    title = f"Number of worms: {int(N)} -- Random seed: {seed}"
    plt.title(f"{title} \n t: {frame_i+1}/{total_frames}")

    # Save frames
    file_path = f't{frame_i+1:02d}.png'
    filename = f'{MOVIE_FRAME_PATH}/{file_path}'
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()

def setup_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, default='N1_seed42', help='Path to experiment folder')
    parser.add_argument('-r', '--fps', type=int, default=5, help='FPS for output movie')
    parser.add_argument('-s', '--stepsize', type=int, default=1, help='Step size for plotting data')
    return parser.parse_args()

def main(exp_path, fps, stepsize):
    # Obtain parameters from config
    script_config = read_config(exp_path)
    X_MIN = script_config['x_min']
    X_MAX = script_config['x_max']
    DX = script_config['dx']
    GRID_SIZE = np.arange(X_MIN, X_MAX+DX, DX).shape[0]
    convert_xy_to_index = lambda xy: ((xy - X_MIN) / (X_MAX - X_MIN)) * GRID_SIZE

    # Get data path
    worm_path = os.path.join(exp_path, "worm_hist.h5")

    # Obtain & process data
    worms = process_data(worm_path)

    # Setup for plotting
    total_frames = len(list(worms.values())[0]['x'])
    texts = ['Worm']
    legend_colors = ['Gray']

    for frame_i in range(0, total_frames, stepsize):
            sys.stdout.write(f"\rMaking frame {frame_i+1}/{total_frames}")
            sys.stdout.flush()

            plot_frame(frame_i, worms, legend_colors, texts, script_config, convert_xy_to_index, total_frames)
    
    # Stitching frames together to create video
    all_img_paths = np.sort(glob2.glob(f"{MOVIE_FRAME_PATH}/*.png"))
    all_imgs = np.array([cv2.imread(img) for img in all_img_paths])
    trial_name = os.path.basename(exp_path)  # Extract just 'N1_seed42'
    savepath = os.path.join(exp_path, f"{trial_name}.mp4")
    imgs2vid(all_imgs, savepath, fps)

if __name__ == '__main__':
    opts = setup_opts()

    TRIAL_PATH = opts.path
    BASE_EXPERIMENT_DIR = f"experiments/{TRIAL_PATH}"
    MOVIE_FRAME_PATH = f"{BASE_EXPERIMENT_DIR}/movie_frames"
    if os.path.exists(MOVIE_FRAME_PATH):
        shutil.rmtree(MOVIE_FRAME_PATH)
    os.makedirs(MOVIE_FRAME_PATH, exist_ok=True)
    print(BASE_EXPERIMENT_DIR)
    print(TRIAL_PATH)

    FPS = opts.fps

    INTERVAL = opts.stepsize

    print("\n---------- Visualizing worm model data ----------")
    main(BASE_EXPERIMENT_DIR, FPS, INTERVAL)
    print("\nDone!\n")