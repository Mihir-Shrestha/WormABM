import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_feeding_diagnostics.py path/to/environment_hist.h5 [dt]")
        sys.exit(1)

    env_path = sys.argv[1]
    dt = float(sys.argv[2]) if len(sys.argv) > 2 else None

    with h5py.File(env_path, "r") as f:
        B_before = f["B_before"][:]
        B_after = f["B_after"][:]
        dB_feed = f["dB_feed"][:]

    # Time axis: timestep index, or minutes if dt provided
    t = np.arange(len(B_before))
    if dt is not None:
        t = t * dt
        t_label = f"time (minutes, dt={dt})"
    else:
        t_label = "timestep index"

    plt.figure(figsize=(10, 6))

    # ---- Top: total bacteria before vs after feeding ----
    plt.subplot(2, 1, 1)
    plt.plot(t, B_before, label="B_before")
    plt.plot(t, B_after, label="B_after")
    plt.ylabel("Total bacteria B")
    plt.title("Total bacteria before vs after feeding")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # ---- Bottom: feeding per step (dB_feed) ----
    plt.subplot(2, 1, 2)
    plt.plot(t, dB_feed, label="dB_feed per step")
    plt.xlabel(t_label)
    plt.ylabel("Bacteria removed per step")
    plt.title("Feeding per timestep")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()