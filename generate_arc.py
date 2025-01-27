"""Generate a one-concept dataset where the task is to fill empty rectangles.
To see examples, refer to ./colab/arc_ds_vizualization.ipynb.
"""

import os
import pathlib
import pickle
import argparse
from collections import Counter

import tqdm
import numpy as np

# Example usage:
# Small dataset for debugging:
# python generate_arc.py --n 100 --width 10 --height 10 --folder ./data/arc/small/

# Large dataset for experiments:
# python generate_arc.py --n 1_000_000 --width 15 --height 15 --max_figs 4 --flip --folder ./data/arc/1M

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ARC-style datasets.")
    parser.add_argument("--n", type=int, default=10000, help="Number of examples to generate.")
    parser.add_argument("--max_figs", type=int, default=2, help="Maximum number of figures to add.")
    parser.add_argument("--width", type=int, default=10, help="Grid width.")
    parser.add_argument("--height", type=int, default=10, help="Grid height.")
    parser.add_argument("--flip", action="store_true", default=False, help="Randomly flip figures.")
    parser.add_argument(
        "--no_overlap", action="store_true", default=True, help="Prevent figure overlap."
    )
    parser.add_argument(
        "--folder", default="./data/arc/", help="Folder to save generated dataset artifacts."
    )

    args = parser.parse_args()

    # Dataset parameters
    width = args.width
    height = args.height
    max_figs = args.max_figs
    n = args.n
    flip = args.flip
    no_overlap = args.no_overlap
    data_folder = args.folder

    # Predefined figures
    fill = 2
    predefined_figs = [
        np.array([[1, 1, 1], [1, fill, 1], [1, 1, 1]]),
        np.array([[1, 1, 1, 1, 1], [1, fill, fill, fill, 1], [1, 1, 1, 1, 1]]),
        np.array([[1, 1, 1, 1, 1, 1, 1], [1, 1, fill, fill, fill, fill, 1], [1, 1, 1, 1, 1, 1, 1]]),
        np.array([[1, 1, 1, 1, 1, 1, 1], [1, fill, fill, fill, fill, fill, 1], [1, 1, 1, 1, 1, 1, 1]]),
    ]

    np.random.seed(42)
    data = []
    # atom_hist = Counter()
    # y_hist = Counter()
    joint_hist = np.zeros((height * width + 1, height * width + 1))

    hashes = set()
    for i in tqdm.trange(n, desc="Generating data"):
        i1 = 0
        while i1 < 1000:
            i1 += 1
            # Randomly select number of figures and their types
            n_figs = np.random.randint(1, max_figs + 1)
            sample_ind = np.random.choice(len(predefined_figs), n_figs, replace=True)
            sample_figs = [predefined_figs[ind] for ind in sample_ind]

            # Randomly flip figures if specified
            if flip:
                for j, fig in enumerate(sample_figs):
                    if np.random.binomial(1, 0.5):  # 50% chance to flip
                        sample_figs[j] = fig.T

            # Create an empty grid
            y = np.zeros((height, width), dtype=np.int64)

            # Place figures on the grid
            for fig in sample_figs:
                h, w = fig.shape
                i2 = 0
                # while no_overlap
                while i2 < 1000:
                    i2 += 1
                    fig_x = np.random.randint(0, width - h + 1)
                    fig_y = np.random.randint(0, height - w + 1)
                    if no_overlap and not np.all(y[fig_x: fig_x + h, fig_y: fig_y + w] == 0):
                        continue
                    y[fig_x: fig_x + h, fig_y: fig_y + w] = fig
                    break
            if yh := str(y) in hashes:
                continue  # try new grid
            hashes.add(yh)

            x = y.copy()
            # remove fill from figures
            x[x == fill] = 0
            x_nodes = x.shape[0] * x.shape[1]
            y_nodes = np.sum(y == fill)
            joint_hist[x_nodes, y_nodes] += 1
            x_coords = np.vstack(np.where(x >= 0)).T
            y_coords = np.vstack(np.where(y == fill)).T
            #  extract only fiiled areas from y
            data.append({"x": x, "x_coords": x_coords,
                         "y": np.full(y_nodes, fill), "y_coords": y_coords})
            break

    train_size = int(0.8 * len(data))
    val_size = int(0.1 * len(data))
    test_size = len(data) - train_size - val_size

    train = data[:train_size]
    val = data[train_size: train_size + val_size]
    test = data[-test_size:]
    print(f"Counts: train={len(train)}, val={len(val)}, test={len(test)}",
          f"Sum={len(train)+len(val)+len(test)}")

    os.makedirs(data_folder, exist_ok=True)
    for fname, ds in zip(["train", "val", "test"], [train, val, test]):
        with open(pathlib.Path(data_folder, f"{fname}.pkl"), "wb") as f:
            pickle.dump(ds, f)

    print("Joint histogram", joint_hist[width*height])
    np.save(pathlib.Path(data_folder, "size_distribution.npy"), joint_hist)


    # atom_hist = Counter()
    # aa_hist = Counter()
    # for d in data:
    #     aa_hist.update(d["x"])
    #     atom_hist.update(d["y"])

    # print("Input (x) value histogram:", aa_hist)
    # print("Output (y) value histogram:", atom_hist)
    # print("Put this data into constants.py")
