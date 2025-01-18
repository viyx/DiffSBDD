"""Generate one-concept dataset where the task is to fill empty rectangles."""
import os
import pathlib
import pickle
import argparse
from collections import Counter

import tqdm
import numpy as np

# small dataset, for debug purposes:
# python generate_arc.py --n 100 --width 10 --height 10 --folder ./data/arc/small/

# big dataset for experiments:
# python generate_arc.py --n 1000000 --width 15 --height 15 --max_figs 4 --flip --folder ./data/arc/1M


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10000)
    parser.add_argument('--max_figs', type=int, default=2,
                        help="Maximum number of figures to add")
    parser.add_argument('--width', type=int, default=10,
                        help="Width of a riddle")
    parser.add_argument('--height', type=int, default=10,
                        help="Height of a riddle")
    parser.add_argument('--flip', action='store_true', default=False,
                        help="Whether to randomly flip figures or not")
    parser.add_argument('--no_overlap', action='store_true', default=True,
                        help="Whether exclude overlapping of figuers")
    parser.add_argument('--folder', default='./data/arc/',
                        help="Folder for artifacts")

    args = parser.parse_args()
    width = args.width
    height = args.height
    max_figs = args.max_figs
    n = args.n
    flip = args.flip
    no_overlap = args.no_overlap
    data_folder = args.folder

    predefined_figs = [
        np.array([[1,1,1],[1,2,1],[1,1,1]]),
        np.array([[1,1,1,1,1],[1,2,2,2,1],[1,1,1,1,1]]),
        np.array([[1,1,1,1,1,1,1],[1,1,2,2,2,2,1],[1,1,1,1,1,1,1]]),
        np.array([[1,1,1,1,1,1,1],[1,2,2,2,2,2,1],[1,1,1,1,1,1,1]])
    ]

    data = []

    for i in tqdm.trange(n):
        n_figs = np.random.randint(1, max_figs+1)
        sample_ind = np.random.choice(range(n_figs), n_figs, replace=True)
        sample_figs = [predefined_figs[ind] for ind in sample_ind]
        if flip:
            for i, fig in enumerate(sample_figs):
                if np.random.binomial(1, 0.5):
                    sample_figs[i] = sample_figs[i].T
        y = np.zeros((height, width), dtype=np.int64)
        for fig in sample_figs:
            h, w = fig.shape
            n_attempts = 1000
            while n_attempts:
                n_attempts -= 1
                x_coord = np.random.choice(range(width - h), 1)[0]
                y_coord = np.random.choice(range(height - w), 1)[0]
                # check for overlapping
                if no_overlap:
                    if not np.all(y[x_coord:x_coord+h, y_coord:y_coord+w] == 0):
                        continue
                y[x_coord:x_coord+h, y_coord:y_coord+w] = fig
                break

            x = y.copy()
            mask = x == 2
            x[mask] = 0
        data.append({'x': x, 'y': y})

    train, val = int(0.8 * len(data)), int(0.1 * len(data))
    test = len(data) - train - val
    train, val, test = data[:train], data[train:train+val], data[-test:]

    os.makedirs(data_folder, exist_ok=True)
    for fname, ds in zip(['train', 'val', 'test'], [train, val, test]):
        with open(pathlib.Path(data_folder, fname+'.pkl'), 'wb') as f:
            pickle.dump(ds, f)

    joint_histogram = np.zeros((height * width + 1, height * width + 1))
    # to sample every time constant #nodes
    joint_histogram[-1, -1] = width * height
    np.save(pathlib.Path(data_folder, 'size_distribution.npy'), joint_histogram)

    atom_hist = Counter()
    aa_hist = Counter()

    for d in data:
        aa_hist.update(d['x'].flatten())
        atom_hist.update(d['y'].flatten())

    print(aa_hist)
    print(atom_hist)
    print("Put this data into constants.py")
