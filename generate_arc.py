import os
import pickle
import argparse
from collections import Counter

import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10000)
    parser.add_argument('--max_figs', type=int, default=2, help="Maximum number of figures to add")
    parser.add_argument('--width', type=int, default=10, help="Width of a riddle")
    parser.add_argument('--height', type=int, default=10, help="Height of a riddle")
    parser.add_argument('--flip', action='store_true', default=False, help="Whether to randomly flip figures or not")
    parser.add_argument('--no_overlap', action='store_true', default=True, help="Whether exclude overlapping of figuers")
    parser.add_argument('--folder', default='./data/arc/', help="Folder for artifacts")

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

    for i in range(n):
        n_figs = np.random.randint(1, max_figs+1)
        sample_ind = np.random.choice(range(n_figs), n_figs, replace=True)
        sample_figs = [predefined_figs[ind] for ind in sample_ind]
        if flip:
            for i, fig in enumerate(sample_figs):
                if np.random.binomial(1, 0.5):
                    sample_figs[i] = sample_figs[i].T
        space = np.zeros((height, width), dtype=np.int64)
        for fig in sample_figs:
            h, w = fig.shape
            n_attempts = 1000
            while n_attempts:
                n_attempts -= 1
                x_coord = np.random.choice(range(width - h), 1)[0]
                y_coord = np.random.choice(range(height - w), 1)[0]
                # check for overlapping
                if no_overlap:
                    if not np.all(space[x_coord:x_coord+h, y_coord:y_coord+w] == 0):
                        continue
                space[x_coord:x_coord+h, y_coord:y_coord+w] = fig
                break

            x = space.copy()
            mask = x == 2
            x[mask] = 0
        data.append({'x': x, 'y': space})

    train, val = int(0.8 * len(data)), int(0.1 * len(data))
    test = len(data) - train - val
    train, val, test = data[:train], data[train:train+val], data[-test:]

    os.makedirs(data_folder, exist_ok=True)
    for fname, ds in zip(['train', 'val', 'test'], [train, val, test]):
        with open(data_folder + fname + '.pkl', 'wb') as f:
            pickle.dump(ds, f)

    joint_histogram = np.zeros((height * width + 1, height * width + 1))
    joint_histogram[-1,-1] = 100
    np.save(data_folder + '/size_distribution.npy', joint_histogram)

    atom_hist = Counter()
    aa_hist = Counter()

    for d in data:
        aa_hist.update(d['x'].flatten())
        atom_hist.update(d['y'].flatten())

    print(aa_hist)
    print(atom_hist)
