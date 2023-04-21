import os
import pickle
import argparse
from collections import Counter

import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10000)

    args = parser.parse_args()
    rect_size = 10
    max_figs = 2
    n = args.n

    # summetry, odds
    figs = [
        np.array([[1,1,1],[1,2,1],[1,1,1]]),
        np.array([[1,1,1,1,1],[1,2,2,2,1],[1,1,1,1,1]])
    ]


    rng = np.random.default_rng(41)

    data = []


    for i in range(n):
        sample_ind = rng.choice(range(len(figs)), max_figs, replace=True)
        sample_figs = [figs[ind] for ind in sample_ind]
        y = np.zeros((rect_size, rect_size), dtype=np.int64)
        for fig in sample_figs:
            # insert fig into episode
            h,w = fig.shape
            x_coord = rng.choice(range(rect_size - h), 1)[0]
            y_coord = rng.choice(range(rect_size - w), 1)[0]
            y[x_coord:x_coord+h,y_coord:y_coord+w] = fig

            x = y.copy()
            mask = x == 2
            x[mask] = 0     
            # # flips
            # h_flip, v_flip = rng.binomial(1, 0.5, 2)
            # if h_flip:
            #     y = np.flipud(y)
            # if v_flip:
            #     y = np.fliplr(y)
        data.append({'x': x, 'y': y})


    data_folder = './data/arc/'
    os.makedirs(data_folder, exist_ok=True)


    train, val = int(0.8 * len(data)), int(0.1 * len(data))
    test = len(data) - train - val

    train, val, test = data[:train], data[train:train+val], data[-test:]

    for fname, ds in zip(['train', 'val', 'test'], [train, val, test]):
        with open(data_folder + fname + '.pkl', 'wb') as f:
            pickle.dump(ds, f)


    joint_histogram = np.zeros((rect_size * rect_size + 1, rect_size * rect_size + 1))
    joint_histogram[-1,-1] = 100
    np.save(data_folder + '/size_distribution.npy', joint_histogram)



    atom_hist = Counter()
    aa_hist = Counter()

    for d in data:
        aa_hist.update(d['x'].flatten())
        atom_hist.update(d['y'].flatten())

    print(aa_hist)
    print(atom_hist)
