import os
import pickle

import numpy as np
import torch
from torch.nn.functional import one_hot
from torch.utils.data import Dataset


class ProcessedLigandPocketDataset(Dataset):
    def __init__(self, npz_path, center=True):

        with np.load(npz_path, allow_pickle=True) as f:
            data = {key: val for key, val in f.items()}

        # split data based on mask
        self.data = {}
        for (k, v) in data.items():
            if k == 'names':
                self.data[k] = v
                continue

            # mask = data['lig_mask'] if 'lig' in k else data['pocket_mask']
            # self.data[k] = [torch.from_numpy(v[mask == i])
            #                 for i in np.unique(mask)]
            sections = np.where(np.diff(data['lig_mask']))[0] + 1 \
                if 'lig' in k \
                else np.where(np.diff(data['pocket_mask']))[0] + 1
            self.data[k] = [torch.from_numpy(x) for x in np.split(v, sections)]

            # add number of nodes for convenience
            if k == 'lig_mask':
                self.data['num_lig_atoms'] = \
                    torch.tensor([len(x) for x in self.data['lig_mask']])
            elif k == 'pocket_mask':
                self.data['num_pocket_nodes'] = \
                    torch.tensor([len(x) for x in self.data['pocket_mask']])

        if center:
            for i in range(len(self.data['lig_coords'])):
                mean = (self.data['lig_coords'][i].sum(0) +
                        self.data['pocket_c_alpha'][i].sum(0)) / \
                       (len(self.data['lig_coords'][i]) + len(self.data['pocket_c_alpha'][i]))
                self.data['lig_coords'][i] = self.data['lig_coords'][i] - mean
                self.data['pocket_c_alpha'][i] = self.data['pocket_c_alpha'][i] - mean

    def __len__(self):
        return len(self.data['names'])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.data.items()}

    @staticmethod
    def collate_fn(batch):
        out = {}
        for prop in batch[0].keys():

            if prop == 'names':
                out[prop] = [x[prop] for x in batch]
            elif prop == 'num_lig_atoms' or prop == 'num_pocket_nodes':
                out[prop] = torch.tensor([x[prop] for x in batch])
            elif 'mask' in prop:
                # make sure indices in batch start at zero (needed for
                # torch_scatter)
                out[prop] = torch.cat([i * torch.ones(len(x[prop]))
                                       for i, x in enumerate(batch)], dim=0)
            else:
                out[prop] = torch.cat([x[prop] for x in batch], dim=0)

        return out


class ARCFillingDataset(Dataset):
    def __init__(self, path):

        with open(path, 'rb') as f:
            self.raw_data = pickle.load(f)

        categories = set()
        for datum in self.raw_data:
            categories |= set(np.unique(datum['x'].flatten()).tolist())
            categories |= set(np.unique(datum['y'].flatten()).tolist())
        
        assert max(categories) == len(categories) - 1
        self.categories = categories

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        return self.raw_data[idx]

    def collate_fn(self, batch):
        out = {}

        pocket_mask = []
        pocket_c_alpha = []
        pocket_one_hot = []
        num_pocket_nodes = []
        lig_mask = []
        lig_coords = []
        lig_one_hot = []
        num_lig_atoms = []


        for i, pair in enumerate(batch):
            x, y = pair['x'], pair['y']
            x_flat, y_flat = x.flatten(), y.flatten()

            pocket_mask.append(i * torch.ones(len(x_flat)))
            lig_mask.append(i * torch.ones(len(y_flat)))

            pocket_c_alpha.append(torch.cartesian_prod(torch.arange(0,x.shape[0]), torch.arange(0,x.shape[1])))
            lig_coords.append(torch.cartesian_prod(torch.arange(0,x.shape[0]), torch.arange(0,x.shape[1])))

            pocket_one_hot.append(one_hot(torch.tensor(x_flat), len(self.categories)))
            lig_one_hot.append(one_hot(torch.tensor(y_flat), len(self.categories)))

            num_pocket_nodes.append(len(x_flat))
            num_lig_atoms.append(len(y_flat))
        
        out['pocket_mask'] = torch.cat(pocket_mask)
        out['lig_mask'] = torch.cat(lig_mask)
        out['pocket_c_alpha'] = torch.cat(pocket_c_alpha, axis=0)
        out['lig_coords'] = torch.cat(lig_coords, axis=0)
        out['pocket_one_hot'] = torch.cat(pocket_one_hot, axis=0)
        out['lig_one_hot'] = torch.cat(lig_one_hot, axis=0)
        out['num_pocket_nodes'] = torch.tensor(num_pocket_nodes)
        out['num_lig_atoms'] = torch.tensor(num_lig_atoms)
        return out
