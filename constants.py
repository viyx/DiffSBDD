import numpy as np
from rdkit import Chem
import torch

# ------------------------------------------------------------------------------
# Computational
# ------------------------------------------------------------------------------
FLOAT_TYPE = torch.float32
INT_TYPE = torch.int64


# ------------------------------------------------------------------------------
# Bond parameters
# ------------------------------------------------------------------------------

# margin1, margin2, margin3 = 10, 5, 3
margin1, margin2, margin3 = 3, 2, 1

allowed_bonds = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1, 'B': 3, 'Al': 3,
                 'Si': 4, 'P': [3, 5],
                 'S': 4, 'Cl': 1, 'As': 3, 'Br': 1, 'I': 1, 'Hg': [1, 2],
                 'Bi': [3, 5]}

# Bond lengths from:
# http://www.wiredchemist.com/chemistry/data/bond_energies_lengths.html
# And:
# http://chemistry-reference.com/tables/Bond%20Lengths%20and%20Enthalpies.pdf
bonds1 = {'H': {'H': 74, 'C': 109, 'N': 101, 'O': 96, 'F': 92,
                'B': 119, 'Si': 148, 'P': 144, 'As': 152, 'S': 134,
                'Cl': 127, 'Br': 141, 'I': 161},
          'C': {'H': 109, 'C': 154, 'N': 147, 'O': 143, 'F': 135,
                'Si': 185, 'P': 184, 'S': 182, 'Cl': 177, 'Br': 194,
                'I': 214},
          'N': {'H': 101, 'C': 147, 'N': 145, 'O': 140, 'F': 136,
                'Cl': 175, 'Br': 214, 'S': 168, 'I': 222, 'P': 177},
          'O': {'H': 96, 'C': 143, 'N': 140, 'O': 148, 'F': 142,
                'Br': 172, 'S': 151, 'P': 163, 'Si': 163, 'Cl': 164,
                'I': 194},
          'F': {'H': 92, 'C': 135, 'N': 136, 'O': 142, 'F': 142,
                'S': 158, 'Si': 160, 'Cl': 166, 'Br': 178, 'P': 156,
                'I': 187},
          'B': {'H':  119, 'Cl': 175},
          'Si': {'Si': 233, 'H': 148, 'C': 185, 'O': 163, 'S': 200,
                 'F': 160, 'Cl': 202, 'Br': 215, 'I': 243 },
          'Cl': {'Cl': 199, 'H': 127, 'C': 177, 'N': 175, 'O': 164,
                 'P': 203, 'S': 207, 'B': 175, 'Si': 202, 'F': 166,
                 'Br': 214},
          'S': {'H': 134, 'C': 182, 'N': 168, 'O': 151, 'S': 204,
                'F': 158, 'Cl': 207, 'Br': 225, 'Si': 200, 'P': 210,
                'I': 234},
          'Br': {'Br': 228, 'H': 141, 'C': 194, 'O': 172, 'N': 214,
                 'Si': 215, 'S': 225, 'F': 178, 'Cl': 214, 'P': 222},
          'P': {'P': 221, 'H': 144, 'C': 184, 'O': 163, 'Cl': 203,
                'S': 210, 'F': 156, 'N': 177, 'Br': 222},
          'I': {'H': 161, 'C': 214, 'Si': 243, 'N': 222, 'O': 194,
                'S': 234, 'F': 187, 'I': 266},
          'As': {'H': 152}
          }

bonds2 = {'C': {'C': 134, 'N': 129, 'O': 120, 'S': 160},
          'N': {'C': 129, 'N': 125, 'O': 121},
          'O': {'C': 120, 'N': 121, 'O': 121, 'P': 150},
          'P': {'O': 150, 'S': 186},
          'S': {'P': 186, 'C': 160}}


bonds3 = {'C': {'C': 120, 'N': 116, 'O': 113},
          'N': {'C': 116, 'N': 110},
          'O': {'C': 113}}

bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
             Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

# https://en.wikipedia.org/wiki/Covalent_radius#Radii_for_multiple_bonds
# (2022/08/14)
covalent_radii = {'H': 32, 'C': 60, 'N': 54, 'O': 53, 'F': 53, 'B': 73,
                  'Al': 111, 'Si': 102, 'P': 94, 'S': 94, 'Cl': 93, 'As': 106,
                  'Br': 109, 'I': 125, 'Hg': 133, 'Bi': 135}

# ------------------------------------------------------------------------------
# Backbone geometry
# Taken from: Bhagavan, N. V., and C. E. Ha.
# "Chapter 4-Three-dimensional structure of proteins and disorders of protein misfolding."
# Essentials of Medical Biochemistry (2015): 31-51.
# https://www.sciencedirect.com/science/article/pii/B978012416687500004X
# ------------------------------------------------------------------------------
N_CA_DIST = 1.47
CA_C_DIST = 1.53
N_CA_C_ANGLE = 110 * np.pi / 180


# ------------------------------------------------------------------------------
# Dataset-specific constants
# ------------------------------------------------------------------------------
dataset_params = {}
dataset_params['bindingmoad'] = {
    'atom_encoder': {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9},
    'atom_decoder': ['C', 'N', 'O', 'S', 'B', 'Br', 'Cl', 'P', 'I', 'F'],
    'aa_encoder': {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19},
    'aa_decoder': ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'],
    # PyMOL colors, see: https://pymolwiki.org/index.php/Color_Values#Chemical_element_colours
    'colors_dic': ['#33ff33', '#3333ff', '#ff4d4d', '#e6c540', '#ffb5b5', '#A62929', '#1FF01F', '#ff8000', '#940094', '#B3FFFF'],
    'radius_dic': [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
    'bonds1': [
        [154.0, 147.0, 143.0, 182.0, 0.0, 194.0, 177.0, 184.0, 214.0, 135.0],
        [147.0, 145.0, 140.0, 168.0, 0.0, 214.0, 175.0, 177.0, 222.0, 136.0],
        [143.0, 140.0, 148.0, 151.0, 0.0, 172.0, 164.0, 163.0, 194.0, 142.0],
        [182.0, 168.0, 151.0, 204.0, 0.0, 225.0, 207.0, 210.0, 234.0, 158.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 175.0, 0.0, 0.0, 0.0],
        [194.0, 214.0, 172.0, 225.0, 0.0, 228.0, 214.0, 222.0, 0.0, 178.0],
        [177.0, 175.0, 164.0, 207.0, 175.0, 214.0, 199.0, 203.0, 0.0, 166.0],
        [184.0, 177.0, 163.0, 210.0, 0.0, 222.0, 203.0, 221.0, 0.0, 156.0],
        [214.0, 222.0, 194.0, 234.0, 0.0, 0.0, 0.0, 0.0, 266.0, 187.0],
        [135.0, 136.0, 142.0, 158.0, 0.0, 178.0, 166.0, 156.0, 187.0, 142.0]],
    'bonds2': [[134.0, 129.0, 120.0, 160.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [129.0, 125.0, 121.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [120.0, 121.0, 121.0, 0.0, 0.0, 0.0, 0.0, 150.0, 0.0, 0.0],
               [160.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 186.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 150.0, 186.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    'bonds3': [[120.0, 116.0, 113.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [116.0, 110.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [113.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    'lennard_jones_rm': [
        [120.0, 116.0, 113.0, 160.0, 133.0, 194.0, 177.0, 184.0, 214.0, 135.0],
        [116.0, 110.0, 121.0, 168.0, 127.0, 214.0, 175.0, 177.0, 222.0, 136.0],
        [113.0, 121.0, 121.0, 151.0, 126.0, 172.0, 164.0, 150.0, 194.0, 142.0],
        [160.0, 168.0, 151.0, 204.0, 167.0, 225.0, 207.0, 186.0, 234.0, 158.0],
        [133.0, 127.0, 126.0, 167.0, 146.0, 182.0, 175.0, 167.0, 198.0, 126.0],
        [194.0, 214.0, 172.0, 225.0, 182.0, 228.0, 214.0, 222.0, 234.0, 178.0],
        [177.0, 175.0, 164.0, 207.0, 175.0, 214.0, 199.0, 203.0, 218.0, 166.0],
        [184.0, 177.0, 150.0, 186.0, 167.0, 222.0, 203.0, 221.0, 219.0, 156.0],
        [214.0, 222.0, 194.0, 234.0, 198.0, 234.0, 218.0, 219.0, 266.0, 187.0],
        [135.0, 136.0, 142.0, 158.0, 126.0, 178.0, 166.0, 156.0, 187.0, 142.0]],
    'atom_hist': {'C': 545542, 'N': 90205, 'O': 132965, 'S': 9342, 'B': 109,
                  'Br': 1424, 'Cl': 5516, 'P': 5154, 'I': 445, 'F': 9742},
    'aa_hist': {'A': 109798, 'C': 31556, 'D': 83921, 'E': 79405, 'F': 97083,
                'G': 139319, 'H': 62661, 'I': 99008, 'K': 62403, 'L': 155105,
                'M': 59977, 'N': 70437, 'P': 58833, 'Q': 48254, 'R': 74215,
                'S': 103286, 'T': 90972, 'V': 119954, 'W': 42017, 'Y': 90596},
}

dataset_params['crossdock_full'] = {
      'atom_encoder': {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9, 'others': 10},
      'atom_decoder': ['C', 'N', 'O', 'S', 'B', 'Br', 'Cl', 'P', 'I', 'F', 'others'],
      'aa_encoder': {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9, 'others': 10},
      'aa_decoder': ['C', 'N', 'O', 'S', 'B', 'Br', 'Cl', 'P', 'I', 'F', 'others'],
      'colors_dic': ['#33ff33', '#3333ff', '#ff4d4d', '#e6c540', '#ffb5b5', '#A62929', '#1FF01F', '#ff8000', '#940094', '#B3FFFF', '#ffb5b5'],
      'radius_dic': [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
      'bonds1': [[154.0, 147.0, 143.0, 182.0, 0.0, 194.0, 177.0, 184.0, 214.0, 135.0, 0.0], [147.0, 145.0, 140.0, 168.0, 0.0, 214.0, 175.0, 177.0, 222.0, 136.0, 0.0], [143.0, 140.0, 148.0, 151.0, 0.0, 172.0, 164.0, 163.0, 194.0, 142.0, 0.0], [182.0, 168.0, 151.0, 204.0, 0.0, 225.0, 207.0, 210.0, 234.0, 158.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 175.0, 0.0, 0.0, 0.0, 0.0], [194.0, 214.0, 172.0, 225.0, 0.0, 228.0, 214.0, 222.0, 0.0, 178.0, 0.0], [177.0, 175.0, 164.0, 207.0, 175.0, 214.0, 199.0, 203.0, 0.0, 166.0, 0.0], [184.0, 177.0, 163.0, 210.0, 0.0, 222.0, 203.0, 221.0, 0.0, 156.0, 0.0], [214.0, 222.0, 194.0, 234.0, 0.0, 0.0, 0.0, 0.0, 266.0, 187.0, 0.0], [135.0, 136.0, 142.0, 158.0, 0.0, 178.0, 166.0, 156.0, 187.0, 142.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
      'bonds2': [[134.0, 129.0, 120.0, 160.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [129.0, 125.0, 121.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [120.0, 121.0, 121.0, 0.0, 0.0, 0.0, 0.0, 150.0, 0.0, 0.0, 0.0], [160.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 186.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 150.0, 186.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
      'bonds3': [[120.0, 116.0, 113.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [116.0, 110.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [113.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
      'lennard_jones_rm': [[120.0, 116.0, 113.0, 160.0, 133.0, 194.0, 177.0, 184.0, 214.0, 135.0, 0.0], [116.0, 110.0, 121.0, 168.0, 127.0, 214.0, 175.0, 177.0, 222.0, 136.0, 0.0], [113.0, 121.0, 121.0, 151.0, 126.0, 172.0, 164.0, 150.0, 194.0, 142.0, 0.0], [160.0, 168.0, 151.0, 204.0, 167.0, 225.0, 207.0, 186.0, 234.0, 158.0, 0.0], [133.0, 127.0, 126.0, 167.0, 146.0, 182.0, 175.0, 167.0, 198.0, 126.0, 0.0], [194.0, 214.0, 172.0, 225.0, 182.0, 228.0, 214.0, 222.0, 234.0, 178.0, 0.0], [177.0, 175.0, 164.0, 207.0, 175.0, 214.0, 199.0, 203.0, 218.0, 166.0, 0.0], [184.0, 177.0, 150.0, 186.0, 167.0, 222.0, 203.0, 221.0, 219.0, 156.0, 0.0], [214.0, 222.0, 194.0, 234.0, 198.0, 234.0, 218.0, 219.0, 266.0, 187.0, 0.0], [135.0, 136.0, 142.0, 158.0, 126.0, 178.0, 166.0, 156.0, 187.0, 142.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
      'atom_hist': {'C': 1570767, 'N': 273858, 'O': 396837, 'S': 26352, 'B': 0, 'Br': 0, 'Cl': 15058, 'P': 25994, 'I': 0, 'F': 30687, 'others': 0},
      'aa_hist': {'C': 23302704, 'N': 6093090, 'O': 6701210, 'S': 276805, 'B': 0, 'Br': 0, 'Cl': 0, 'P': 0, 'I': 0, 'F': 0, 'others': 0},
      'x_dims': 3,
}

dataset_params['arc_filling'] = {
      'atom_encoder': {'0': 0, '1': 1, '2': 2},
      'atom_decoder': ['0', '1', '2'],
      'aa_encoder':  {'0': 0, '1': 1, '2': 2},
      'aa_decoder': ['0', '1', '2'],
      'colors_dic': [ 
            "#D3D3D3",
            "#0074D9",
            "#FF4136",
            "#2ECC40",
            "#FFDC00",
            "#AAAAAA",
            "#F011BE",
            "#FF851B",
            "#7FDBFF",
            "#870C25",],
      'radius_dic': [0.3, 0.3, 0.3, 0.3],
      # 'bonds1': [[154.0, 147.0, 143.0, 182.0, 0.0, 194.0, 177.0, 184.0, 214.0, 135.0, 0.0], [147.0, 145.0, 140.0, 168.0, 0.0, 214.0, 175.0, 177.0, 222.0, 136.0, 0.0], [143.0, 140.0, 148.0, 151.0, 0.0, 172.0, 164.0, 163.0, 194.0, 142.0, 0.0], [182.0, 168.0, 151.0, 204.0, 0.0, 225.0, 207.0, 210.0, 234.0, 158.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 175.0, 0.0, 0.0, 0.0, 0.0], [194.0, 214.0, 172.0, 225.0, 0.0, 228.0, 214.0, 222.0, 0.0, 178.0, 0.0], [177.0, 175.0, 164.0, 207.0, 175.0, 214.0, 199.0, 203.0, 0.0, 166.0, 0.0], [184.0, 177.0, 163.0, 210.0, 0.0, 222.0, 203.0, 221.0, 0.0, 156.0, 0.0], [214.0, 222.0, 194.0, 234.0, 0.0, 0.0, 0.0, 0.0, 266.0, 187.0, 0.0], [135.0, 136.0, 142.0, 158.0, 0.0, 178.0, 166.0, 156.0, 187.0, 142.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
      # 'bonds2': [[134.0, 129.0, 120.0, 160.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [129.0, 125.0, 121.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [120.0, 121.0, 121.0, 0.0, 0.0, 0.0, 0.0, 150.0, 0.0, 0.0, 0.0], [160.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 186.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 150.0, 186.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
      # 'bonds3': [[120.0, 116.0, 113.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [116.0, 110.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [113.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
      # 'lennard_jones_rm': [[120.0, 116.0, 113.0, 160.0, 133.0, 194.0, 177.0, 184.0, 214.0, 135.0, 0.0], [116.0, 110.0, 121.0, 168.0, 127.0, 214.0, 175.0, 177.0, 222.0, 136.0, 0.0], [113.0, 121.0, 121.0, 151.0, 126.0, 172.0, 164.0, 150.0, 194.0, 142.0, 0.0], [160.0, 168.0, 151.0, 204.0, 167.0, 225.0, 207.0, 186.0, 234.0, 158.0, 0.0], [133.0, 127.0, 126.0, 167.0, 146.0, 182.0, 175.0, 167.0, 198.0, 126.0, 0.0], [194.0, 214.0, 172.0, 225.0, 182.0, 228.0, 214.0, 222.0, 234.0, 178.0, 0.0], [177.0, 175.0, 164.0, 207.0, 175.0, 214.0, 199.0, 203.0, 218.0, 166.0, 0.0], [184.0, 177.0, 150.0, 186.0, 167.0, 222.0, 203.0, 221.0, 219.0, 156.0, 0.0], [214.0, 222.0, 194.0, 234.0, 198.0, 234.0, 218.0, 219.0, 266.0, 187.0, 0.0], [135.0, 136.0, 142.0, 158.0, 126.0, 178.0, 166.0, 156.0, 187.0, 142.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
      'atom_hist': {'0': 183798882, '1': 33087000, '2': 8114118},
      'aa_hist': {'0': 191913000, '1': 33087000},
      'x_dims': 2,
      'grid_width': 15,
      'grid_height': 15,
}

dataset_params['arc_small'] = {
      'atom_encoder': {'0': 0, '1': 1, '2': 2},
      'atom_decoder': ['0', '1', '2'],
      'aa_encoder':  {'0': 0, '1': 1, '2': 2},
      'aa_decoder': ['0', '1', '2'],
      'colors_dic': [ 
            "#D3D3D3",
            "#0074D9",
            "#FF4136",
            "#2ECC40",
            "#FFDC00",
            "#AAAAAA",
            "#F011BE",
            "#FF851B",
            "#7FDBFF",
            "#870C25",],
      'radius_dic': [0.3, 0.3, 0.3, 0.3],
      # 'bonds1': [[154.0, 147.0, 143.0, 182.0, 0.0, 194.0, 177.0, 184.0, 214.0, 135.0, 0.0], [147.0, 145.0, 140.0, 168.0, 0.0, 214.0, 175.0, 177.0, 222.0, 136.0, 0.0], [143.0, 140.0, 148.0, 151.0, 0.0, 172.0, 164.0, 163.0, 194.0, 142.0, 0.0], [182.0, 168.0, 151.0, 204.0, 0.0, 225.0, 207.0, 210.0, 234.0, 158.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 175.0, 0.0, 0.0, 0.0, 0.0], [194.0, 214.0, 172.0, 225.0, 0.0, 228.0, 214.0, 222.0, 0.0, 178.0, 0.0], [177.0, 175.0, 164.0, 207.0, 175.0, 214.0, 199.0, 203.0, 0.0, 166.0, 0.0], [184.0, 177.0, 163.0, 210.0, 0.0, 222.0, 203.0, 221.0, 0.0, 156.0, 0.0], [214.0, 222.0, 194.0, 234.0, 0.0, 0.0, 0.0, 0.0, 266.0, 187.0, 0.0], [135.0, 136.0, 142.0, 158.0, 0.0, 178.0, 166.0, 156.0, 187.0, 142.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
      # 'bonds2': [[134.0, 129.0, 120.0, 160.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [129.0, 125.0, 121.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [120.0, 121.0, 121.0, 0.0, 0.0, 0.0, 0.0, 150.0, 0.0, 0.0, 0.0], [160.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 186.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 150.0, 186.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
      # 'bonds3': [[120.0, 116.0, 113.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [116.0, 110.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [113.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
      # 'lennard_jones_rm': [[120.0, 116.0, 113.0, 160.0, 133.0, 194.0, 177.0, 184.0, 214.0, 135.0, 0.0], [116.0, 110.0, 121.0, 168.0, 127.0, 214.0, 175.0, 177.0, 222.0, 136.0, 0.0], [113.0, 121.0, 121.0, 151.0, 126.0, 172.0, 164.0, 150.0, 194.0, 142.0, 0.0], [160.0, 168.0, 151.0, 204.0, 167.0, 225.0, 207.0, 186.0, 234.0, 158.0, 0.0], [133.0, 127.0, 126.0, 167.0, 146.0, 182.0, 175.0, 167.0, 198.0, 126.0, 0.0], [194.0, 214.0, 172.0, 225.0, 182.0, 228.0, 214.0, 222.0, 234.0, 178.0, 0.0], [177.0, 175.0, 164.0, 207.0, 175.0, 214.0, 199.0, 203.0, 218.0, 166.0, 0.0], [184.0, 177.0, 150.0, 186.0, 167.0, 222.0, 203.0, 221.0, 219.0, 156.0, 0.0], [214.0, 222.0, 194.0, 234.0, 198.0, 234.0, 218.0, 219.0, 266.0, 187.0, 0.0], [135.0, 136.0, 142.0, 158.0, 126.0, 178.0, 166.0, 156.0, 187.0, 142.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
      'atom_hist': {'0': 8263, '1': 1472, '2': 265},
      'aa_hist': {'0': 8528, '1': 1472},
      'x_dims': 2,
      'grid_width': 10,
      'grid_height': 10,
}

dataset_params['crossdock'] = {
      'atom_encoder': {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9},
      'atom_decoder': ['C', 'N', 'O', 'S', 'B', 'Br', 'Cl', 'P', 'I', 'F'],
      'aa_encoder': {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19},
      'aa_decoder': ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'],
      # PyMOL colors, see: https://pymolwiki.org/index.php/Color_Values#Chemical_element_colours
      'colors_dic': ['#33ff33', '#3333ff', '#ff4d4d', '#e6c540', '#ffb5b5', '#A62929', '#1FF01F', '#ff8000', '#940094', '#B3FFFF'],
      'radius_dic': [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
      'bonds1': [[154.0, 147.0, 143.0, 182.0, 0.0, 194.0, 177.0, 184.0, 214.0, 135.0], [147.0, 145.0, 140.0, 168.0, 0.0, 214.0, 175.0, 177.0, 222.0, 136.0], [143.0, 140.0, 148.0, 151.0, 0.0, 172.0, 164.0, 163.0, 194.0, 142.0], [182.0, 168.0, 151.0, 204.0, 0.0, 225.0, 207.0, 210.0, 234.0, 158.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 175.0, 0.0, 0.0, 0.0], [194.0, 214.0, 172.0, 225.0, 0.0, 228.0, 214.0, 222.0, 0.0, 178.0], [177.0, 175.0, 164.0, 207.0, 175.0, 214.0, 199.0, 203.0, 0.0, 166.0], [184.0, 177.0, 163.0, 210.0, 0.0, 222.0, 203.0, 221.0, 0.0, 156.0], [214.0, 222.0, 194.0, 234.0, 0.0, 0.0, 0.0, 0.0, 266.0, 187.0], [135.0, 136.0, 142.0, 158.0, 0.0, 178.0, 166.0, 156.0, 187.0, 142.0]],
      'bonds2': [[134.0, 129.0, 120.0, 160.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [129.0, 125.0, 121.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [120.0, 121.0, 121.0, 0.0, 0.0, 0.0, 0.0, 150.0, 0.0, 0.0], [160.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 186.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 150.0, 186.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
      'bonds3': [[120.0, 116.0, 113.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [116.0, 110.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [113.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
      'lennard_jones_rm': [[120.0, 116.0, 113.0, 160.0, 133.0, 194.0, 177.0, 184.0, 214.0, 135.0], [116.0, 110.0, 121.0, 168.0, 127.0, 214.0, 175.0, 177.0, 222.0, 136.0], [113.0, 121.0, 121.0, 151.0, 126.0, 172.0, 164.0, 150.0, 194.0, 142.0], [160.0, 168.0, 151.0, 204.0, 167.0, 225.0, 207.0, 186.0, 234.0, 158.0], [133.0, 127.0, 126.0, 167.0, 146.0, 182.0, 175.0, 167.0, 198.0, 126.0], [194.0, 214.0, 172.0, 225.0, 182.0, 228.0, 214.0, 222.0, 234.0, 178.0], [177.0, 175.0, 164.0, 207.0, 175.0, 214.0, 199.0, 203.0, 218.0, 166.0], [184.0, 177.0, 150.0, 186.0, 167.0, 222.0, 203.0, 221.0, 219.0, 156.0], [214.0, 222.0, 194.0, 234.0, 198.0, 234.0, 218.0, 219.0, 266.0, 187.0], [135.0, 136.0, 142.0, 158.0, 126.0, 178.0, 166.0, 156.0, 187.0, 142.0]],
      'atom_hist': {'C': 1570032, 'N': 273792, 'O': 396623, 'S': 26339, 'B': 0, 'Br': 0, 'Cl': 15055, 'P': 25975, 'I': 0, 'F': 30673},
      'aa_hist': {'A': 277175, 'C': 92406, 'D': 254046, 'E': 201833, 'F': 234995, 'G': 376966, 'H': 147704, 'I': 290683, 'K': 173210, 'L': 421883, 'M': 157813, 'N': 174241, 'P': 148581, 'Q': 120232, 'R': 173848, 'S': 274430, 'T': 247605, 'V': 326134, 'W': 88552, 'Y': 226668},
}


dataset_params['pdbbind'] = {
      'atom_encoder': {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9},
      'atom_decoder': ['C', 'N', 'O', 'S', 'B', 'Br', 'Cl', 'P', 'I', 'F'],
      'aa_encoder': {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19},
      'aa_decoder': ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'],
      # PyMOL colors, see: https://pymolwiki.org/index.php/Color_Values#Chemical_element_colours
      'colors_dic': ['#33ff33', '#3333ff', '#ff4d4d', '#e6c540', '#ffb5b5', '#A62929', '#1FF01F', '#ff8000', '#940094', '#B3FFFF'],
      'radius_dic': [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
      'bonds1': [[154.0, 147.0, 143.0, 182.0, 0.0, 194.0, 177.0, 184.0, 214.0, 135.0], [147.0, 145.0, 140.0, 168.0, 0.0, 214.0, 175.0, 177.0, 222.0, 136.0], [143.0, 140.0, 148.0, 151.0, 0.0, 172.0, 164.0, 163.0, 194.0, 142.0], [182.0, 168.0, 151.0, 204.0, 0.0, 225.0, 207.0, 210.0, 234.0, 158.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 175.0, 0.0, 0.0, 0.0], [194.0, 214.0, 172.0, 225.0, 0.0, 228.0, 214.0, 222.0, 0.0, 178.0], [177.0, 175.0, 164.0, 207.0, 175.0, 214.0, 199.0, 203.0, 0.0, 166.0], [184.0, 177.0, 163.0, 210.0, 0.0, 222.0, 203.0, 221.0, 0.0, 156.0], [214.0, 222.0, 194.0, 234.0, 0.0, 0.0, 0.0, 0.0, 266.0, 187.0], [135.0, 136.0, 142.0, 158.0, 0.0, 178.0, 166.0, 156.0, 187.0, 142.0]],
      'bonds2': [[134.0, 129.0, 120.0, 160.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [129.0, 125.0, 121.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [120.0, 121.0, 121.0, 0.0, 0.0, 0.0, 0.0, 150.0, 0.0, 0.0], [160.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 186.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 150.0, 186.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
      'bonds3': [[120.0, 116.0, 113.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [116.0, 110.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [113.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
      'lennard_jones_rm': [[120.0, 116.0, 113.0, 160.0, 133.0, 194.0, 177.0, 184.0, 214.0, 135.0], [116.0, 110.0, 121.0, 168.0, 127.0, 214.0, 175.0, 177.0, 222.0, 136.0], [113.0, 121.0, 121.0, 151.0, 126.0, 172.0, 164.0, 150.0, 194.0, 142.0], [160.0, 168.0, 151.0, 204.0, 167.0, 225.0, 207.0, 186.0, 234.0, 158.0], [133.0, 127.0, 126.0, 167.0, 146.0, 182.0, 175.0, 167.0, 198.0, 126.0], [194.0, 214.0, 172.0, 225.0, 182.0, 228.0, 214.0, 222.0, 234.0, 178.0], [177.0, 175.0, 164.0, 207.0, 175.0, 214.0, 199.0, 203.0, 218.0, 166.0], [184.0, 177.0, 150.0, 186.0, 167.0, 222.0, 203.0, 221.0, 219.0, 156.0], [214.0, 222.0, 194.0, 234.0, 198.0, 234.0, 218.0, 219.0, 266.0, 187.0], [135.0, 136.0, 142.0, 158.0, 126.0, 178.0, 166.0, 156.0, 187.0, 142.0]],
      'atom_hist': {'C': 312417, 'N': 63669, 'O': 77484, 'S': 4873, 'B': 183, 'Br': 450, 'Cl': 2162, 'P': 1886, 'I': 132, 'F': 3661},
      'aa_hist': {'A': 44196, 'C': 13612, 'D': 37912, 'E': 32114, 'F': 37951, 'G': 59510, 'H': 24023, 'I': 44121, 'K': 31776, 'L': 67994, 'M': 21040, 'N': 28809, 'P': 23556, 'Q': 21029, 'R': 29685, 'S': 39793, 'T': 35862, 'V': 52548, 'W': 15784, 'Y': 37883},
}
