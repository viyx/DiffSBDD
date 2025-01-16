mamba create -n sbdd-env
mamba activate sbdd-env
mamba install pytorch cudatoolkit=10.2 -c pytorch
mamba install -c conda-forge pytorch-lightning -y
mamba install -c conda-forge wandb -y
mamba install -c conda-forge rdkit -y
mamba install -c conda-forge biopython=1.79 -y
mamba install -c conda-forge imageio -y
mamba install -c anaconda scipy -y
mamba install -c pyg pytorch-scatter -y
mamba install -c conda-forge openbabel -y
mamba install seaborn -y