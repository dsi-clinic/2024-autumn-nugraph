#!/bin/bash
#SBATCH -J exatrkx
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --mail-type=BEGIN,END
#SBATCH --open-mode=append
#SBATCH --signal=SIGUSR1@90

# process in parallel and then merge output
ENABLE_CONNECTIONS_DEV=true
if [ "$ENABLE_CONNECTIONS_DEV" = true ]; then
    srun python process.py -i /net/projects/fermi-gnn/data_no_readwrite_permission/nugraph3.evt.h5 -o /net/projects/fermi-gnn/data_no_readwrite_permission/nugraph3_dev_output.gnn.h5 --label-vertex --connections-dev
else
    srun python process.py -i /net/projects/fermi-gnn/data_no_readwrite_permission/nugraph3.evt.h5 -o /net/projects/fermi-gnn/data_no_readwrite_permission/nugraph3_dev_output.gnn.h5 --label-vertex
fi
srun python merge.py -f /net/projects/fermi-gnn/data_no_readwrite_permission/nugraph3_dev_output.gnn.h5