#!/bin/bash
#SBATCH -J exatrkx
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=30G
#SBATCH --open-mode=append # So that outcomes are appended, not rewritten
#SBATCH --signal=SIGUSR1@90
export NUGRAPH_DIR=/net/projects/fermi-gnn/24autumn/meghane/next_try/2024-autumn-nugraph/nugraph/nugraph
export NUGRAPH_DATA=/net/projects/fermi-gnn/24summer/helal/preprocessing3/graph.h5
export NUGRAPH_LOG=/net/projects/fermi-gnn/24autumn/meghane/next_try/2024-autumn-nugraph/checkpoints

export WANDB_API_KEY=4664589d68421aad612bfac0a95ff1d73ef2312b

srun python train.py $@