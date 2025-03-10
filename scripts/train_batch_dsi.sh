#!/bin/bash
#SBATCH -J exatrkx
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mail-user=msaidenberg@uchicago.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --mem-per-cpu=16G
#SBATCH --open-mode=append # So that outcomes are appended, not rewritten
#SBATCH --signal=SIGUSR1@90

# If there are existing checkpoint files, specify ckpt_dir
ckpt_dir=/net/projects/fermi-gnn/24autumn/meghane/winter_dsi_final/checkpoints
files=$(ls -t $ckpt_dir)
latest_file=$(echo "$files" | head -n 1) # Get the first (latest) file
ckpt_path=$ckpt_dir$latest_file # Construct the full path to the most recent checkpoint file
echo $ckpt_path # This will print ckpt_path in .out file

# Add either of these two flag combinations to srun command below
# Choose based on if you have ckpt files or not
# Choice 1: --logdir /net/projects/fermi-gnn/%u --name vertex
# Choice 2: --resume $ckpt_path
srun python /net/projects/fermi-gnn/24autumn/meghane/winter_dsi_final/2024-autumn-nugraph/scripts/train.py --name pmt --version 24.7.1 --project nugraph --device 0 --data-path /net/projects/fermi-gnn/24summer/helal/preprocessing3/graph.h5 --event --semantic --filter --instance --vertex

# This is an examble submission command
# You submit through the terminal after activating numl-dsi
# sbatch /net/projects/fermi-gnn/24summer/ashmit/NuGraph/scripts/train_batch_dsi.sh --version attentional-mlp-64-sementic-filter --semantic --filter
