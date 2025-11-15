#!/bin/bash
#SBATCH -p batch
#SBATCH -J comet
#SBATCH -N 1
#SBATCH --gres=gpu:NVIDIAA100-PCIE-40GB:1
#SBATCH -o %j.out
#SBATCH -e %j.err

source activate py37
python test_comet.py